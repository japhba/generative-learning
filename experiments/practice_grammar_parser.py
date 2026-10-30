"""Learned causal parsing of a recursive grammar using self-generated trees.

Coupled independence Metropolis estimates remove finite-chain bias from the
generator's observed-likelihood gradient. Recognition learns from dreams,
from posterior parses of old observations, or stays frozen.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_recursive_grammar_screen as grammar
from practice_recursive_grammar_analysis import one_step


L = grammar.LENGTH
NODES = 2*L-1
WORDS = grammar.WORDS
SPANS = [(i, j) for width in range(1, L+1) for i in range(L-width+1) for j in [i+width]]
SPAN_INDEX = np.full((L+1, L+1), -1, dtype=int)
for index, (i, j) in enumerate(SPANS): SPAN_INDEX[i, j] = index
INTERNAL = [s for s, (i, j) in enumerate(SPANS) if j-i > 1]
INTERNAL_INDEX = np.full(len(SPANS), -1, dtype=int)
INTERNAL_INDEX[INTERNAL] = np.arange(len(INTERNAL))
PAIRS = [(i, j) for i in range(L) for j in range(i+1, L)]
F = 1+(L-1)+L+len(PAIRS)
CHOICES = (L-1)*4
CHOICE_SPLIT = np.repeat(np.arange(1, L), 4)
CHOICE_LEFT = np.tile(np.repeat([0, 1], 2), L-1)
CHOICE_RIGHT = np.tile([0, 1], 2*(L-1))
CONFIG = dict(worlds=4, seed=20102900, examples=128, batch=64, steps=1500, rate=.1,
              eval_every=25, minimum_steps=500, stationary_slope=.005,
              uniform_mass=.02, exploration=.1, maximum_meeting=20000)


def logsumexp(x, axis=-1):
    peak = x.max(axis=axis, keepdims=True)
    return np.squeeze(peak+np.log(np.exp(x-peak).sum(axis=axis, keepdims=True)), axis=axis)


def categorical(rng, probabilities): return (rng.random(probabilities.shape[:-1])[..., None] > probabilities.cumsum(axis=-1)).sum(axis=-1)


def features():
    phi = np.zeros((len(WORDS), len(SPANS), F))
    for s in INTERNAL:
        i, j = SPANS[s]; width = j-i
        phi[:, s, 0] = 1
        phi[:, s, width-1] = 1
        phi[:, s, L:L+width] = 2*WORDS[:, i:j]-1
        for k, (u, v) in enumerate(PAIRS):
            if v < width: phi[:, s, 2*L+k] = (2*WORDS[:, i+u]-1)*(2*WORDS[:, i+v]-1)
        phi[:, s] /= np.sqrt(2+width+width*(width-1)/2)
    return phi


PHI = features()
LEFT_SPAN = np.zeros((len(INTERNAL), CHOICES), dtype=int)
RIGHT_SPAN = np.zeros_like(LEFT_SPAN)
BASE = np.full((len(WORDS), len(INTERNAL), CHOICES), -np.inf)
for internal, s in enumerate(INTERNAL):
    i, j = SPANS[s]; width = j-i
    for c in range(CHOICES):
        split = CHOICE_SPLIT[c]
        if split >= width: continue
        k = i+split
        LEFT_SPAN[internal, c] = SPAN_INDEX[i, k]
        RIGHT_SPAN[internal, c] = SPAN_INDEX[k, j]
        compatible = np.ones(len(WORDS), dtype=bool)
        if split == 1: compatible &= WORDS[:, i] == CHOICE_LEFT[c]
        if width-split == 1: compatible &= WORDS[:, j-1] == CHOICE_RIGHT[c]
        # These uniform inside-message offsets ensure that zero recognition
        # weights give uniform tree shapes AND independent internal labels.
        value = np.log(grammar.CATALAN[split]*grammar.CATALAN[width-split]/grammar.CATALAN[width])
        value -= (split if split > 1 else 0)*np.log(2)
        value -= (width-split if width-split > 1 else 0)*np.log(2)
        BASE[compatible, internal, c] = value


def unpack_recognition(a): return a[:2], a[2:10].reshape(2, 4), a[10:].reshape(2, F)


def recognition_tables(a):
    root, rules, messages = unpack_recognition(a)
    values = PHI @ messages.T
    root_logits = values+root
    root_log = root_logits-logsumexp(root_logits)[..., None]
    left = values[:, LEFT_SPAN, CHOICE_LEFT[None]]
    right = values[:, RIGHT_SPAN, CHOICE_RIGHT[None]]
    logits = BASE[:, :, None]+left[:, :, None]+right[:, :, None]+rules[None, None, :, np.arange(CHOICES) % 4]
    local_log = logits-logsumexp(logits)[..., None]
    return dict(root=root_log, local=local_log)


UNIFORM_TABLES = recognition_tables(np.zeros(10+2*F))


def generator_probabilities(w):
    mass = CONFIG['uniform_mass']
    root = mass/2+(1-mass)*grammar.softmax(w[:2])
    rules = mass/4+(1-mass)*grammar.softmax(w[2:].reshape(2, 4))
    return root, rules.reshape(2, 2, 2)


def empty_tree(count, length=L):
    tree = {name: np.zeros((count, 2*length-1), dtype=int) for name in ['state', 'split', 'start', 'length']}
    tree['length'][:, 0] = length
    return tree


def subset(tree, indices): return {key: value[indices].copy() for key, value in tree.items()}


def assign(destination, indices, source):
    for key in destination: destination[key][indices] = source[key]


def set_children(tree, rows, node, split, left, right):
    start, width = tree['start'][rows, node], tree['length'][rows, node]
    li, ri = np.full(len(rows), node+1), node+2*split
    tree['split'][rows, node] = split
    tree['state'][rows, li] = left; tree['state'][rows, ri] = right
    tree['start'][rows, li] = start; tree['start'][rows, ri] = start+split
    tree['length'][rows, li] = split; tree['length'][rows, ri] = width-split


def sample_recognition(rng, tables, word_ids, length=L):
    tree = empty_tree(len(word_ids), length)
    tree['state'][:, 0] = categorical(rng, np.exp(tables['root'][word_ids, SPAN_INDEX[0, length]]))
    for node in range(2*length-1):
        rows = np.flatnonzero(tree['length'][:, node] > 1)
        start, width = tree['start'][rows, node], tree['length'][rows, node]
        spans = INTERNAL_INDEX[SPAN_INDEX[start, start+width]]
        choice = categorical(rng, np.exp(tables['local'][word_ids[rows], spans, tree['state'][rows, node]]))
        set_children(tree, rows, node, CHOICE_SPLIT[choice], CHOICE_LEFT[choice], CHOICE_RIGHT[choice])
    return tree


def sample_generator(rng, w, count, length=L):
    root, rules = generator_probabilities(w)
    tree = empty_tree(count, length)
    tree['state'][:, 0] = categorical(rng, np.broadcast_to(root, (count, 2)))
    words = np.zeros((count, L), dtype=int)
    for node in range(2*length-1):
        leaves = np.flatnonzero(tree['length'][:, node] == 1)
        words[leaves, tree['start'][leaves, node]] = tree['state'][leaves, node]
        rows = np.flatnonzero(tree['length'][:, node] > 1)
        width = tree['length'][rows, node]
        splits = np.arange(1, L)[None]
        valid = splits < width[:, None]
        remaining = np.maximum(width[:, None]-splits, 0)
        probability = np.where(valid, grammar.CATALAN[splits]*grammar.CATALAN[remaining]/grammar.CATALAN[width, None], 0.)
        split = categorical(rng, probability)+1
        pair = categorical(rng, rules.reshape(2, 4)[tree['state'][rows, node]])
        set_children(tree, rows, node, split, pair//2, pair % 2)
    return words @ grammar.POWERS, tree


def events(tree):
    rows, node = np.nonzero(tree['length'] > 1)
    split = tree['split'][rows, node]
    start, width = tree['start'][rows, node], tree['length'][rows, node]
    parent = tree['state'][rows, node]
    left, right = tree['state'][rows, node+1], tree['state'][rows, node+2*split]
    choice = (split-1)*4+2*left+right
    spans = INTERNAL_INDEX[SPAN_INDEX[start, start+width]]
    return rows, parent, choice, spans


def tree_counts(tree):
    count = len(tree['state'])
    root = np.eye(2)[tree['state'][:, 0]]
    rules = np.zeros((count, 2, 4))
    rows, parent, choice, _ = events(tree)
    np.add.at(rules, (rows, parent, choice % 4), 1.)
    return root, rules


def generator_scores_from_counts(w, root_counts, rule_counts):
    root, rules = generator_probabilities(w)
    signal_root, signal_rule = grammar.softmax(w[:2]), grammar.softmax(w[2:].reshape(2, 4))
    rc = root_counts*(1-CONFIG['uniform_mass']/2/root)
    cc = rule_counts*(1-CONFIG['uniform_mass']/4/rules.reshape(2, 4))
    root_score = rc-rc.sum(axis=-1, keepdims=True)*signal_root
    rule_score = cc-cc.sum(axis=-1, keepdims=True)*signal_rule
    return np.concatenate([root_score, rule_score.reshape(len(rc), 8)], axis=-1)


def generator_score(w, tree): return generator_scores_from_counts(w, *tree_counts(tree))


def log_joint(w, tree):
    root, rules = generator_probabilities(w)
    rc, cc = tree_counts(tree)
    length = tree['length'][:, 0]
    return rc @ np.log(root)+np.einsum('nab,ab->n', cc, np.log(rules.reshape(2, 4)))-np.log(grammar.CATALAN[length])


def recognition_log_probability(tables, word_ids, tree):
    root_span = SPAN_INDEX[0, tree['length'][:, 0]]
    value = tables['root'][word_ids, root_span, tree['state'][:, 0]].copy()
    rows, parent, choice, spans = events(tree)
    np.add.at(value, rows, tables['local'][word_ids[rows], spans, parent, choice])
    return value


def recognition_score(tables, word_ids, tree):
    count = len(word_ids)
    root_span = SPAN_INDEX[0, tree['length'][:, 0]]
    root_score = np.eye(2)[tree['state'][:, 0]]-np.exp(tables['root'][word_ids, root_span])
    message_score = root_score[:, :, None]*PHI[word_ids, root_span, None]
    rule_score = np.zeros((count, 2, 4))
    rows, parent, choice, spans = events(tree)
    residual = -np.exp(tables['local'][word_ids[rows], spans, parent])
    residual[np.arange(len(rows)), choice] += 1
    np.add.at(rule_score, (rows, parent), residual.reshape(len(rows), L-1, 4).sum(axis=1))
    left_phi = PHI[word_ids[rows, None], LEFT_SPAN[spans]]
    right_phi = PHI[word_ids[rows, None], RIGHT_SPAN[spans]]
    for state in range(2):
        gradient = np.einsum('nc,ncf->nf', residual*(CHOICE_LEFT == state), left_phi)
        gradient += np.einsum('nc,ncf->nf', residual*(CHOICE_RIGHT == state), right_phi)
        np.add.at(message_score[:, state], rows, gradient)
    return np.concatenate([root_score, rule_score.reshape(count, 8), message_score.reshape(count, 2*F)], axis=-1)


def sample_proposal(rng, tables, word_ids, length=L):
    tree = sample_recognition(rng, tables, word_ids, length)
    rows = np.flatnonzero(rng.random(len(word_ids)) < CONFIG['exploration'])
    assign(tree, rows, sample_recognition(rng, UNIFORM_TABLES, word_ids[rows], length))
    return tree


def log_weight(w, tables, word_ids, tree):
    length = tree['length'][:, 0]
    uniform_log = -np.log(grammar.CATALAN[length])-(length-1)*np.log(2)
    recognition_log = recognition_log_probability(tables, word_ids, tree)
    mixture_log = np.logaddexp(np.log1p(-CONFIG['exploration'])+recognition_log, np.log(CONFIG['exploration'])+uniform_log)
    return log_joint(w, tree)-mixture_log


def different(x, y): return np.any((x['state'] != y['state']) | (x['split'] != y['split']), axis=-1)


def coupled_gradient(rng, w, tables, word_ids, initial, recognition_update):
    count, length = len(word_ids), int(initial['length'][0, 0])
    y = subset(initial, np.arange(count))
    estimate_g = generator_score(w, y)
    estimate_r = recognition_score(tables, word_ids, y) if recognition_update else np.zeros((count, 10+2*F))
    candidate = sample_proposal(rng, tables, word_ids, length)
    accepted = np.log(rng.random(count)) < log_weight(w, tables, word_ids, candidate)-log_weight(w, tables, word_ids, y)
    x = subset(y, np.arange(count))
    assign(x, accepted, subset(candidate, accepted))
    proposals, accepts, transitions = count, int(accepted.sum()), count
    meeting = np.ones(count, dtype=int)
    while different(x, y).any():
        rows = np.flatnonzero(different(x, y))
        xa, ya = subset(x, rows), subset(y, rows)
        ids = word_ids[rows]
        estimate_g[rows] += generator_score(w, xa)-generator_score(w, ya)
        if recognition_update: estimate_r[rows] += recognition_score(tables, ids, xa)-recognition_score(tables, ids, ya)
        candidate = sample_proposal(rng, tables, ids, length)
        log_u = np.log(rng.random(len(rows)))
        proposed_weight = log_weight(w, tables, ids, candidate)
        ax = log_u < proposed_weight-log_weight(w, tables, ids, xa)
        ay = log_u < proposed_weight-log_weight(w, tables, ids, ya)
        assign(x, rows[ax], subset(candidate, ax)); assign(y, rows[ay], subset(candidate, ay))
        proposals += len(rows); accepts += int(ax.sum()+ay.sum()); transitions += 2*len(rows)
        meeting[rows] += 1
        assert meeting.max() < CONFIG['maximum_meeting'], 'Coupling failed; do not truncate the estimator.'
    diagnostics = dict(proposals=proposals, acceptance=accepts/transitions, meeting=float(meeting.mean()),
                       maximum_meeting=int(meeting.max()), gradient_variance=float(estimate_g.var(axis=0).sum()))
    return estimate_g, estimate_r, x, diagnostics


def exact_recognition(w, tables, word_ids, length=L):
    root, rules = generator_probabilities(w)
    words = WORDS[word_ids, :length]
    probability, inside = grammar.inside(root, rules, words)
    outside = np.zeros_like(inside)
    outside[:, 0, length] = root[None]/probability[:, None]
    roots = inside[:, 0, length]*root/probability[:, None]
    counts = np.zeros((len(words), 2, 2, 2))
    log_r = np.sum(roots*tables['root'][word_ids, SPAN_INDEX[0, length]], axis=-1)
    for width in range(length, 1, -1):
        for i in range(length-width+1):
            j = i+width
            span = INTERNAL_INDEX[SPAN_INDEX[i, j]]
            for k in range(i+1, j):
                split = grammar.CATALAN[k-i]*grammar.CATALAN[j-k]/grammar.CATALAN[width]
                parent, left, right = outside[:, i, j], inside[:, i, k], inside[:, k, j]
                joint = split*np.einsum('na,abc,nb,nc->nabc', parent, rules, left, right)
                counts += joint
                logs = tables['local'][word_ids, span, :, 4*(k-i-1):4*(k-i)].reshape(len(words), 2, 2, 2)
                # Incompatible leaf events have zero posterior mass. Their
                # contribution to the expectation is defined to be zero.
                log_r += np.sum(joint*np.where(joint > 0, logs, 0.), axis=(1, 2, 3))
                outside[:, i, k] += split*np.einsum('na,abc,nc->nb', parent, rules, right)
                outside[:, k, j] += split*np.einsum('na,abc,nb->nc', parent, rules, left)
    log_g = roots @ np.log(root)+np.einsum('nabc,abc->n', counts, np.log(rules))-np.log(grammar.CATALAN[length])
    entropy = np.log(probability)-log_g
    return probability, -log_r-entropy, -log_r, roots, counts


def evaluate(w, tables, world):
    p, kl_r, ce_r, _, _ = exact_recognition(w, tables, np.arange(len(WORDS)))
    assert abs(p.sum()-1) < 1e-10 and kl_r.min() > -1e-10
    truth = world['truth']
    return np.array([-np.log(p[world['data'] @ grammar.POWERS]).mean(), truth @ (np.log(truth)-np.log(p)),
                     p @ kl_r, p @ ce_r, kl_r[world['data'] @ grammar.POWERS].mean()])


def fit(rng, w, a, world, condition, steps):
    word_ids = world['data'] @ grammar.POWERS
    cache = sample_recognition(rng, UNIFORM_TABLES, word_ids)
    records, times, costs, wall_seconds, weights, recognizers, diagnostics = [], [], [], [], [], [], []
    cumulative, started = 0, time.monotonic()
    status = 'step cap'
    progress = tqdm(range(steps+1), desc=condition, leave=False)
    for step in progress:
        tables = recognition_tables(a)
        if step % CONFIG['eval_every'] == 0:
            record = evaluate(w, tables, world)
            assert np.isfinite(record).all()
            records.append(record); times.append(step); costs.append(cumulative); wall_seconds.append(time.monotonic()-started)
            weights.append(w.copy()); recognizers.append(a.copy())
            progress.set_postfix(nll=round(record[0], 4), test_kl=round(record[1], 4), recognition_kl=round(record[2], 4), proposals=cumulative)
            assert record[0] < records[0][0]+.5, 'Broken likelihood trajectory: redesign rather than extending.'
            if step >= CONFIG['minimum_steps']:
                losses = np.array(records[-20:])[:, [0, 3]]
                slopes = np.polyfit(np.log(np.array(times[-20:])+1), np.log(losses), 1)[0]
                if np.max(np.abs(slopes)) < CONFIG['stationary_slope']: status = 'stationary internal likelihood and dream recognition loss'; break
        if step == steps: break
        indices = rng.choice(len(word_ids), CONFIG['batch'], replace=False)
        observed = word_ids[indices]
        if condition in ['sleep', 'exact']:
            dream_ids, dream_tree = sample_generator(rng, w, CONFIG['batch'])
            dream_gradient = recognition_score(tables, dream_ids, dream_tree).mean(axis=0)
        if condition == 'exact':
            root, rules = generator_probabilities(w)
            _, rc, cc, _ = grammar.posterior_counts(root, rules, world['data'])
            dw = generator_scores_from_counts(w, rc, cc.reshape(len(rc), 2, 4)).mean(axis=0)
            da = dream_gradient
            info = dict(proposals=0, acceptance=0., meeting=0., maximum_meeting=0, gradient_variance=0.)
        else:
            dg, dr, latent, info = coupled_gradient(rng, w, tables, observed, subset(cache, indices), condition == 'wake')
            assign(cache, indices, latent)
            dw = dg.mean(axis=0)
            if condition == 'sleep': da = dream_gradient
            elif condition == 'wake': da = dr.mean(axis=0)
            elif condition == 'frozen': da = np.zeros_like(a)
            else: raise ValueError(condition)
        rate = CONFIG['rate']/(1+step/500)**.6
        w += rate*dw
        w[:2] -= w[:2].mean(); w[2:].reshape(2, 4)[:] -= w[2:].reshape(2, 4).mean(axis=-1, keepdims=True)
        a += rate*da
        ar, ac, _ = unpack_recognition(a)
        ar -= ar.mean(); ac -= ac.mean(axis=-1, keepdims=True)
        assert np.isfinite(w).all() and np.isfinite(a).all()
        cumulative += info['proposals']+(CONFIG['batch'] if condition in ['sleep', 'exact'] else 0)
        diagnostics.append([info[k] for k in ['proposals', 'acceptance', 'meeting', 'maximum_meeting', 'gradient_variance']])
    return dict(records=np.array(records), steps=np.array(times), costs=np.array(costs), wall_seconds=np.array(wall_seconds),
                w=np.array(weights), a=np.array(recognizers), diagnostics=np.array(diagnostics), status=np.array(status))


def plot(output, conditions):
    rows = json.loads((output/'summary.json').read_text())
    fig, axes = plt.subplots(2, 1, layout='constrained')
    for condition in conditions:
        runs = [np.load(output/f'{condition}_{i}.npz') for i in range(len(rows))]
        grid = np.arange(0, max(r['steps'][-1] for r in runs)+1, CONFIG['eval_every'])
        for column, ax in zip([1, 2], axes):
            ax.plot(grid, np.mean([np.interp(grid, r['steps'], r['records'][:, column]) for r in runs], axis=0), label=condition)
    axes[0].axhline(np.mean([r['direct']['cv_selected'][1] for r in rows]), linestyle=':', color='black', label='CV-selected direct fit')
    axes[0].set(ylabel='True predictive KL (nats)')
    axes[1].set(ylabel='Causal posterior recognition KL', xlabel='Updates; stopped models held fixed')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3)
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-grammar-parser-learning.png')
    plt.close(fig)


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_grammar_parser/pilot')
    parser.add_argument('--worlds', type=int, default=CONFIG['worlds'])
    parser.add_argument('--steps', type=int, default=CONFIG['steps'])
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--conditions', nargs='+', default=['sleep', 'wake', 'frozen', 'exact'])
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(uk_start=datetime.now(ZoneInfo('Europe/London')).isoformat(), config=CONFIG,
        worlds=args.worlds, steps=args.steps, seed=args.seed, conditions=args.conditions,
        objective='Test an actual learned causal parser and own-generation loop in the recursive grammar that passed a direct-fit screen.',
        primary='Sleep recognition versus frozen and wake recognition for held-out observation generation, with identical generator gradients in expectation.',
        controls='Same generator/recognizer rate and schedule; one analytic grammar step, CV-selected direct counts, exact likelihood-gradient reference.',
        stopping='Only training likelihood and internally available dream recognition cross entropy; never teacher risk.',
        caveats='Known two-percent noise floor. Finite-step stochastic improvements are empirical; mean-field likelihood improvement is the theoretical guarantee. Exact dynamic programming remains an alternative.'), indent=2)+'\n')
    shutil.copy2(__file__, args.output/'source.py'); shutil.copy2(grammar.__file__, args.output/'grammar_source.py')
    rows = []
    for world_index in tqdm(range(args.worlds), desc='Learned grammar parser'):
        rng = np.random.default_rng(args.seed+world_index)
        root = .98*rng.dirichlet(np.full(2, .35))+.01
        rules = (.98*rng.dirichlet(np.full(4, .35), 2)+.005).reshape(2, 2, 2)
        truth = grammar.inside(root, rules, WORDS)[0]
        data = grammar.sample(rng, root, rules, CONFIG['examples'])[0]
        world = dict(root=root, rules=rules, truth=truth, data=data)
        direct = grammar.direct_fits(data)
        choice, cv = grammar.crossvalidated_direct(data, rng)
        direct['cv_selected'] = direct[choice]
        direct_scores = {name: [float(-np.log(p[data @ grammar.POWERS]).mean()), float(truth @ (np.log(truth)-np.log(p)))] for name, p in direct.items()}
        direct_scores['one_step_grammar'] = one_step(data, truth)['scores']
        means = np.array([1-data.mean(), data.mean()])
        initial = np.concatenate([np.log(means), np.tile(np.log(means[:, None]*means[None]).reshape(4), 2)])+rng.normal(0, .3, 10)
        initial_a = np.zeros(10+2*F)
        np.savez_compressed(args.output/f'world_{world_index}.npz', **world, initial=initial, initial_a=initial_a)
        row = dict(world=world_index, direct=direct_scores, cv_choice=choice, cv_scores=cv, conditions={})
        for condition in args.conditions:
            run = fit(np.random.default_rng(args.seed+10000+world_index), initial.copy(), initial_a.copy(), world, condition, args.steps)
            np.savez_compressed(args.output/f'{condition}_{world_index}.npz', **run)
            row['conditions'][condition] = dict(final=run['records'][-1].tolist(), steps=int(run['steps'][-1]), cost=int(run['costs'][-1]),
                seconds=float(run['wall_seconds'][-1]), status=str(run['status']))
            (args.output/'partial_summary.json').write_text(json.dumps(rows+[row], indent=2)+'\n')
        rows.append(row)
        (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    if not args.no_plot: plot(args.output, args.conditions)
    print(json.dumps(dict(output=str(args.output), conditions={c: np.mean([r['conditions'][c]['final'] for r in rows], axis=0).tolist() for c in args.conditions}), indent=2))


if __name__ == '__main__': main()
