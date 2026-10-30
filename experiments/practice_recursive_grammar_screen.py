"""Early shortcut screen for an ambiguous recursive binary grammar.

This deliberately uses exact inside-outside EM, not a learned practice loop.
It tests whether a hierarchy is worth developing before training a recognizer.
"""

import argparse
from datetime import datetime
from itertools import product
import json
from math import comb
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


LENGTH = 8
WORDS = np.array(list(product((0, 1), repeat=LENGTH)))
POWERS = 2**np.arange(LENGTH-1, -1, -1)
CATALAN = np.array([1.]+[comb(2*(n-1), n-1)/n for n in range(1, LENGTH+1)])
CONFIG = dict(length=LENGTH, examples=128, worlds=8, seed=20102717,
              starts=4, em_iterations=240, evaluation_every=4,
              concentration=.35, uniform_mass=.02)


def softmax(x):
    p = np.exp(x-x.max(axis=-1, keepdims=True))
    return p/p.sum(axis=-1, keepdims=True)


def inside(root, rules, words):
    n = words.shape[1]
    messages = np.zeros((len(words), n+1, n+1, 2))
    for i in range(n): messages[:, i, i+1] = np.eye(2)[words[:, i]]
    for width in range(2, n+1):
        for i in range(n-width+1):
            j = i+width
            for k in range(i+1, j):
                split = CATALAN[k-i]*CATALAN[j-k]/CATALAN[width]
                messages[:, i, j] += split*np.einsum('abc,nb,nc->na', rules, messages[:, i, k], messages[:, k, j])
    probability = messages[:, 0, n] @ root
    return probability, messages


def posterior_counts(root, rules, words):
    probability, messages = inside(root, rules, words)
    n = words.shape[1]
    outside = np.zeros_like(messages)
    outside[:, 0, n] = root[None]/probability[:, None]
    counts = np.zeros((len(words), 2, 2, 2))
    for width in range(n, 1, -1):
        for i in range(n-width+1):
            j = i+width
            for k in range(i+1, j):
                split = CATALAN[k-i]*CATALAN[j-k]/CATALAN[width]
                parent, left, right = outside[:, i, j], messages[:, i, k], messages[:, k, j]
                counts += split*np.einsum('na,abc,nb,nc->nabc', parent, rules, left, right)
                outside[:, i, k] += split*np.einsum('na,abc,nc->nb', parent, rules, right)
                outside[:, k, j] += split*np.einsum('na,abc,nb->nc', parent, rules, left)
    root_counts = messages[:, 0, n]*root[None]/probability[:, None]
    log_joint = root_counts @ np.log(root)+np.einsum('nabc,abc->n', counts, np.log(rules))-np.log(CATALAN[n])
    entropy = np.log(probability)-log_joint
    return probability, root_counts, counts, entropy


def sample(rng, root, rules, count, length=LENGTH):
    words, root_labels, all_counts = [], [], []
    for _ in range(count):
        root_state = int(rng.choice(2, p=root))
        counts = np.zeros((2, 2, 2))
        def expand(state, length):
            if length == 1: return [state]
            split_p = CATALAN[1:length]*CATALAN[length-1:0:-1]/CATALAN[length]
            left_length = int(rng.choice(np.arange(1, length), p=split_p))
            pair = int(rng.choice(4, p=rules[state].reshape(4)))
            left, right = divmod(pair, 2)
            counts[state, left, right] += 1
            return expand(left, left_length)+expand(right, length-left_length)
        words.append(expand(root_state, length)); root_labels.append(root_state); all_counts.append(counts)
    return np.array(words), np.array(root_labels), np.array(all_counts)


def ngram(data, order):
    result = np.ones(len(WORDS))
    for t in range(LENGTH):
        size = min(order, t)
        powers = 2**np.arange(size-1, -1, -1)
        counts = np.full((2**size, 2), .5)
        positions = range(order, LENGTH) if t >= order else [t]
        for u in positions: np.add.at(counts, (data[:, u-size:u] @ powers, data[:, u]), 1)
        counts /= counts.sum(axis=-1, keepdims=True)
        result *= counts[WORDS[:, t-size:t] @ powers, WORDS[:, t]]
    return result


def direct_fits(data):
    result = {f'order_{order}': ngram(data, order) for order in range(4)}
    means = (data.sum(axis=0)+.5)/(len(data)+1)
    result['position_means'] = np.prod(np.where(WORDS, means, 1-means), axis=-1)
    counts = np.bincount(data.sum(axis=-1), minlength=LENGTH+1)+.5
    counts /= counts.sum()
    result['bit_count'] = counts[WORDS.sum(axis=-1)]/np.array([comb(LENGTH, k) for k in WORDS.sum(axis=-1)])
    word_counts = np.bincount(data @ POWERS, minlength=len(WORDS))
    for alpha in [.05, .5]: result[f'word_count_{alpha}'] = (word_counts+alpha)/(len(data)+alpha*len(WORDS))
    # Chow--Liu: a maximum-weight spanning tree of pair mutual informations,
    # followed by normalized empirical conditionals. No iterative model fit.
    pairs = np.zeros((LENGTH, LENGTH, 2, 2))
    information = np.zeros((LENGTH, LENGTH))
    for i in range(LENGTH):
        for j in range(i+1, LENGTH):
            joint = np.full((2, 2), .5)
            np.add.at(joint, (data[:, i], data[:, j]), 1)
            pairs[i, j] = joint; pairs[j, i] = joint.T
            joint /= joint.sum()
            information[i, j] = information[j, i] = np.sum(joint*np.log(joint/(joint.sum(axis=0)[None]*joint.sum(axis=1)[:, None])))
    seen, edges = [0], []
    while len(seen) < LENGTH:
        unseen = [j for j in range(LENGTH) if j not in seen]
        values = information[np.ix_(seen, unseen)]
        i, j = np.unravel_index(values.argmax(), values.shape)
        parent, child = seen[i], unseen[j]
        edges.append((parent, child)); seen.append(child)
    tree_probability = np.where(WORDS[:, 0], means[0], 1-means[0])
    for parent, child in edges:
        conditional = pairs[parent, child]/pairs[parent, child].sum(axis=-1, keepdims=True)
        tree_probability *= conditional[WORDS[:, parent], WORDS[:, child]]
    result['chow_liu'] = tree_probability
    for name, probability in result.items():
        assert np.all(probability > 0), name
        assert abs(probability.sum()-1) < 1e-12, (name, probability.sum())
    return result


def crossvalidated_direct(data, rng):
    shuffled = rng.permutation(len(data))
    scores = {name: 0. for name in direct_fits(data)}
    for test in np.array_split(shuffled, 4):
        train = np.setdiff1d(np.arange(len(data)), test)
        for name, probability in direct_fits(data[train]).items(): scores[name] -= np.log(probability[data[test] @ POWERS]).sum()
    return min(scores, key=scores.__getitem__), scores


def constrained_counts(counts, floor):
    """Exact categorical M step with a fixed lower bound on each probability."""
    fixed = np.zeros_like(counts, dtype=bool)
    for _ in range(counts.shape[-1]):
        free = np.where(fixed, 0., counts)
        probability = floor*fixed+(1-floor*fixed.sum(axis=-1, keepdims=True))*free/free.sum(axis=-1, keepdims=True)
        below = probability < floor
        if not below.any(): return probability
        fixed |= below
    raise AssertionError('No feasible categorical M step')


def em_fit(root, rules, data, truth, uniform_floor_mass=0.):
    root = constrained_counts(root, uniform_floor_mass/2)
    rules = constrained_counts(rules.reshape(2, 4), uniform_floor_mass/4).reshape(2, 2, 2)
    records, roots, rule_history, steps = [], [], [], []
    status = 'iteration cap'
    for iteration in range(CONFIG['em_iterations']+1):
        p_data, root_counts, counts, entropy = posterior_counts(root, rules, data)
        nll = -np.log(p_data).mean()
        if iteration % CONFIG['evaluation_every'] == 0:
            p_all = inside(root, rules, WORDS)[0]
            assert abs(p_all.sum()-1) < 1e-10
            kl = truth @ (np.log(truth)-np.log(p_all))
            records.append([nll, kl, entropy.mean()]); roots.append(root.copy()); rule_history.append(rules.copy()); steps.append(iteration)
            assert np.isfinite(records[-1]).all()
            if len(records) > 1: assert nll-records[-2][0] < 1e-10
            if iteration >= 80:
                slopes = np.polyfit(np.log(np.array(steps[-20:])+1), np.log(np.array(records[-20:])[:, [0, 2]]), 1)[0]
                if np.max(np.abs(slopes)) < .005: status = 'stationary internal likelihood and posterior entropy'; break
        if iteration == CONFIG['em_iterations']: break
        root = constrained_counts(root_counts.sum(axis=0), uniform_floor_mass/2)
        counts = counts.sum(axis=0)
        rules = constrained_counts(counts.reshape(2, 4), uniform_floor_mass/4).reshape(2, 2, 2)
    return dict(records=np.array(records), roots=np.array(roots), rules=np.array(rule_history), steps=np.array(steps), status=np.array(status))


def plot(output, count):
    rows = json.loads((output/'summary.json').read_text())
    chosen = [np.load(output/f'em_{i}_{rows[i]["chosen_start"]}.npz') for i in range(count)]
    grid = np.arange(0, max(r['steps'][-1] for r in chosen)+1, CONFIG['evaluation_every'])
    fig, axes = plt.subplots(2, 1, layout='constrained')
    for m, ax in enumerate(axes):
        ax.plot(grid, np.mean([np.interp(grid, r['steps'], r['records'][:, m]) for r in chosen], axis=0), label='Grammar EM, training-selected start')
        for index, (name, label) in enumerate([('order_3', 'Direct order-three counts'), ('bit_count', 'Direct bit-count distribution'), ('chow_liu', 'Direct correlation tree'), ('cv_selected', 'CV-selected direct fit')]):
            ax.axhline(np.mean([r['direct_scores'][name][m] for r in rows]), linestyle=':', color=f'C{index+1}', label=label)
    axes[0].set(ylabel='Observed-data NLL (nats)')
    axes[1].set(ylabel='True predictive KL (nats)', xlabel='EM iterations; stopped models held fixed')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2)
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-recursive-grammar-screen.png')
    plt.close(fig)


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_recursive_grammar/screen')
    parser.add_argument('--worlds', type=int, default=CONFIG['worlds'])
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--known-noise-floor', action='store_true', help='Give the learner the teacher family\'s two-percent uniform floor; this is additional prior knowledge.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(uk_start=datetime.now(ZoneInfo('Europe/London')).isoformat(), config=CONFIG, worlds=args.worlds, seed=args.seed, known_noise_floor=args.known_noise_floor,
        purpose='Pre-practice gate: test whether a two-category ambiguous recursive grammar can beat direct observed-data counts. Exact EM is a reference only; no learned recognizer or self-practice claim.',
        controls='Orders 0--3, position means, count-of-ones distribution, smoothed whole-word counts, Chow--Liu tree, and four-fold CV selection among them.',
        selection='Choose among four EM starts by final training likelihood only. All stopping uses training NLL and latent posterior entropy; true risk is diagnostic.'), indent=2)+'\n')
    shutil.copy2(__file__, args.output/'source.py')
    summary = []
    for i in tqdm(range(args.worlds), desc='Recursive grammar shortcut screen'):
        rng = np.random.default_rng(args.seed+i)
        root = rng.dirichlet(np.full(2, CONFIG['concentration']))
        rules = rng.dirichlet(np.full(4, CONFIG['concentration']), 2).reshape(2, 2, 2)
        root = (1-CONFIG['uniform_mass'])*root+CONFIG['uniform_mass']/2
        rules = (1-CONFIG['uniform_mass'])*rules+CONFIG['uniform_mass']/4
        truth = inside(root, rules, WORDS)[0]
        assert abs(truth.sum()-1) < 1e-12
        data, root_labels, counts = sample(rng, root, rules, CONFIG['examples'])
        np.savez_compressed(args.output/f'world_{i}.npz', root=root, rules=rules, truth=truth, data=data, sampled_root_labels_for_audit=root_labels, sampled_rule_counts_for_audit=counts)
        direct = direct_fits(data)
        selected, cv = crossvalidated_direct(data, rng)
        direct['cv_selected'] = direct[selected]
        direct_scores = {name: [float(-np.log(p[data @ POWERS]).mean()), float(truth @ (np.log(truth)-np.log(p)))] for name, p in direct.items()}
        means = np.array([1-data.mean(), data.mean()])
        starts = []
        for start in tqdm(range(CONFIG['starts']), desc='EM starts', leave=False):
            initial_root = softmax(np.log(means)+rng.normal(0, .3, 2))
            base = np.log(means[:, None]*means[None]).reshape(4)
            initial_rules = softmax(base[None]+rng.normal(0, .3, (2, 4))).reshape(2, 2, 2)
            run = em_fit(initial_root, initial_rules, data, truth, CONFIG['uniform_mass'] if args.known_noise_floor else 0.)
            np.savez_compressed(args.output/f'em_{i}_{start}.npz', **run)
            starts.append(dict(start=start, final=run['records'][-1].tolist(), status=str(run['status']), steps=int(run['steps'][-1])))
        chosen = min(range(len(starts)), key=lambda j: starts[j]['final'][0])
        row = dict(world=i, chosen_start=chosen, grammar=starts[chosen]['final'], starts=starts, direct_scores=direct_scores, cv_choice=selected, cv_scores=cv)
        summary.append(row)
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(args.output, args.worlds)
    print(json.dumps(dict(output=str(args.output), grammar_mean=np.mean([r['grammar'] for r in summary], axis=0).tolist(),
                          direct_means={k: np.mean([r['direct_scores'][k] for r in summary], axis=0).tolist() for k in summary[0]['direct_scores']}), indent=2))


if __name__ == '__main__': main()
