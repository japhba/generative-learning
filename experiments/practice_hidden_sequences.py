"""Hidden-state density learning after deterministic reconstruction is perfect.

All observed bits have two compatible hidden states. Recognition learns from
dreams or posterior inference; coupled MH removes finite-chain gradient bias.
There is no separately fitted density teacher. Direct EM is an explicit control.
"""

import argparse
from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


LENGTH, STATES = 8, 4
WORDS = np.array(list(product((0, 1), repeat=LENGTH)))
POWERS = 2**np.arange(LENGTH-1, -1, -1)
FEATURE_COUNT = 1+LENGTH+LENGTH-1
CONFIG = dict(length=LENGTH, states=STATES, examples=128, worlds=8, seed=20102691,
              rate=.025, batch=64, steps=2000, eval_every=20, exploration=.1,
              teacher_concentration=.25, teacher_floor=.02)


def sigmoid(x): return np.exp(-np.logaddexp(0, -x))


def logsumexp(x, axis=-1):
    maximum = x.max(axis=axis, keepdims=True)
    return np.squeeze(maximum+np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True)), axis=axis)


def softmax(x): return np.exp(x-logsumexp(x)[..., None])


def categorical(rng, p): return (rng.random(p.shape[:-1])[..., None] > p.cumsum(axis=-1)).sum(axis=-1)


def features(words):
    result = np.zeros((len(words), LENGTH, FEATURE_COUNT))
    result[:, :, 0] = 1
    spins = 2*words-1
    for t in range(LENGTH):
        result[:, t, 1:1+LENGTH-t] = spins[:, t:]
        result[:, t, 1+LENGTH:1+LENGTH+LENGTH-t-1] = spins[:, t:-1]*spins[:, t+1:]
    return result


def sample_generator(rng, w, count):
    p = softmax(w)
    hidden = np.zeros((count, LENGTH), dtype=int)
    hidden[:, 0] = categorical(rng, np.broadcast_to(p[STATES], (count, STATES)))
    for t in range(1, LENGTH): hidden[:, t] = categorical(rng, p[hidden[:, t-1]])
    return hidden//2, hidden


def joint_log_probability(w, hidden):
    logp = w-logsumexp(w)[:, None]
    return logp[STATES, hidden[:, 0]]+logp[hidden[:, :-1], hidden[:, 1:]].sum(axis=-1)


def generator_score(w, hidden):
    count = np.zeros((len(hidden), STATES+1, STATES))
    batch = np.arange(len(hidden))
    np.add.at(count, (batch, np.full(len(hidden), STATES), hidden[:, 0]), 1.)
    for t in range(1, LENGTH): np.add.at(count, (batch, hidden[:, t-1], hidden[:, t]), 1.)
    return count-count.sum(axis=-1, keepdims=True)*softmax(w)[None]


def forward_backward(w, words):
    p = softmax(w)
    compatible = (np.arange(STATES)[None, None]//2 == words[:, :, None])
    alpha = np.zeros((len(words), LENGTH, STATES))
    alpha[:, 0] = p[STATES]*compatible[:, 0]
    for t in range(1, LENGTH): alpha[:, t] = (alpha[:, t-1] @ p[:STATES])*compatible[:, t]
    likelihood = alpha[:, -1].sum(axis=-1)
    beta = np.ones_like(alpha)
    for t in range(LENGTH-2, -1, -1): beta[:, t] = (beta[:, t+1]*compatible[:, t+1]) @ p[:STATES].T
    counts = np.zeros((len(words), STATES+1, STATES))
    counts[:, STATES] = alpha[:, 0]*beta[:, 0]/likelihood[:, None]
    for t in range(LENGTH-1):
        counts[:, :STATES] += alpha[:, t, :, None]*p[None, :STATES]*(compatible[:, t+1]*beta[:, t+1])[:, None]/likelihood[:, None, None]
    gradient = counts-counts.sum(axis=-1, keepdims=True)*p[None]
    return np.log(likelihood), gradient, counts


def recognition_log_probability(w_rec, f, hidden):
    prior_state = np.column_stack([np.full(len(hidden), STATES), hidden[:, :-1]])
    logits = np.einsum('ntf,ntf->nt', w_rec[prior_state], f)
    return ((hidden % 2)*logits-np.logaddexp(0, logits)).sum(axis=-1)


def recognition_gradient(w_rec, f, hidden):
    prior_state = np.column_stack([np.full(len(hidden), STATES), hidden[:, :-1]])
    logits = np.einsum('ntf,ntf->nt', w_rec[prior_state], f)
    alias = hidden % 2
    residual = alias-sigmoid(logits)
    gradient = np.zeros((len(hidden), STATES+1, FEATURE_COUNT))
    for t in range(LENGTH): np.add.at(gradient, (np.arange(len(hidden)), prior_state[:, t]), residual[:, t, None]*f[:, t])
    return gradient


def proposal_log_probability(a, f, hidden):
    log_r = recognition_log_probability(a, f, hidden)
    return np.logaddexp(np.log1p(-CONFIG['exploration'])+log_r, np.log(CONFIG['exploration'])-LENGTH*np.log(2))


def sample_proposal(rng, a, words, f):
    hidden = np.zeros_like(words)
    previous = np.full(len(words), STATES)
    for t in range(LENGTH):
        alias = rng.random(len(words)) < sigmoid(np.einsum('nf,nf->n', a[previous], f[:, t]))
        hidden[:, t] = 2*words[:, t]+alias
        previous = hidden[:, t]
    uniform = rng.random(len(words)) < CONFIG['exploration']
    hidden[uniform] = 2*words[uniform]+rng.integers(2, size=(uniform.sum(), LENGTH))
    return hidden


def log_weight(w, a, f, hidden): return joint_log_probability(w, hidden)-proposal_log_probability(a, f, hidden)


def coupled_gradient(rng, w, a, words, initial):
    f = features(words)
    y = initial.copy()
    gy = generator_score(w, y)
    ry = recognition_gradient(a, f, y)
    estimate_g, estimate_r = gy.copy(), ry.copy()
    candidate = sample_proposal(rng, a, words, f)
    accepted = np.log(rng.random(len(words))) < log_weight(w, a, f, candidate)-log_weight(w, a, f, y)
    x = np.where(accepted[:, None], candidate, y)
    proposals, accepts, transitions = len(words), int(accepted.sum()), len(words)
    meeting = np.ones(len(words), dtype=int)
    while np.any(np.any(x != y, axis=-1)):
        active = np.flatnonzero(np.any(x != y, axis=-1))
        fa, wa = f[active], words[active]
        estimate_g[active] += generator_score(w, x[active])-generator_score(w, y[active])
        estimate_r[active] += recognition_gradient(a, fa, x[active])-recognition_gradient(a, fa, y[active])
        candidate = sample_proposal(rng, a, wa, fa)
        log_u = np.log(rng.random(len(active)))
        proposed_weight = log_weight(w, a, fa, candidate)
        ax = log_u < proposed_weight-log_weight(w, a, fa, x[active])
        ay = log_u < proposed_weight-log_weight(w, a, fa, y[active])
        x[active] = np.where(ax[:, None], candidate, x[active])
        y[active] = np.where(ay[:, None], candidate, y[active])
        proposals += len(active); transitions += 2*len(active); accepts += int(ax.sum()+ay.sum())
        meeting[active] += 1
        assert meeting.max() < 20000, 'Failed to couple; do not truncate the unbiased estimator.'
    info = dict(proposals=proposals, acceptance=accepts/transitions, meeting=float(meeting.mean()),
                max_meeting=int(meeting.max()), variance=float(estimate_g.var(axis=0).sum()),
                component_variance=estimate_g.var(axis=0), recognition_variance=estimate_r.var(axis=0))
    return estimate_g.mean(axis=0), estimate_r.mean(axis=0), x, info


def count_fit(hidden):
    counts = np.full((STATES+1, STATES), .5)
    np.add.at(counts, (np.full(len(hidden), STATES), hidden[:, 0]), 1)
    for t in range(1, LENGTH): np.add.at(counts, (hidden[:, t-1], hidden[:, t]), 1)
    logits = np.log(counts)
    return logits-logits.mean(axis=-1, keepdims=True)


def viterbi(w, words):
    logp = w-logsumexp(w)[:, None]
    compatible = np.arange(STATES)[None, None]//2 == words[:, :, None]
    value = logp[STATES]+np.where(compatible[:, 0], 0, -np.inf)
    back = []
    for t in range(1, LENGTH):
        edge = value[:, :, None]+logp[None, :STATES]
        back.append(edge.argmax(axis=1))
        value = edge.max(axis=1)+np.where(compatible[:, t], 0, -np.inf)
    hidden = np.zeros_like(words)
    hidden[:, -1] = value.argmax(axis=-1)
    for t in range(LENGTH-2, -1, -1): hidden[:, t] = back[t][np.arange(len(words)), hidden[:, t+1]]
    return hidden


def make_world(rng):
    p = rng.dirichlet(np.full(STATES, CONFIG['teacher_concentration']), STATES+1)
    p = (1-CONFIG['teacher_floor'])*p+CONFIG['teacher_floor']/STATES
    truth = np.log(p)
    data, hidden = sample_generator(rng, truth, CONFIG['examples'])
    log_true = forward_backward(truth, WORDS)[0]
    assert abs(np.exp(log_true).sum()-1) < 1e-12
    return dict(truth=truth, data=data, hidden_truth_for_audit=hidden, log_true=log_true)


def evaluate(w, a, world):
    log_q, _, _ = forward_backward(w, WORDS)
    # Fixed-seed Monte Carlo measures posterior recognition under the current
    # generator. Density and data-likelihood metrics are exact enumerations.
    x, state_paths = sample_generator(np.random.default_rng(20102692), w, 8192)
    log_joint = joint_log_probability(w, state_paths)
    log_r = recognition_log_probability(a, features(x), state_paths)
    rkl = np.mean(log_joint-log_q[x @ POWERS]-log_r)
    log_true = world['log_true']
    return np.array([-log_q[world['data'] @ POWERS].mean(),
                     np.exp(log_true) @ (log_true-log_q), rkl, -log_r.mean()])


def fit(rng, w, a, world, condition, steps):
    cache = 2*world['data']+rng.integers(2, size=world['data'].shape)
    records, times, costs, weights, recognition_weights, diagnostics = [], [], [], [], [], []
    total_cost = 0
    status = 'step cap'
    for step in tqdm(range(steps+1), desc=condition, leave=False):
        if step % CONFIG['eval_every'] == 0:
            record = evaluate(w, a, world)
            assert np.isfinite(record).all(), (condition, step, record)
            records.append(record); times.append(step); costs.append(total_cost)
            weights.append(w.copy()); recognition_weights.append(a.copy())
            if step >= 600:
                # Both stopping criteria are available internally. The true
                # world's predictive KL is a diagnostic, never a decision.
                losses = np.array(records[-20:])[:, [0, 3]]
                slopes = np.polyfit(np.log(np.array(times[-20:])+1), np.log(losses+1e-8), 1)[0]
                if np.max(np.abs(slopes)) < .01: status = 'stationary log-loss/log-step slopes'; break
            assert record[0] < records[0][0]+.8, 'Clearly degraded training likelihood; redesign instead of extending.'
        if step == steps: break
        indices = rng.choice(len(world['data']), CONFIG['batch'], replace=False)
        observed = world['data'][indices]
        dream_x, dream_z = sample_generator(rng, w, CONFIG['batch'])
        dream_gradient = recognition_gradient(a, features(dream_x), dream_z).mean(axis=0)
        info = dict(proposals=0, acceptance=0., meeting=0., variance=0.)
        if condition == 'exact':
            dw = forward_backward(w, world['data'])[1].mean(axis=0)
            da = dream_gradient
        else:
            dw, wake_gradient, latent, info = coupled_gradient(rng, w, a, observed, cache[indices])
            cache[indices] = latent
            if condition == 'sleep': da = dream_gradient
            elif condition == 'wake': da = wake_gradient
            elif condition == 'frozen': da = np.zeros_like(a)
            else: raise ValueError(condition)
        rate = CONFIG['rate']/(1+step/500)**.6
        w += rate*dw
        w -= w.mean(axis=-1, keepdims=True)
        a += rate*da
        total_cost += info['proposals']+CONFIG['batch']
        diagnostics.append([info[k] for k in ['proposals', 'acceptance', 'meeting', 'variance']])
        assert np.isfinite(w).all() and np.isfinite(a).all()
    return dict(records=np.array(records), steps=np.array(times), costs=np.array(costs), w=np.array(weights),
                a=np.array(recognition_weights), diagnostics=np.array(diagnostics), status=np.array(status))


def ngram_log_probability(data, order):
    probability = np.ones(len(WORDS))
    for t in range(LENGTH):
        context_length = min(order, t)
        powers = 2**np.arange(context_length-1, -1, -1)
        counts = np.full((2**context_length, 2), .5)
        # Pool positions with equal available context length.
        positions = range(order, LENGTH) if t >= order else [t]
        for u in positions:
            context = data[:, u-context_length:u] @ powers
            np.add.at(counts, (context, data[:, u]), 1)
        counts /= counts.sum(axis=-1, keepdims=True)
        context = WORDS[:, t-context_length:t] @ powers
        probability *= counts[context, WORDS[:, t]]
    assert abs(probability.sum()-1) < 1e-12
    return np.log(probability)


def direct_controls(w, world):
    result = {}
    truth = np.exp(world['log_true'])
    for order in [0, 1, 2, 3]:
        logp = ngram_log_probability(world['data'], order)
        result[f'order_{order}'] = [float(-logp[world['data'] @ POWERS].mean()), float(truth @ (world['log_true']-logp))]
    trajectories = {}
    for method in ['hard_em', 'em']:
        current, records, matrices = w.copy(), [], []
        for step in range(201):
            logp = forward_backward(current, WORDS)[0]
            records.append([-logp[world['data'] @ POWERS].mean(), truth @ (world['log_true']-logp)])
            matrices.append(current.copy())
            if step == 200: break
            if method == 'hard_em': updated = count_fit(viterbi(current, world['data']))
            else:
                counts = forward_backward(current, world['data'])[2].sum(axis=0)
                updated = np.log(counts/counts.sum(axis=-1, keepdims=True))
                updated -= updated.mean(axis=-1, keepdims=True)
            if np.max(np.abs(updated-current)) < 1e-8: break
            current = updated
        trajectories[method] = dict(records=np.array(records), w=np.array(matrices))
        result[method] = records[-1]
        result[method+'_best_true'] = float(np.array(records)[:, 1].min())
    return result, trajectories


def plot(output, count):
    fig, axes = plt.subplots(2, 1, layout='constrained')
    for condition, label in [('sleep', 'Recognition from dreams'), ('wake', 'Recognition from old-data posteriors'), ('frozen', 'Frozen recognition'), ('exact', 'Exact likelihood gradient')]:
        runs = [np.load(output/f'{condition}_{i}.npz') for i in range(count)]
        end = max(r['steps'][-1] for r in runs)
        grid = np.arange(0, end+1, CONFIG['eval_every'])
        for m, ax in enumerate(axes):
            v = np.array([np.interp(grid, r['steps'], r['records'][:, m]) for r in runs])
            ax.plot(grid, v.mean(axis=0), label=label)
    summary = json.loads((output/'summary.json').read_text())
    for index, (name, label) in enumerate([('order_1', 'Direct bigram fit'), ('order_3', 'Direct order-3 fit'), ('em', 'Direct EM')]):
        for m, ax in enumerate(axes): ax.axhline(np.mean([s['controls'][name][m] for s in summary]), linestyle=':', color=f'C{4+index}', label=label)
    axes[0].set(ylabel='Observed-data NLL (nats)')
    axes[1].set(ylabel='True predictive KL (nats)', xlabel='Offline update budget; stopped models held fixed')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2)
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-hidden-sequences-learning.png')
    plt.close(fig)


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_hidden_sequences/internal_stopping')
    parser.add_argument('--worlds', type=int, default=CONFIG['worlds'])
    parser.add_argument('--steps', type=int, default=CONFIG['steps'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(uk_start=datetime.now(ZoneInfo('Europe/London')).isoformat(), config=CONFIG,
                    worlds=args.worlds, steps=args.steps, status='Exploratory screen, not a full-match claim.',
                    stopping='After 600 updates, absolute double-log slopes below .01 for observed-data NLL and self-generated recognition cross entropy over 20 evaluations. True predictive risk is never used to stop or select a model.',
                    amendment='Same worlds and seeds as the pilot, correcting a stopping rule that inspected true predictive risk. The pilot is preserved and is not independent confirmation.',
                    premise='All proposed paths reconstruct old observations exactly. Learn transitions and a causal recognizer in tandem; no fixed learned density teacher. Test dream recognition against unbiased old-data posterior learning, frozen recognition, exact likelihood gradients, n-gram fits, hard EM and exact EM.')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, args.output/'source.py')
    summary = []
    for i in tqdm(range(args.worlds), desc='Hidden sequence worlds'):
        rng = np.random.default_rng(CONFIG['seed']+i)
        world = make_world(rng)
        # One arbitrary valid explanation per observation. Its deterministic
        # reconstruction error is already zero; this is only an initialization.
        w = count_fit(2*world['data']+rng.integers(2, size=world['data'].shape))
        a = np.zeros((STATES+1, FEATURE_COUNT))
        np.savez_compressed(args.output/f'world_{i}.npz', **world, initial_w=w, initial_a=a)
        controls, curves = direct_controls(w, world)
        for name, values in curves.items(): np.savez_compressed(args.output/f'{name}_{i}.npz', **values)
        row = dict(world=i, initial=evaluate(w, a, world).tolist(), controls=controls)
        for condition in ['sleep', 'wake', 'frozen', 'exact']:
            trajectory = fit(np.random.default_rng(CONFIG['seed']+10000+i), w.copy(), a.copy(), world, condition, args.steps)
            np.savez_compressed(args.output/f'{condition}_{i}.npz', **trajectory)
            row[condition] = dict(final=trajectory['records'][-1].tolist(), status=str(trajectory['status']),
                                  steps=int(trajectory['steps'][-1]), proposals=int(trajectory['costs'][-1]))
        summary.append(row)
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(args.output, args.worlds)
    print(json.dumps(dict(output=str(args.output), initial=np.mean([s['initial'] for s in summary], axis=0).tolist(),
                          final={c: np.mean([s[c]['final'] for s in summary], axis=0).tolist() for c in ['sleep', 'wake', 'frozen', 'exact']},
                          controls={c: np.mean([s['controls'][c] for s in summary], axis=0).tolist() for c in summary[0]['controls']}), indent=2))


if __name__ == '__main__': main()
