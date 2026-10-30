"""Finite-data density learning with causal recognition and coupled-MH dreams.

An exploratory mechanism screen, not a claimed solution to the README.
The retained Bayesian density is a strong additional source of knowledge.
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


D, K, DELTA = 8, 4, .1
STATES = np.array(list(product((0., 1.), repeat=D)))
FEATURES = np.column_stack([STATES, np.ones(len(STATES))])
POWERS = 2**np.arange(D-1, -1, -1)
CONFIG = dict(dimensions=D, causes=K, hypotheses=16, clauses=12, observations=24,
              inverse_temperature=2., uniform_proposal_fraction=DELTA, rate=.4,
              batch=128, initial_steps=2400, practice_steps=1600, eval_every=20,
              logit_limit=6., seed=20102673, worlds=8)


def sigmoid(x): return np.exp(-np.logaddexp(0, -x))


def logsumexp(x, axis=-1):
    maximum = x.max(axis=axis, keepdims=True)
    return np.squeeze(maximum+np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True)), axis=axis)


def softmax(x): return np.exp(x-logsumexp(x)[..., None])


def categorical(rng, probability):
    return (rng.random(probability.shape[:-1])[..., None] > probability.cumsum(axis=-1)).sum(axis=-1)


def make_world(rng):
    clauses = np.array([rng.choice(D, 3, replace=False) for _ in range(CONFIG['clauses'])])
    signs = np.broadcast_to(rng.integers(2, size=(CONFIG['clauses'], 3)), (16, CONFIG['clauses'], 3)).copy()
    signs[:, np.arange(4), 0] ^= np.array(list(product((0, 1), repeat=4)))
    satisfied = (STATES[None, :, clauses] == signs[:, None]).any(axis=-1)
    log_world = -CONFIG['inverse_temperature']*(~satisfied).sum(axis=-1)
    log_world -= logsumexp(log_world)[:, None]
    truth = int(rng.integers(16))
    data = rng.choice(len(STATES), CONFIG['observations'], p=np.exp(log_world[truth]))
    posterior = softmax(log_world[:, data].sum(axis=-1))
    target = posterior @ np.exp(log_world)
    empirical = np.bincount(data, minlength=len(STATES))/len(data)
    return dict(clauses=clauses, signs=signs, log_world=log_world, truth=truth,
                data=data, posterior=posterior, target=target, empirical=empirical)


def tables(w, a):
    probability = sigmoid(w)
    log_joint = STATES @ w.T-np.logaddexp(0, w).sum(axis=-1)-np.log(K)
    log_q = logsumexp(log_joint)
    posterior = np.exp(log_joint-log_q[:, None])
    score = posterior[:, :, None]*(STATES[:, None]-probability[None])
    recognition = softmax(FEATURES @ a.T)
    proposal = (1-DELTA)*np.exp(log_joint)+DELTA/(K*len(STATES))
    return dict(probability=probability, log_joint=log_joint, log_q=log_q,
                posterior=posterior, score=score, recognition=recognition, proposal=proposal)


def sample_generator(rng, w, count):
    z = rng.integers(K, size=count)
    x = ((rng.random((count, D)) < sigmoid(w)[z]) @ POWERS).astype(int)
    return x, z


def sample_proposal(rng, w, count):
    x, z = sample_generator(rng, w, count)
    uniform = rng.random(count) < DELTA
    x[uniform] = rng.integers(len(STATES), size=uniform.sum())
    z[uniform] = rng.integers(K, size=uniform.sum())
    return x, z


def transition(rng, current, candidate, latent, uniform, refresh_uniform, table, target):
    cause = (refresh_uniform[:, None] > table['recognition'][current].cumsum(axis=-1)).sum(axis=-1)
    ratio = target[:, None]*table['recognition']/table['proposal']
    accept = uniform < np.minimum(1., ratio[candidate, latent]/ratio[current, cause])
    return np.where(accept, candidate, current), accept


def coupled_gradient(rng, w, table, target, data, batch):
    # X_0 and Y_0 have equal laws (indeed identical values). The lag-one
    # correction is unbiased conditional on this initialization, not burn-in.
    y = rng.choice(data, batch)
    estimate = table['score'][y].copy()
    candidate, latent = sample_proposal(rng, w, batch)
    x, accepted = transition(rng, y, candidate, latent, rng.random(batch), rng.random(batch), table, target)
    proposals, accepts, transitions, time = batch, int(accepted.sum()), batch, 1
    meeting = np.ones(batch, dtype=int)
    while np.any(x != y):
        active = np.flatnonzero(x != y)
        estimate[active] += table['score'][x[active]]-table['score'][y[active]]
        candidate, latent = sample_proposal(rng, w, len(active))
        uniform, refresh = rng.random(len(active)), rng.random(len(active))
        x[active], ax = transition(rng, x[active], candidate, latent, uniform, refresh, table, target)
        y[active], ay = transition(rng, y[active], candidate, latent, uniform, refresh, table, target)
        proposals += len(active); transitions += 2*len(active); accepts += int(ax.sum()+ay.sum())
        meeting[active] += 1
        time += 1
        assert time < 20000, 'Coupling did not meet; abort rather than truncate the estimator.'
    return estimate.mean(axis=0), dict(proposals=proposals, transitions=transitions,
                                      acceptance=accepts/transitions, mean_meeting=float(meeting.mean()),
                                      max_meeting=int(meeting.max()), gradient_variance=float(estimate.var(axis=0).sum()),
                                      component_variance=estimate.var(axis=0))


def recognition_gradient(table, x, z):
    return (np.eye(K)[z]-table['recognition'][x]).T @ FEATURES[x]/len(x)


def evaluate(w, a, world):
    t = tables(w, a)
    p, truth = world['target'], np.exp(world['log_world'][int(world['truth'])])
    rkl = np.sum(np.exp(t['log_joint'])*(np.log(t['posterior'])-np.log(t['recognition'])))
    return np.array([p @ (np.log(p)-t['log_q']), -world['empirical'] @ t['log_q'],
                     rkl, truth @ (np.log(truth)-t['log_q'])])


def stationary(records, times, condition):
    if times[-1] < 600: return False
    values = np.array(records[-20:])
    losses = values[:, [1, 0]] if condition == 'original' else values[:, [0, 2]]
    slopes = np.polyfit(np.log(np.array(times[-20:])+1), np.log(losses+1e-8), 1)[0]
    return bool(np.max(np.abs(slopes)) < .015)


def fit(rng, w, a, world, condition, steps, batch):
    records, times, costs, diagnostics, weights, recognition_weights = [], [], [], [], [], []
    proposals = 0
    status = 'step cap'
    for step in tqdm(range(steps+1), desc=condition, leave=False):
        t = tables(w, a)
        if step % CONFIG['eval_every'] == 0:
            value = evaluate(w, a, world)
            assert np.isfinite(value).all(), (condition, step, value)
            records.append(value); times.append(step); costs.append(proposals)
            weights.append(w.copy()); recognition_weights.append(a.copy())
            if stationary(records, times, condition):
                status = 'stationary double-log slopes'; break
        if step == steps: break
        x, z = sample_generator(rng, w, batch)
        da = recognition_gradient(t, x, z)
        info = dict(proposals=batch, transitions=0, acceptance=0., mean_meeting=0., max_meeting=0, gradient_variance=0.)
        if condition in ['joint', 'frozen', 'collapsed']:
            sampling_table = t if condition != 'collapsed' else t | {'recognition': t['proposal']/t['proposal'].sum(axis=-1, keepdims=True)}
            dw, info = coupled_gradient(rng, w, sampling_table, world['target'], world['data'], batch)
            info['proposals'] += batch  # Equal recognition-training sample cost in both conditions.
        elif condition == 'original':
            dw = np.einsum('s,skd->kd', world['empirical'], t['score'])
            da = (world['empirical'][:, None]*(t['posterior']-t['recognition'])).T @ FEATURES
            info['proposals'] = 0
        elif condition == 'oracle': dw = np.einsum('s,skd->kd', world['target'], t['score'])
        elif condition == 'self': dw = t['score'][x].mean(axis=0)
        else: raise ValueError(condition)
        if condition == 'frozen': da *= 0
        rate = CONFIG['rate'] if condition in ['original', 'oracle'] else CONFIG['rate']/(1+step/400)**.6
        w = np.clip(w+rate*dw, -CONFIG['logit_limit'], CONFIG['logit_limit'])
        a += rate*da
        assert np.isfinite(w).all() and np.isfinite(a).all()
        proposals += info['proposals']
        diagnostics.append([info[key] for key in ['proposals', 'transitions', 'acceptance', 'mean_meeting', 'max_meeting', 'gradient_variance']])
    return dict(records=np.array(records), steps=np.array(times), costs=np.array(costs),
                diagnostics=np.array(diagnostics), w=np.array(weights), a=np.array(recognition_weights), status=np.array(status))


def analytic_baselines(world):
    result = {}
    for name, p in [('empirical', world['empirical']), ('predictive', world['target'])]:
        mean = p @ STATES
        covariance = np.einsum('s,si,sj->ij', p, STATES-mean, STATES-mean)
        covariance -= np.diag(np.diag(covariance))
        values, vectors = np.linalg.eigh(covariance)
        loadings = vectors[:, -2:]*np.sqrt(np.maximum(values[-2:], 0))
        smoothed_mean = (len(world['data'])*mean+.5)/(len(world['data'])+1) if name == 'empirical' else mean
        for variant in ['independent', 'spectral']:
            probabilities = np.tile(smoothed_mean, (K, 1))
            if variant == 'spectral': probabilities += np.array(list(product((-1, 1), repeat=2))) @ loadings.T
            probabilities = np.clip(probabilities, sigmoid(-CONFIG['logit_limit']), sigmoid(CONFIG['logit_limit']))
            w = np.log(probabilities)-np.log1p(-probabilities)
            result[name+'_'+variant] = float(evaluate(w, np.zeros((K, D+1)), world)[0])
    return result


def run(output, count, seed, batch, steps):
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(uk_start=datetime.now(ZoneInfo('Europe/London')).isoformat(), config=CONFIG,
                    worlds=count, seed=seed, batch=batch, steps=steps,
                    purpose='Exploratory density screen. Exact Bayesian target from a supplied finite soft-clause prior; learned recognition infers generator causes. Coupled auxiliary MH gives unbiased predictive-likelihood gradients. Test original-data early stopping and frozen recognition; do not claim a full README match.',
                    stationarity='After 600 steps, absolute log-loss/log-step slopes below .015 for both tracked losses over the last 20 evaluations.',
                    fixed_target_limitation='The predictive density remains a frozen source of judgement; learning auxiliary recognition need not be essential.')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, output/'source.py')
    summaries = []
    for index in tqdm(range(count), desc='Finite-data density worlds'):
        rng = np.random.default_rng(seed+index)
        world = make_world(rng)
        np.savez_compressed(output/f'world_{index}.npz', **world)
        initial = fit(rng, rng.normal(0, .35, (K, D)), np.zeros((K, D+1)), world, 'original', CONFIG['initial_steps'], batch)
        np.savez_compressed(output/f'initial_{index}.npz', **initial)
        result = dict(world=index, initial=initial['records'][-1].tolist(),
                      initial_best_predictive=float(initial['records'][:, 0].min()),
                      initial_status=str(initial['status']), posterior_max=float(world['posterior'].max()),
                      analytic_baselines=analytic_baselines(world))
        for condition in ['joint', 'frozen', 'collapsed', 'original', 'self', 'oracle']:
            run_rng = np.random.default_rng(seed+10000+index)
            trajectory = fit(run_rng, initial['w'][-1].copy(), initial['a'][-1].copy(), world, condition, steps, batch)
            np.savez_compressed(output/f'{condition}_{index}.npz', **trajectory)
            result[condition] = dict(final=trajectory['records'][-1].tolist(), status=str(trajectory['status']),
                                     steps=int(trajectory['steps'][-1]), proposals=int(trajectory['costs'][-1]))
        summaries.append(result)
        (output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    return summaries


def plot(output, count):
    fig, ax = plt.subplots(layout='constrained')
    diagnostic_fig, diagnostic_axes = plt.subplots(2, 1, layout='constrained')
    labels = dict(joint='Learned recognition', frozen='Frozen recognition', collapsed='Marginal MH', original='Original-data fit', self='Uncorrected dreams', oracle='Exact predictive gradient')
    for condition, label in labels.items():
        runs = [np.load(output/f'{condition}_{i}.npz') for i in range(count)]
        end = min(r['steps'][-1] for r in runs)
        grid = np.arange(0, end+1, CONFIG['eval_every'])
        values = np.array([[np.interp(grid, r['steps'], r['records'][:, m]) for m in [0, 1, 2]] for r in runs])
        ax.plot(grid, values[:, 0].mean(axis=0), label=label)
        if condition in ['joint', 'frozen', 'collapsed']:
            diagnostic_axes[0].plot(grid, values[:, 2].mean(axis=0), label=label)
            budget = min(r['costs'][-1] for r in runs)
            common = np.linspace(0, budget, 100)
            v = np.array([np.interp(common, r['costs'], r['records'][:, 0]) for r in runs])
            diagnostic_axes[1].plot(common, v.mean(axis=0), label=label)
    summary = json.loads((output/'summary.json').read_text())
    best = np.mean([s['initial_best_predictive'] for s in summary])
    ax.axhline(best, linestyle=':', color='black', label='Best original-data checkpoint')
    ax.axhline(np.mean([s['analytic_baselines']['predictive_spectral'] for s in summary]), linestyle='--', color='gray', label='Predictive-moment spectral fit')
    ax.set(ylabel='Predictive KL (nats; lower is better)', xlabel='Offline updates')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside upper center', ncol=2)
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-corrected-dreams-learning.png')
    plt.close(fig)
    diagnostic_axes[0].set(ylabel='Cause-posterior KL (nats)', xlabel='Offline updates')
    diagnostic_axes[1].set(ylabel='Predictive KL (nats)', xlabel='Generated proposal count')
    diagnostic_fig.legend(*diagnostic_axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2)
    diagnostic_fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-corrected-dreams-recognition.png')
    plt.close(diagnostic_fig)
    fig, axes = plt.subplots(2, 1)
    for i in range(count):
        r = np.load(output/f'initial_{i}.npz')
        axes[0].plot(r['steps'], r['records'][:, 0])
        axes[1].plot(r['steps'], r['records'][:, 1])
    axes[0].set(ylabel='Predictive KL (nats)')
    axes[1].set(ylabel='Original-data NLL', xlabel='Initial original-data fitting updates')
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-corrected-dreams-initial.png')
    plt.close(fig)


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_corrected_dreams/pilot')
    parser.add_argument('--worlds', type=int, default=CONFIG['worlds'])
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--batch', type=int, default=CONFIG['batch'])
    parser.add_argument('--steps', type=int, default=CONFIG['practice_steps'])
    args = parser.parse_args()
    summary = run(args.output, args.worlds, args.seed, args.batch, args.steps)
    plot(args.output, args.worlds)
    print(json.dumps(dict(output=str(args.output), initial_mean=float(np.mean([s['initial'][0] for s in summary])),
                          best_initial_mean=float(np.mean([s['initial_best_predictive'] for s in summary])),
                          final_mean={k: float(np.mean([s[k]['final'][0] for s in summary])) for k in ['joint', 'frozen', 'collapsed', 'original', 'self', 'oracle']}), indent=2))


if __name__ == '__main__': main()
