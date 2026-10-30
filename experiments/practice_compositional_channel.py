"""Finite-data Boolean-channel practice with shared encoder/decoder weights.

Exploratory expected-update screen. The hidden channel is evaluation-only.
Original-pair reconstruction, matched-architecture spectral constructions,
and greedy code selection are explicit controls, not training signals.
"""

import argparse
from datetime import datetime
from itertools import permutations, product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


ACTION = np.array(list(product([0., 1.], repeat=6)))
OUTPUT = ACTION.copy()
CAUSE = np.array(list(product([0., 1.], repeat=2)))
FZ = np.column_stack([np.ones(4), 2*CAUSE-1])
FY = np.column_stack([np.ones(64), 2*OUTPUT-1])
BETA = 4.
METRICS = ['posterior_joint_success', 'posterior_generator_success', 'true_joint_success', 'true_generator_success', 'original_reconstruction_nll']


def sigmoid(x): return np.exp(-np.logaddexp(0, -x))
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
def tangent(grad, weight): return grad - np.sum(grad*weight, axis=-1, keepdims=True)*weight


def normalize_construction(x):
    # A zero Walsh projection (pure parity) or null spectral row has no
    # preferred unit direction. Choose the positive constant-bias direction.
    x = x.copy()
    x[np.linalg.norm(x, axis=-1) < 1e-12] = np.array([1., 0., 0.])
    return unit(x)


def distributions(w, v):
    lg = BETA*np.einsum('wan,zn->wza', w, FZ)
    lr = BETA*np.einsum('wkn,yn->wyk', v, FY)
    g = np.exp(np.einsum('an,wzn->wza', ACTION, lg)-np.logaddexp(0, lg).sum(axis=-1)[..., None])
    r = np.exp(np.einsum('zk,wyk->wyz', CAUSE, lr)-np.logaddexp(0, lr).sum(axis=-1)[..., None])
    return g, r, sigmoid(lg), sigmoid(lr)


def make_worlds(rng, worlds, observations, noise):
    wiring = np.array([[rng.choice(6, 2, replace=False) for _ in range(6)] for _ in range(worlds)])
    gates = rng.integers(2, size=(worlds, 6, 4))
    inputs = rng.integers(64, size=(worlds, observations))
    observed = np.zeros((worlds, observations, 6), dtype=int)
    counts = np.zeros((worlds, 6, 4, 2), dtype=int)
    truth = np.zeros((worlds, 64, 64))
    channel = np.zeros_like(truth)
    posterior = np.zeros((worlds, 6, 4))
    for s in range(worlds):
        local = (2*ACTION[:, wiring[s, :, 0]]+ACTION[:, wiring[s, :, 1]]).astype(int)
        true_bits = gates[s][np.arange(6)[None, :], local]
        observed[s] = np.logical_xor(true_bits[inputs[s]], rng.random((observations, 6)) < noise)
        for j in range(6): np.add.at(counts[s, j], (local[inputs[s], j], observed[s, :, j]), 1)
        posterior[s] = sigmoid((counts[s, :, :, 1]-counts[s, :, :, 0])*np.log((1-noise)/noise))
        for destination, bitprob in [(truth, noise+(1-2*noise)*true_bits), (channel, noise+(1-2*noise)*posterior[s][np.arange(6)[None, :], local])]:
            destination[s] = np.exp(np.log(bitprob) @ OUTPUT.T + np.log1p(-bitprob) @ (1-OUTPUT).T)
    yindex = (observed @ (2**np.arange(5, -1, -1))).astype(int)
    assert np.allclose(channel.sum(axis=-1), 1)
    assert np.allclose(truth.sum(axis=-1), 1)
    return dict(channel=channel, truth=truth, wiring=wiring, gates=gates, counts=counts, posterior=posterior, inputs=inputs, observed=observed, yindex=yindex)


def reconstruction(g, r, pg, pr, data):
    index = np.arange(len(g))[:, None]
    gi = g.transpose(0, 2, 1)[index, data['inputs']]
    ri = r[index, data['yindex']]
    mass = gi*ri
    probability = mass.sum(axis=-1)
    responsibility = mass/probability[..., None]
    dg = np.einsum('wiz,win->wzn', responsibility, ACTION[data['inputs']])
    dg -= responsibility.sum(axis=1)[..., None]*pg
    dw = BETA*np.einsum('wzn,zk->wnk', dg, FZ)/data['inputs'].shape[1]
    dr = responsibility @ CAUSE-pr[index, data['yindex']]
    dv = BETA*np.einsum('wik,win->wkn', dr, FY[data['yindex']])/data['inputs'].shape[1]
    return -np.log(probability).mean(axis=1), dw, dv


def practice_gradient(g, r, pg, pr, channel):
    mass = g*(channel @ r).transpose(0, 2, 1)/4
    dg = mass @ ACTION-mass.sum(axis=-1)[..., None]*pg
    dw = BETA*np.einsum('wzn,zk->wnk', dg, FZ)
    mass = (g @ channel).transpose(0, 2, 1)*r/4
    dr = mass @ CAUSE-mass.sum(axis=-1)[..., None]*pr
    dv = BETA*np.einsum('wyk,yn->wkn', dr, FY)
    return dw, dv


def field(w, v, data, condition):
    g, r, pg, pr = distributions(w, v)
    if condition == 'reconstruction': _, dw, dv = reconstruction(g, r, pg, pr, data)
    else: dw, dv = practice_gradient(g, r, pg, pr, data['channel'])
    if condition == 'frozen_recognizer': dv *= 0
    if condition == 'recognition_only': dw *= 0
    return tangent(dw, w), tangent(dv, v)


def step(w, v, data, condition, rate):
    a, b = field(w, v, data, condition)
    c, d = field(w+rate*a/2, v+rate*b/2, data, condition)
    e, f = field(w+rate*c/2, v+rate*d/2, data, condition)
    h, i = field(w+rate*e, v+rate*f, data, condition)
    return unit(w+rate*(a+2*c+2*e+h)/6), unit(v+rate*(b+2*d+2*f+i)/6)


def evaluate(w, v, data):
    g, r, pg, pr = distributions(w, v)
    metrics = []
    for channel in [data['channel'], data['truth']]:
        received = g @ channel
        metrics.extend([np.einsum('wzy,wyz->w', received, r)/4, received.max(axis=1).sum(axis=-1)/4])
    metrics.append(reconstruction(g, r, pg, pr, data)[0])
    return np.array(metrics).T


def fit(w, v, data, condition, steps, rate, minimum):
    records, ws, vs, iterations = [], [], [], []
    stationary = False
    for iteration in tqdm(range(steps+1), desc=condition, leave=False):
        if iteration % 20 == 0:
            record = evaluate(w, v, data)
            assert np.isfinite(record).all()
            records.append(record); ws.append(w.copy()); vs.append(v.copy()); iterations.append(iteration)
            if len(records) > 1:
                improvement = records[-2][:, 4]-record[:, 4] if condition == 'reconstruction' else record[:, 0]-records[-2][:, 0]
                assert improvement.min() >= -1e-7, (condition, iteration, improvement.min())
            if iteration >= minimum:
                measured = np.array(records[-20:]).mean(axis=1)
                losses = measured[:, [4]] if condition == 'reconstruction' else 1-measured[:, :2]
                slopes = np.polyfit(np.log(np.array(iterations[-20:])+1), np.log(losses), 1)[0]
                if np.max(np.abs(slopes)) < .005:
                    stationary = True
                    break
        if iteration == steps: break
        w, v = step(w, v, data, condition, rate)
    return dict(records=np.array(records), w=np.array(ws), v=np.array(vs), steps=np.array(iterations), stationary=np.array(stationary))


def score_w(w, channel):
    logits = BETA*np.einsum('ban,zn->bza', w, FZ)
    g = np.exp(np.einsum('an,bzn->bza', ACTION, logits)-np.logaddexp(0, logits).sum(axis=-1)[..., None])
    return (g @ channel).max(axis=1).sum(axis=-1)/4


def projected_code(channel, code):
    assignments = np.array(list(permutations(code)))
    weights = normalize_construction(np.einsum('bzn,zk->bnk', 2*ACTION[assignments]-1, FZ)/4)
    best = score_w(weights, channel).argmax()
    return weights[best], assignments[best]


def direct_baselines(data):
    results = {name: [] for name in ['raw_pair_svd', 'channel_covariance_svd', 'channel_spectral', 'greedy_projected']}
    weights = {name: [] for name in results}
    codes = {name: [] for name in ['channel_spectral', 'greedy_projected']}
    for s, channel in enumerate(tqdm(data['channel'], desc='Matched direct constructions')):
        covariance = (2*ACTION-1).T @ channel @ (2*OUTPUT-1)/64
        empirical = (2*ACTION[data['inputs'][s]]-1).T @ (2*data['observed'][s]-1)/data['inputs'].shape[1]
        for name, matrix in [('raw_pair_svd', empirical), ('channel_covariance_svd', covariance)]:
            u, _, _ = np.linalg.svd(matrix, full_matrices=False)
            # A small, declared analytic candidate family: 2 singular-vector axes,
            # eight rotations, both orientations. Select with the learned channel.
            candidates = []
            for angle in np.arange(8)*np.pi/16:
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                for orientation in [np.eye(2), np.diag([1., -1.])]:
                    features = u[:, :2] @ orientation @ rotation
                    candidates.append(normalize_construction(np.column_stack([np.zeros(6), features])))
            candidates = np.array(candidates)
            weights[name].append(candidates[score_w(candidates, channel).argmax()])
        u, singular, _ = np.linalg.svd(channel-channel.mean(axis=0), full_matrices=False)
        features = u[:, :3]*singular[:3]
        distance = np.sum((features[:, None]-features[None])**2, axis=-1)
        spectral = [int(np.sum(features**2, axis=-1).argmax())]
        for _ in range(3):
            scores = distance[:, spectral].min(axis=1); scores[spectral] = -np.inf
            spectral.append(int(scores.argmax()))
        greedy, covered = [], np.zeros(64)
        for _ in range(4):
            scores = np.maximum(channel, covered).sum(axis=1); scores[greedy] = -np.inf
            greedy.append(int(scores.argmax())); covered = np.maximum(covered, channel[greedy[-1]])
        for name, code in [('channel_spectral', spectral), ('greedy_projected', greedy)]:
            w, assignment = projected_code(channel, code)
            weights[name].append(w); codes[name].append(assignment)
        for name in results:
            w = weights[name][-1][None]
            results[name].append([score_w(w, channel)[0], score_w(w, data['truth'][s])[0]])
    return {name: np.array(value) for name, value in results.items()}, {name: np.array(value) for name, value in weights.items()}, codes


def audit(w, v, data):
    rng = np.random.default_rng(372601)
    dw, dv = tangent(rng.normal(size=w.shape), w), tangent(rng.normal(size=v.shape), v)
    errors = {}
    epsilon = 1e-6
    for condition, column, sign in [('joint', 0, 1), ('reconstruction', 4, -1)]:
        plus = sign*evaluate(unit(w+epsilon*dw), unit(v+epsilon*dv), data)[:, column].sum()
        minus = sign*evaluate(unit(w-epsilon*dw), unit(v-epsilon*dv), data)[:, column].sum()
        gw, gv = field(w, v, data, condition)
        errors[condition] = float(abs((plus-minus)/(2*epsilon)-np.sum(gw*dw)-np.sum(gv*dv)))
        assert errors[condition] < 1e-7, errors
    return errors


def plot_runs(runs, baseline):
    fig, axes = plt.subplots(3, 1)
    labels = {'joint': 'joint practice', 'reconstruction': 'original-data fit', 'frozen_recognizer': 'frozen recognizer', 'recognition_only': 'recognition only'}
    for name, run in runs.items():
        if name == 'initial_fit': continue
        mean = run['records'].mean(axis=1)
        for ax, column in zip(axes, [0, 1, 4]): ax.plot(run['steps'], mean[:, column], label=labels[name] if column == 0 else None)
    baseline_labels = {'raw_pair_svd': 'raw SVD', 'channel_covariance_svd': 'channel SVD', 'channel_spectral': 'spectral code', 'greedy_projected': 'greedy code'}
    for i, (name, result) in enumerate(baseline.items()): axes[1].axhline(result[:, 0].mean(), color=f'C{i+4}', linestyle='--', label=baseline_labels[name])
    axes[0].set(ylabel='Joint accuracy')
    axes[1].set(ylabel='Generator accuracy')
    axes[2].set(ylabel='Original-pair NLL', xlabel='Offline expected-update steps')
    axes[0].legend(ncol=2)
    axes[1].legend(ncol=2, loc='lower right')
    fig.tight_layout()
    path = Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-compositional-channel-learning.png'
    fig.savefig(path); plt.close(fig)
    fig, axes = plt.subplots(2, 1)
    initial = runs['initial_fit']
    axes[0].plot(initial['steps'], initial['records'][:, :, 4].mean(axis=1))
    axes[1].plot(initial['steps'], initial['records'][:, :, 1].mean(axis=1))
    axes[0].set(ylabel='Original-pair NLL')
    axes[1].set(ylabel='Generator accuracy', xlabel='Initial reconstruction steps')
    fig.tight_layout()
    fig.savefig(path.with_name('practice-compositional-channel-initial-fit.png')); plt.close(fig)


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_compositional_channel/pilot')
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--rate', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=710204)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = vars(args) | {'output': str(args.output), 'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'action_bits': 6, 'output_bits': 6, 'cause_bits': 2, 'observations': 24, 'noise': .1, 'beta': BETA, 'scope': 'Exploratory expected-flow pilot, no fresh confirmation. Independent uniform Boolean two-input truth tables, known wiring and bit noise. Exact posterior from finite original pairs. Both networks have unit-norm rows, fixed inverse temperature, shared linear features. Initial original-command reconstruction through the same stochastic bottleneck. Then joint practice, continued reconstruction, frozen recognizer, and recognition only. Stop each after >=400 steps when absolute double-log slopes of mean training objective loss and generator error are <.005 over 20 evaluations; maximum is steps. Reconstruction stops on original NLL only. Monotonicity of the actually optimized objective is asserted. Spectral controls have the identical generator architecture, both column orientations and eight rotations for covariance SVD; full-channel spectral/farthest-point and greedy codes use all 24 label assignments projected onto identical weights. Selection uses posterior channel only. Greedy search is an efficiency control, not an analytic disqualifier. No sampled-update or per-world generation-only theorem claimed.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, args.output/'source.py')
    rng = np.random.default_rng(args.seed)
    data = make_worlds(rng, args.worlds, 24, .1)
    w = unit(rng.normal(size=(args.worlds, 6, 3)))
    v = unit(rng.normal(size=(args.worlds, 2, 7)))
    np.savez_compressed(args.output/'worlds.npz', **data, initial_w=w, initial_v=v)
    errors = audit(w, v, data)
    baseline, weights, codes = direct_baselines(data)
    np.savez_compressed(args.output/'baselines.npz', **baseline)
    np.savez_compressed(args.output/'baseline_weights.npz', **weights)
    initial = fit(w, v, data, 'reconstruction', args.steps, args.rate, 400)
    np.savez_compressed(args.output/'initial_fit.npz', **initial)
    runs = {'initial_fit': initial}
    for condition in tqdm(['joint', 'reconstruction', 'frozen_recognizer', 'recognition_only'], desc='Offline conditions'):
        run = fit(initial['w'][-1], initial['v'][-1], data, condition, args.steps, args.rate, 400)
        np.savez_compressed(args.output/f'{condition}.npz', **run)
        runs[condition] = run
    summary = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'gradient_errors': errors, 'metric_order': METRICS, 'runs': {name: {'last_step': int(run['steps'][-1]), 'stationary': bool(run['stationary']), 'initial': run['records'][0].mean(axis=0).tolist(), 'final': run['records'][-1].mean(axis=0).tolist()} for name, run in runs.items()}, 'baseline_means_posterior_true': {name: value.mean(axis=0).tolist() for name, value in baseline.items()}, 'baseline_codes': {name: np.array(value).tolist() for name, value in codes.items()}}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot_runs(runs, baseline)
    print(json.dumps({key: value for key, value in summary.items() if key != 'baseline_codes'}, indent=2))


if __name__ == '__main__': main()
