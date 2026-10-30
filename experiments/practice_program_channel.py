"""Screen learned program generation/recognition against direct planning.

The finite prior is over noncommuting primitive permutations. Only initial
noisy program/outcome examples update that posterior. All offline feedback
compares an internally chosen cause with the learned recognizer's answer.
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


WORDS = np.array(list(product(range(3), repeat=6)))
FEATURES = np.eye(3)[WORDS]
BETA = 4.


def softmax(x):
    p = np.exp(x-x.max(axis=-1, keepdims=True))
    return p/p.sum(axis=-1, keepdims=True)


def unit(x):
    x = x-x.mean(axis=-1, keepdims=True)
    return x/np.linalg.norm(x, axis=-1, keepdims=True)


def tangent(grad, weight):
    grad = grad-grad.mean(axis=-1, keepdims=True)
    return grad-np.sum(grad*weight, axis=-1, keepdims=True)*weight


def distributions(w, v):
    p = softmax(BETA*w)
    g = np.exp(np.einsum('ntk,wztk->wzn', FEATURES, np.log(p)))
    return g, softmax(BETA*v), p


def worlds(rng, count, hypotheses=64, observations=5, noise=.25):
    primitives = np.array([[[rng.permutation(8) for _ in range(3)] for _ in range(hypotheses)] for _ in range(count)])
    index = np.arange(hypotheses)[None, :, None]
    batch = np.arange(count)[:, None, None]
    endpoints = np.zeros((count, hypotheses, len(WORDS)), dtype=int)
    for t in range(6): endpoints = primitives[batch, index, WORDS[None, None, :, t], endpoints]
    true_index = rng.integers(hypotheses, size=count)
    inputs = rng.integers(len(WORDS), size=(count, observations))
    predicted = endpoints[np.arange(count)[:, None], true_index[:, None], inputs]
    observed = np.where(rng.random((count, observations)) < noise, rng.integers(8, size=predicted.shape), predicted)
    likelihood = (1-noise)*(endpoints[np.arange(count)[:, None, None], np.arange(hypotheses)[None, :, None], inputs[:, None]] == observed[:, None])+noise/8
    posterior = softmax(np.log(likelihood).sum(axis=-1))
    channel = noise/8+(1-noise)*np.einsum('wh,whny->wny', posterior, np.eye(8)[endpoints])
    truth = noise/8+(1-noise)*np.eye(8)[endpoints[np.arange(count), true_index]]
    assert np.allclose(channel.sum(axis=-1), 1)
    return dict(primitives=primitives, endpoints=endpoints, true_index=true_index, inputs=inputs, observed=observed, posterior=posterior, channel=channel, truth=truth)


def reconstruction(g, r, p, data):
    batch = np.arange(len(g))[:, None]
    gi = g.transpose(0, 2, 1)[batch, data['inputs']]
    ri = r[batch, data['observed']]
    mass = gi*ri
    likelihood = mass.sum(axis=-1)
    responsibility = mass/likelihood[..., None]
    dw = np.einsum('wiz,witk->wztk', responsibility, FEATURES[data['inputs']])
    dw -= responsibility.sum(axis=1)[:, :, None, None]*p
    dv = np.zeros_like(r)
    np.add.at(dv, (batch, data['observed']), responsibility-ri)
    return -np.log(likelihood).mean(axis=1), BETA*dw/data['inputs'].shape[1], BETA*dv/data['inputs'].shape[1]


def field(w, v, data, condition):
    g, r, p = distributions(w, v)
    if condition == 'reconstruction': _, dw, dv = reconstruction(g, r, p, data)
    else:
        mass = g*(data['channel'] @ r).transpose(0, 2, 1)/4
        dw = BETA*(np.einsum('wzn,ntk->wztk', mass, FEATURES)-mass.sum(axis=-1)[:, :, None, None]*p)
        received = (g @ data['channel']).transpose(0, 2, 1)
        dv = BETA*r*(received-(received*r).sum(axis=-1, keepdims=True))/4
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
    g, r, p = distributions(w, v)
    result = []
    for channel in [data['channel'], data['truth']]:
        received = g @ channel
        result.extend([np.einsum('wzy,wyz->w', received, r)/4, received.max(axis=1).sum(axis=-1)/4])
    result.append(reconstruction(g, r, p, data)[0])
    return np.array(result).T


def fit(w, v, data, condition, steps, rate, start=0):
    records, ws, vs, times = [], [], [], []
    stationary = False
    for iteration in tqdm(range(start, start+steps+1), desc=condition, leave=False):
        if iteration % 20 == 0:
            record = evaluate(w, v, data)
            assert np.isfinite(record).all()
            records.append(record); ws.append(w.copy()); vs.append(v.copy()); times.append(iteration)
            if len(records) > 1:
                change = records[-2][:, 4]-record[:, 4] if condition == 'reconstruction' else record[:, 0]-records[-2][:, 0]
                assert change.min() > -1e-7, (condition, iteration, change.min())
            if iteration >= start+600:
                means = np.array(records[-20:]).mean(axis=1)
                loss = means[:, [4]] if condition == 'reconstruction' else 1-means[:, :2]
                slopes = np.polyfit(np.log(np.array(times[-20:])+1), np.log(loss), 1)[0]
                if np.max(np.abs(slopes)) < .005:
                    stationary = True
                    break
        if iteration == start+steps: break
        w, v = step(w, v, data, condition, rate)
    return dict(records=np.array(records), w=np.array(ws), v=np.array(vs), steps=np.array(times), stationary=np.array(stationary))


def code_weights(code): return unit(np.eye(3)[WORDS[code]])


def scores(weights, channel):
    p = softmax(BETA*weights)
    g = np.exp(np.einsum('ntk,bztk->bzn', FEATURES, np.log(p)))
    return (g @ channel).max(axis=1).sum(axis=-1)/4


def select_code(candidates, channel):
    # Small declared greedy search over the supplied program candidates.
    logits = BETA*unit(np.eye(3)[WORDS[candidates]])
    probabilities = np.exp(np.einsum('ntk,ctk->cn', FEATURES, np.log(softmax(logits))))
    received = probabilities @ channel
    code, covered = [], np.zeros(8)
    for _ in range(4):
        value = np.maximum(received, covered).sum(axis=-1)
        chosen = int(value.argmax()); code.append(chosen)
        covered = np.maximum(covered, received[chosen])
    return np.array(candidates)[code]


def viterbi_candidates(transition):
    # Most likely individual path to each endpoint under a single Markov
    # transition model. No exponential enumeration of all words is needed.
    value = np.zeros(8); value[0] = 1.
    back = []
    for _ in range(6):
        joint = value[:, None, None]*transition.transpose(1, 0, 2)
        flattened = joint.reshape(24, 8)
        choice = flattened.argmax(axis=0)
        value = flattened.max(axis=0)
        back.append(choice)
    words = []
    for target in range(8):
        state, word = target, []
        for choice in back[::-1]:
            state_action = int(choice[state]); word.append(state_action % 3); state = state_action//3
        words.append(word[::-1])
    return np.unique(np.array(words) @ (3**np.arange(5, -1, -1)))


def baselines(data):
    names = ['map_graph', 'mean_graph', 'short_periodic', 'enumerated_spectral', 'enumerated_greedy']
    results, codes = {name: [] for name in names}, {name: [] for name in names}
    templates = []
    for length in [1, 2, 3]:
        for word in product(range(3), repeat=length): templates.append((word*6)[:6])
    templates = np.unique(np.array(templates) @ (3**np.arange(5, -1, -1)))
    for s, channel in enumerate(tqdm(data['channel'], desc='Program baselines')):
        maps = np.eye(8)[data['primitives'][s]]
        candidates = {'map_graph': viterbi_candidates(maps[data['posterior'][s].argmax()]), 'mean_graph': viterbi_candidates(np.einsum('h,haxy->axy', data['posterior'][s], maps)), 'short_periodic': templates, 'enumerated_greedy': np.arange(len(WORDS))}
        centered = channel-channel.mean(axis=0)
        u, singular, _ = np.linalg.svd(centered, full_matrices=False)
        features = u[:, :3]*singular[:3]
        distance = np.sum((features[:, None]-features[None])**2, axis=-1)
        chosen = [int(np.sum(features**2, axis=-1).argmax())]
        for _ in range(3):
            value = distance[:, chosen].min(axis=1); value[chosen] = -np.inf
            chosen.append(int(value.argmax()))
        for name in names:
            code = np.array(chosen) if name == 'enumerated_spectral' else select_code(candidates[name], channel)
            codes[name].append(code)
            weight = code_weights(code)[None]
            results[name].append([scores(weight, channel)[0], scores(weight, data['truth'][s])[0]])
    return {name: np.array(value) for name, value in results.items()}, {name: np.array(value) for name, value in codes.items()}


def audit(w, v, data):
    rng = np.random.default_rng(501009)
    dw, dv = tangent(rng.normal(size=w.shape), w), tangent(rng.normal(size=v.shape), v)
    error = {}
    for condition, column, sign in [('joint', 0, 1), ('reconstruction', 4, -1)]:
        epsilon = 1e-6
        plus = sign*evaluate(unit(w+epsilon*dw), unit(v+epsilon*dv), data)[:, column].sum()
        minus = sign*evaluate(unit(w-epsilon*dw), unit(v-epsilon*dv), data)[:, column].sum()
        gw, gv = field(w, v, data, condition)
        error[condition] = float(abs((plus-minus)/(2*epsilon)-np.sum(gw*dw)-np.sum(gv*dv)))
        assert error[condition] < 1e-6, error
    return error


def plot(runs, baseline):
    fig, axes = plt.subplots(3, 1)
    for name, run in runs.items():
        if name == 'initial_fit': continue
        mean = run['records'].mean(axis=1)
        for ax, column in zip(axes, [0, 1, 4]): ax.plot(run['steps'], mean[:, column], label=name.replace('_', ' ') if column == 0 else None)
    labels = {'map_graph': 'MAP graph', 'mean_graph': 'mean graph', 'short_periodic': 'periodic search', 'enumerated_spectral': 'enumerated SVD', 'enumerated_greedy': 'full greedy search', 'raw_svd': 'raw SVD', 'posterior_svd': 'posterior SVD'}
    for i, (name, result) in enumerate(baseline.items()): axes[1].axhline(result[:, 0].mean(), color=f'C{i+4}', linestyle='--', label=labels[name])
    axes[0].set(ylabel='Joint accuracy'); axes[0].legend(ncol=2)
    axes[1].set(ylabel='Generator accuracy'); axes[1].legend(ncol=2, loc='lower right')
    axes[2].set(ylabel='Original-pair NLL', xlabel='Offline expected-update steps')
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-program-channel-learning.png'); plt.close(fig)


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_program_channel/pilot')
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2010261)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--rate', type=float, default=.1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = vars(args) | {'output': str(args.output), 'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'hypotheses': 64, 'states': 8, 'primitives': 3, 'program_length': 6, 'initial_examples': 5, 'noise': .25, 'messages': 4, 'beta': BETA, 'scope': 'Exploratory expected-gradient pilot. Prior hypotheses are supplied triples of independent random permutations; a hidden index is drawn uniformly. Five noisy full program/outcome observations update the exact posterior. All candidates and controls use that posterior only. Initial original-pair stochastic autoencoder uses identical grammar and recognition parameters. Both learn on unit centered weight spheres at the same rate. Stop after >=600 steps when the optimized loss (and generator error in practice) has absolute mean double-log slope <.005 over 20 evaluations. Direct MAP and posterior-mean graph planners, fixed short-periodic programs, and full enumeration search controls. Full enumeration generates all 729 candidate programs and is not called a simple original-data analytic shortcut. No claim of learned library structure or a recognition lead.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, args.output/'source.py')
    rng = np.random.default_rng(args.seed)
    data = worlds(rng, args.worlds)
    w, v = unit(rng.normal(size=(args.worlds, 4, 6, 3))), unit(rng.normal(size=(args.worlds, 8, 4)))
    np.savez_compressed(args.output/'worlds.npz', **data, initial_w=w, initial_v=v)
    errors = audit(w, v, data)
    baseline, codes = baselines(data)
    np.savez_compressed(args.output/'baselines.npz', **baseline)
    np.savez_compressed(args.output/'baseline_codes.npz', **codes)
    initial = fit(w, v, data, 'reconstruction', args.steps, args.rate)
    runs = {'initial_fit': initial}
    np.savez_compressed(args.output/'initial_fit.npz', **initial)
    for condition in tqdm(['joint', 'reconstruction', 'frozen_recognizer', 'recognition_only'], desc='Offline program conditions'):
        runs[condition] = fit(initial['w'][-1], initial['v'][-1], data, condition, args.steps, args.rate)
        np.savez_compressed(args.output/f'{condition}.npz', **runs[condition])
    summary = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'gradient_errors': errors, 'metrics': ['posterior_joint_success', 'posterior_generator_success', 'true_joint_success', 'true_generator_success', 'original_reconstruction_nll'], 'posterior_entropy_mean': float(-(data['posterior']*np.log(data['posterior'])).sum(axis=-1).mean()), 'posterior_max_probability_mean': float(data['posterior'].max(axis=-1).mean()), 'runs': {name: {'last_step': int(run['steps'][-1]), 'stationary': bool(run['stationary']), 'initial': run['records'][0].mean(axis=0).tolist(), 'final': run['records'][-1].mean(axis=0).tolist()} for name, run in runs.items()}, 'baselines_posterior_true': {name: value.mean(axis=0).tolist() for name, value in baseline.items()}}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(runs, {name: baseline[name] for name in ['map_graph', 'mean_graph', 'short_periodic', 'enumerated_spectral']})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
