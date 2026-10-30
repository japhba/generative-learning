"""Screen joint cause coding against cheap direct code-design alternatives.

All learning uses the posterior predictive channel from finite initial data.
The actual world is evaluation-only. This is an expected-update screen: it
tests the strongest noiseless version before spending compute on sampling.
"""

import argparse
from datetime import datetime
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def softmax(x):
    exponential = np.exp(x - x.max(axis=-1, keepdims=True))
    return exponential / exponential.sum(axis=-1, keepdims=True)


def evaluate(g, r, channel):
    likelihood = g @ channel
    joint = np.einsum('wzy,wyz->w', likelihood, r) / g.shape[1]
    optimal_recognition = likelihood.max(axis=1).sum(axis=1) / g.shape[1]
    return np.stack([joint, optimal_recognition], axis=-1)


def gradient(g_logits, r_logits, channel):
    g, r = softmax(g_logits), softmax(r_logits)
    rewards = channel @ r
    rewards = rewards.transpose(0, 2, 1)
    dg = g * (rewards - (g * rewards).sum(axis=-1, keepdims=True)) / g.shape[1]
    received = (g @ channel).transpose(0, 2, 1)
    dr = r * (received - (received * r).sum(axis=-1, keepdims=True)) / g.shape[1]
    return dg, dr


def step(g, r, channel, rate, condition):
    def field(a, b):
        dg, dr = gradient(a, b, channel)
        if condition == 'recognition_only': dg *= 0
        if condition == 'frozen_recognizer': dr *= 0
        return dg, dr
    a, b = field(g, r)
    c, d = field(g + rate*a/2, r + rate*b/2)
    e, f = field(g + rate*c/2, r + rate*d/2)
    h, i = field(g + rate*e, r + rate*f)
    return g + rate*(a + 2*c + 2*e + h)/6, r + rate*(b + 2*d + 2*f + i)/6


def make_worlds(rng, worlds, actions, outputs, observations, noise):
    supports = np.array(list(combinations(range(outputs), 3)))
    incidence = np.eye(outputs)[supports].sum(axis=1)
    hidden_support = rng.integers(len(supports), size=(worlds, actions))
    truth = noise / outputs + (1-noise) * incidence[hidden_support] / 3
    counts = np.array([[rng.multinomial(observations, row) for row in world] for world in truth])
    log_ratio = np.log((noise / outputs + (1-noise)/3) / (noise / outputs))
    posterior = softmax(log_ratio * (counts @ incidence.T))
    learned = noise / outputs + (1-noise) * (posterior @ incidence) / 3
    assert np.allclose(learned.sum(axis=-1), 1)
    return truth, learned, counts, hidden_support, posterior, supports


def code_success(channel, code): return channel[code].max(axis=0).sum() / len(code)


def direct_codes(channel, messages):
    actions = len(channel)
    chosen, covered = [], np.zeros(channel.shape[1])
    for _ in range(messages):
        scores = np.maximum(channel, covered).sum(axis=1)
        scores[chosen] = -np.inf
        chosen.append(int(scores.argmax()))
        covered = np.maximum(covered, channel[chosen[-1]])
    greedy = np.array(chosen)
    swap = greedy.copy()
    swap_rounds = 0
    while True:
        alternatives = []
        for slot in range(messages):
            for action in range(actions):
                candidate = swap.copy(); candidate[slot] = action
                if len(np.unique(candidate)) == messages: alternatives.append(candidate)
        alternatives = np.array(alternatives)
        values = channel[alternatives].max(axis=1).sum(axis=1) / messages
        best = int(values.argmax())
        if values[best] <= code_success(channel, swap) + 1e-12: break
        swap = alternatives[best]; swap_rounds += 1
    centered = channel - channel.mean(axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    features = u[:, :messages-1] * singular[:messages-1]
    distance = np.sum((features[:, None] - features[None])**2, axis=-1)
    spectral = [int(np.sum(features**2, axis=-1).argmax())]
    for _ in range(messages-1):
        scores = distance[:, spectral].min(axis=1)
        scores[spectral] = -np.inf
        spectral.append(int(scores.argmax()))
    all_codes = np.array(list(combinations(range(actions), messages)))
    scores = channel[all_codes].max(axis=1).sum(axis=1) / messages
    return {'greedy': greedy, 'greedy_swap': swap, 'spectral': np.array(spectral), 'exhaustive': all_codes[scores.argmax()]}, swap_rounds


def audit_gradient(g, r, channel):
    dg, dr = gradient(g, r, channel)
    rng = np.random.default_rng(91019)
    vg, vr = rng.normal(size=g.shape), rng.normal(size=r.shape)
    epsilon = 1e-5
    plus = evaluate(softmax(g+epsilon*vg), softmax(r+epsilon*vr), channel)[:, 0].sum()
    minus = evaluate(softmax(g-epsilon*vg), softmax(r-epsilon*vr), channel)[:, 0].sum()
    discrepancy = abs((plus-minus)/(2*epsilon) - np.sum(dg*vg) - np.sum(dr*vr))
    assert discrepancy < 1e-8, discrepancy
    return float(discrepancy)


def plot_runs(runs, baselines):
    fig, axes = plt.subplots(2, 1)
    for name, run in runs.items():
        axes[0].plot(run['steps'], run['records'][:, :, 0].mean(axis=1), label=name.replace('_', ' '))
        axes[1].plot(run['steps'], run['records'][:, :, 1].mean(axis=1), label=name.replace('_', ' '))
    for index, name in enumerate(['spectral', 'greedy', 'greedy_swap', 'exhaustive']):
        axes[1].axhline(baselines[name][:, 0].mean(), linestyle='--', color=f'C{index+3}', label=name.replace('_', ' '))
    axes[0].set(xlabel='Expected-update steps', ylabel='Joint accuracy')
    axes[1].set(xlabel='Expected-update steps', ylabel='Generator accuracy')
    axes[0].legend()
    axes[1].legend(ncol=2, loc='lower right')
    fig.tight_layout()
    plot = Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-discrete-channel-learning.png'
    fig.savefig(plot); plt.close(fig)


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_discrete_channel')
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--rate', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=719213)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = vars(args) | {'output': str(args.output), 'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'actions': 20, 'outputs': 12, 'messages': 4, 'initial_samples_per_action': 12, 'noise': .1, 'initial_encoder_logit_std': 1.4, 'evaluation_every': 20, 'scope': 'Exploratory expected-gradient screen. Same rate for generator and recognizer; exact conditional Bayes population success. Greedy, one-swap, spectral and exhaustive code selection are evaluation-only baselines. No claim of matched compute or sampled-learning performance. Stop when both mean joint error and optimal-decoder error have absolute log-log slope below .01 over the latest 20 evaluations after 1000 steps.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, args.output/'source.py')
    rng = np.random.default_rng(args.seed)
    truth, channel, counts, hidden, posterior, supports = make_worlds(rng, args.worlds, 20, 12, 12, .1)
    initial_g = rng.normal(0, 1.4, (args.worlds, 4, 20))
    initial_r = np.zeros((args.worlds, 12, 4))
    gradient_error = audit_gradient(initial_g, rng.normal(size=initial_r.shape), channel)
    initial_dg, initial_dr = gradient(initial_g, initial_r, channel)
    assert np.max(np.abs(initial_dg)) < 1e-15
    assert np.all(np.linalg.norm(initial_dr, axis=(1, 2)) > 0)
    np.savez_compressed(args.output/'worlds.npz', truth=truth, channel=channel, counts=counts, hidden=hidden, posterior=posterior, supports=supports, initial_g=initial_g, initial_r=initial_r)
    baselines = {name: [] for name in ['greedy', 'greedy_swap', 'spectral', 'exhaustive']}
    baseline_codes = {name: [] for name in baselines}
    swap_rounds = []
    for w in tqdm(range(args.worlds), desc='Direct baseline screen'):
        codes, rounds = direct_codes(channel[w], 4); swap_rounds.append(rounds)
        for name, code in codes.items():
            baselines[name].append([code_success(channel[w], code), code_success(truth[w], code)])
            baseline_codes[name].append(code.tolist())
    baselines = {name: np.array(value) for name, value in baselines.items()}
    runs = {}
    initial_received = (softmax(initial_g) @ channel).transpose(0, 2, 1)
    fixed_r = np.eye(4)[initial_received.argmax(axis=-1)]
    fixed_limit = (channel @ fixed_r).max(axis=1).mean(axis=-1)
    for condition in tqdm(['joint', 'recognition_only', 'frozen_recognizer'], desc='Learning conditions'):
        g, r = initial_g.copy(), initial_r.copy()
        if condition == 'frozen_recognizer': r = 60*fixed_r
        records, gs, rs, times = [], [], [], []
        for iteration in tqdm(range(args.steps+1), desc=condition, leave=False):
            if iteration % 20 == 0:
                gp, rp = softmax(g), softmax(r)
                record = np.concatenate([evaluate(gp, rp, channel), evaluate(gp, rp, truth)], axis=-1)
                assert np.isfinite(record).all()
                records.append(record); gs.append(g.copy()); rs.append(r.copy()); times.append(iteration)
                if condition == 'joint' and len(records) > 1:
                    assert np.min(record[:, 0]-records[-2][:, 0]) >= -1e-8
                if iteration >= 1000:
                    errors = 1 - np.array(records[-20:])[:, :, :2].mean(axis=1)
                    slopes = np.polyfit(np.log(np.array(times[-20:])+1), np.log(errors), 1)[0]
                    if np.max(np.abs(slopes)) < .01: break
            if iteration == args.steps: break
            g, r = step(g, r, channel, args.rate, condition)
        runs[condition] = {'records': np.array(records), 'steps': np.array(times), 'g': np.array(gs), 'r': np.array(rs)}
        np.savez_compressed(args.output/f'{condition}.npz', **runs[condition])
    joint = runs['joint']['records']
    summary = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'gradient_directional_error': gradient_error, 'initial_generator_drift_max': float(np.max(np.abs(initial_dg))), 'initial_recognizer_drift_mean_norm': float(np.linalg.norm(initial_dr, axis=(1, 2)).mean()), 'metric_order': ['posterior_joint_success', 'posterior_optimal_decoder_success', 'true_joint_success', 'true_optimal_decoder_success'], 'runs': {name: {'last_step': int(run['steps'][-1]), 'initial': run['records'][0].mean(axis=0).tolist(), 'final': run['records'][-1].mean(axis=0).tolist()} for name, run in runs.items()}, 'baseline_means_posterior_then_true': {name: value.mean(axis=0).tolist() for name, value in baselines.items()}, 'frozen_recognizer_exact_limit': float(fixed_limit.mean()), 'greedy_swap_rounds': swap_rounds, 'greedy_swap_minus_joint_optimal_decoder_mean': float(np.mean(baselines['greedy_swap'][:, 0]-joint[-1, :, 1])), 'greedy_swap_beats_joint_worlds': int(np.sum(baselines['greedy_swap'][:, 0] > joint[-1, :, 1]+1e-6)), 'joint_optimal_decoder_max_recorded_decrease': float(-np.diff(joint[:, :, 1], axis=0).min()), 'baseline_codes': baseline_codes}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(args.output/'baselines.npz', **baselines, frozen_limit=fixed_limit)
    plot_runs(runs, baselines)
    print(json.dumps({key: value for key, value in summary.items() if key != 'baseline_codes'}, indent=2))


if __name__ == '__main__': main()
