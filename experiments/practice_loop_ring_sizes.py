"""Predeclared size stress test of naturally pretrained cyclic composers."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange

import practice_loop_ring as ring


def metrics(theta, truth, learned, queries, horizon, beta):
    return np.stack([ring.exact(theta, truth, queries, horizon, beta), ring.exact(theta, learned, queries, horizon, beta), ring.atomic(theta, learned, beta)[0]], axis=1)


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/sizes_learned_queries'
    out.mkdir(parents=True, exist_ok=True)
    sizes, seeds, pretrain_steps, steps, cadence = [8, 10, 12, 14, 16], 16, 5000, 6000, 100
    beta, rate, batch, observations, corruption = 6., .1, 64, 5, .05
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'sizes': sizes, 'seeds_per_size': seeds, 'world_seed_start': 991000, 'training_seed_start': 993000, 'pretraining_updates': pretrain_steps, 'maximum_practice_updates': steps, 'evaluation_cadence': cadence, 'beta': beta, 'rate': rate, 'atomic_rate': 1., 'batch': batch, 'observations_per_context': observations, 'corruption': corruption, 'horizon': 'states // 2', 'metrics': ['true_success', 'internal_success', 'atomic_nll'], 'scope': 'Secondary robustness test fixed before execution, fresh finite worlds and random phase initializations. Sizes include cycles without the exact fourfold alias optimum. Same learning rates and batch size as twelve-state model; no tuning by size. Every world retained, including erroneous learned transition tables. Queries exclude LEARNED one-step neighbors only, and the task law uses no true transitions. Different per-world query counts are equalized by exact integer repetition to their least common multiple, preserving uniform per-world sampling. Cohort stops after three stationary checks on both internal failure-risk and atomic loss, using two ten-evaluation log-log windows after 2000 updates.', 'audit_amendment': 'The initial size test, preserved under sizes/, defined task queries using true neighbors. An independent audit identified this as extra topology information through the supplied query catalogue. This run changes only query construction to use finite learned transitions. All seeds, rates, sizes, budgets, and outcomes are retained; this is an information-access correction, not outcome tuning.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py'); shutil.copy2(Path(ring.__file__), out / 'shared_source.py')
    summaries = {}
    fig, axes = plt.subplots(2, 1)
    for d in tqdm(sizes, desc='Cycle sizes'):
        horizon = d // 2
        truth = np.zeros((seeds, d, 2), int); counts = np.zeros((seeds, d, 2, d), int)
        ordering, queries = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(991000 + 100 * d + seed)
            order = rng.permutation(d); ordering.append(order)
            truth[seed, order, 0] = np.roll(order, -1); truth[seed, order, 1] = np.roll(order, 1)
            for s in range(d):
                for a in range(2):
                    sample = np.full(observations, truth[seed, s, a])
                    noisy = rng.random(observations) < corruption
                    sample[noisy] = rng.integers(d, size=noisy.sum())
                    np.add.at(counts[seed, s, a], sample, 1)
        learned, ordering = counts.argmax(axis=3), np.array(ordering)
        queries = [np.array([(s, g) for s in range(d) for g in range(d) if s != g and g not in model[s]]) for model in learned]
        query_count = np.lcm.reduce([len(q) for q in queries])
        queries = np.array([np.tile(q, (query_count // len(q), 1)) for q in queries])
        rng = np.random.default_rng(993000 + d)
        theta = rng.uniform(-np.pi, np.pi, (seeds, d))
        pretraining, pretrain_angles = [], []
        for step in trange(pretrain_steps + 1, desc=f'{d} states: atomic training', leave=False):
            if step % cadence == 0:
                pretraining.append(metrics(theta, truth, learned, queries, horizon, beta)); pretrain_angles.append(theta.copy())
            if step < pretrain_steps: theta += ring.atomic(theta, learned, beta)[1]
        angles = {name: theta.copy() for name in ['practice', 'atomic', 'shuffled']}
        curves = {name: [] for name in angles}; histories = {name: [] for name in angles}
        active = {name: True for name in angles}; stationary = {name: 0 for name in angles}; stopping = {}
        checkpoints = np.arange(0, steps + 1, cadence)
        for step in trange(steps + 1, desc=f'{d} states: offline learning', leave=False):
            if step % cadence == 0:
                for name, phase in angles.items():
                    curve = metrics(phase, truth, learned, queries, horizon, beta)
                    curves[name].append(curve); histories[name].append(phase.copy())
                    if step >= 2000 and active[name]:
                        history = np.array(curves[name])
                        risk = np.stack([1 - history[:, :, 1].mean(axis=1), history[:, :, 2].mean(axis=1)], axis=1)
                        slopes = np.array([[np.polyfit(np.log(checkpoints[:len(history)][segment]), np.log(risk[segment, metric]), 1)[0] for metric in range(2)] for segment in [slice(-20, -10), slice(-10, None)]])
                        stationary[name] = stationary[name] + 1 if np.all(np.abs(slopes) < .01) else 0
                        if stationary[name] == 3:
                            active[name] = False; stopping[name] = {'step': step, 'reason': 'internal_objectives_stationary', 'log_log_slopes': slopes.tolist()}
            if step == steps: break
            for name, phase in angles.items():
                if not active[name]: continue
                phase += ring.atomic(phase, learned, beta)[1] if name == 'atomic' else rate * ring.sampled(phase, learned, queries, horizon, beta, rng, batch, 'practice' if name == 'practice' else 'shuffled')
                assert np.isfinite(phase).all()
        for name in angles:
            if active[name]: stopping[name] = {'step': steps, 'reason': 'budget'}
        curves, pretraining = {name: np.array(value) for name, value in curves.items()}, np.array(pretraining)
        difference = curves['practice'][-1, :, 0] - curves['atomic'][-1, :, 0]
        bootstrap_rng = np.random.default_rng(994000 + d)
        bootstrap = difference[bootstrap_rng.integers(seeds, size=(10000, seeds))].mean(axis=1)
        target = np.empty_like(theta); target[np.arange(seeds)[:, None], ordering] = 2 * np.pi * np.arange(d) / d
        summaries[str(d)] = {'checker_accuracy_by_world': np.mean(learned == truth, axis=(1, 2)).tolist(), 'pretrained': pretraining[-1].mean(axis=0).tolist(), 'final': {name: value[-1].mean(axis=0).tolist() for name, value in curves.items()}, 'practice_minus_atomic': {'mean': float(difference.mean()), 'bootstrap95': np.quantile(bootstrap, [.025, .975]).tolist(), 'by_seed': difference.tolist()}, 'passive_best_common_true_success': float(np.concatenate([pretraining[:, :, 0], curves['atomic'][1:, :, 0]]).mean(axis=1).max()), 'ideal_winding_one_true_success': float(ring.exact(target, truth, queries, horizon, beta).mean()), 'stopping': stopping}
        np.savez_compressed(out / f'states{d}.npz', steps=checkpoints, pretrain_steps=np.arange(0, pretrain_steps + 1, cadence), truth=truth, learned=learned, queries=queries, ordering=ordering, counts=counts, pretraining=pretraining, pretrain_angles=pretrain_angles, **{f'curves_{name}': value for name, value in curves.items()}, **{f'angles_{name}': value for name, value in histories.items()})
        line, = axes[0].plot(checkpoints, curves['practice'][:, :, 0].mean(axis=1), label=f'{d} states')
        axes[0].plot(checkpoints, curves['atomic'][:, :, 0].mean(axis=1), color=line.get_color(), linestyle=':')
        axes[1].plot(checkpoints, curves['practice'][:, :, 2].mean(axis=1), color=line.get_color())
        axes[1].plot(checkpoints, curves['atomic'][:, :, 2].mean(axis=1), color=line.get_color(), linestyle=':')
        (out / 'summary.json').write_text(json.dumps(protocol | {'results': summaries}, indent=2) + '\n')
    axes[0].set(ylabel='Fresh-route success', title='Practice (solid) and original reconstruction (dotted)'); axes[0].legend()
    axes[1].set(xlabel='Offline updates after atomic pretraining', ylabel='Atomic reconstruction NLL', yscale='log')
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-sizes.png'); plt.close(fig)
    (out / 'summary.json').write_text(json.dumps(protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'results': summaries}, indent=2) + '\n')
    print(json.dumps({d: {'pretrained': value['pretrained'], 'final': value['final'], 'effect': value['practice_minus_atomic']['mean'], 'interval': value['practice_minus_atomic']['bootstrap95']} for d, value in summaries.items()}, indent=2))


if __name__ == '__main__': main()
