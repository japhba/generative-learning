"""Optional finite-rollout acceptance controller for monotonically certified practice blocks."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange

import practice_loop_ring as ring


def successes(theta, transition, queries, uniforms, beta):
    sid = np.arange(theta.shape[0])[:, None]
    state, goal = queries[:, :, 0].copy(), queries[:, :, 1]
    alive = state != goal
    for uniform in uniforms:
        probability = ring.sigmoid(beta * np.sin(theta[sid, goal] - theta[sid, state]))
        action = uniform < probability
        state = np.where(alive, transition[sid, state, 1 - action.astype(int)], state)
        alive &= state != goal
    return (state == goal).astype(float)


def main():
    load_dotenv(Path.home() / '.env')
    root = Path(__file__).resolve().parents[1]
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring'
    out = base / 'certified'
    out.mkdir(parents=True, exist_ok=True)
    archive = np.load(base / 'confirmation/trajectories.npz')
    seeds, blocks, updates, validation, delta, rate, beta, horizon, batch = 8, 8, 1000, 20000, .01, .1, 6., 6, 64
    theta = archive['initial'][:seeds].copy()
    theta = np.array([0., np.pi / 2, np.pi, -np.pi / 2])[np.round(theta / (np.pi / 2)).astype(int) % 4]
    truth, learned, queries = archive['truth'][:seeds], archive['learned'][:seeds], archive['queries'][:seeds]
    proposal_rng = np.random.default_rng(170052)
    validation_rng = np.random.default_rng(170053)
    threshold = np.sqrt(2 * np.log(blocks / delta) / validation)
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'seeds': seeds, 'blocks': blocks, 'proposal_updates_per_block': updates, 'validation_rollouts_per_world_per_block': validation, 'confidence_failure_per_world': delta, 'threshold': float(threshold), 'rate': rate, 'beta': beta, 'horizon': horizon, 'batch': batch, 'proposal_seed': 170052, 'validation_seed': 170053, 'specification': 'Optional controller, distinct from plain local REINFORCE. Each block proposes1000localupdates in temporary weights, then compares committed and candidate via20000fresh paired rollouts with independent validation randomness. Accept iff mean reward difference exceeds one-sided Hoeffding threshold sqrt(2log(K/delta)/N). Per-world high probability applies to every accepted block jointly. True dynamics and exact success are evaluation only; acceptance reads only frozen learned checker outcomes. Canonical four-valued principal phases avoid storing unwrapped initial order in generator parameters.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copy2(__file__, out / 'run_source.py'); shutil.copy2(Path(ring.__file__), out / 'shared_source.py')
    history = [theta.copy()]
    committed_success = [ring.exact(theta, truth, queries, horizon, beta)]
    provisional_curves, proposals, decisions, comparisons, exact_differences = [], [], [], [], []
    for block in trange(blocks, desc='Certified practice blocks'):
        candidate = theta.copy()
        curve = []
        for step in range(updates + 1):
            if step % 50 == 0: curve.append(ring.exact(candidate, truth, queries, horizon, beta))
            if step < updates: candidate += rate * ring.sampled(candidate, learned, queries, horizon, beta, proposal_rng, batch, 'practice')
        selected = validation_rng.integers(queries.shape[1], size=(seeds, validation))
        validation_queries = queries[np.arange(seeds)[:, None], selected]
        uniforms = validation_rng.random((horizon, seeds, validation))
        before = successes(theta, learned, validation_queries, uniforms, beta)
        after = successes(candidate, learned, validation_queries, uniforms, beta)
        difference = (after - before).mean(axis=1)
        accept = difference > threshold
        actual_difference = ring.exact(candidate, truth, queries, horizon, beta) - committed_success[-1]
        theta[accept] = candidate[accept]
        history.append(theta.copy()); committed_success.append(ring.exact(theta, truth, queries, horizon, beta))
        provisional_curves.append(curve); proposals.append(candidate); decisions.append(accept); comparisons.append(difference); exact_differences.append(actual_difference)
    history, committed_success, decisions, comparisons, exact_differences = map(np.array, (history, committed_success, decisions, comparisons, exact_differences))
    assert np.all(exact_differences[decisions] > 0), 'An empirical bound failed in this realization; retain and report.'
    summary = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'initial_success': committed_success[0].tolist(), 'final_success': committed_success[-1].tolist(), 'final_mean_success': float(committed_success[-1].mean()), 'accepted_blocks_per_world': decisions.sum(axis=0).tolist(), 'accepted_count': int(decisions.sum()), 'minimum_realized_accepted_true_gain': float(exact_differences[decisions].min()), 'minimum_empirical_lower_bound_on_accepted_gain': float((comparisons[decisions] - threshold).min()), 'scope': 'Population guarantee is per world over fresh validation randomness, conditional on the frozen table being correct; finite noisy atomic data gives a separate≥.9784519 correct-table probability. Combining gives≥.9684519 per-world guarantee by union bound. No simultaneous8-world guarantee at .99 claimed. Non-accepted temporary proposals can decrease success.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(out / 'trajectories.npz', committed_angles=history, committed_true_success=committed_success, proposed_angles=proposals, provisional_success_curves=provisional_curves, accepted=decisions, validation_mean_difference=comparisons, realized_true_difference=exact_differences, threshold=threshold)
    fig, ax = plt.subplots()
    for seed in range(seeds): ax.step(np.arange(blocks + 1), committed_success[:, seed], where='post', alpha=.5)
    ax.plot(np.arange(blocks + 1), committed_success.mean(axis=1), color='black', label='Mean committed generator')
    ax.set(xlabel='Proposed practice block', ylabel='True fresh-route success'); ax.legend()
    fig.tight_layout(); fig.savefig(root / 'tex/figures/practice-loop-ring-certified.png'); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
