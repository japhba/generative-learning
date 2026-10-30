"""Independent finite-data, local-update, integration and shortcut audits."""

import json
from itertools import product
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import trange

from practice_discrete_channel import evaluate, gradient, plot_runs, softmax, step


def main():
    load_dotenv(Path.home()/'.env')
    base = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_discrete_channel/screen'
    worlds = np.load(base/'worlds.npz')
    runs = {name: dict(np.load(base/f'{name}.npz')) for name in ['joint', 'recognition_only', 'frozen_recognizer']}
    baselines = dict(np.load(base/'baselines.npz'))
    # Reconstruct the posterior from full categorical likelihoods, independently
    # of the sufficient-statistic reduction used by the experiment.
    incidence = np.eye(12)[worlds['supports']].sum(axis=1)
    candidates = .1/12 + .9*incidence/3
    posterior = softmax(worlds['counts'] @ np.log(candidates).T)
    posterior_channel = posterior @ candidates
    posterior_error = float(np.max(np.abs(posterior_channel-worlds['channel'])))
    assert posterior_error < 1e-13
    # Enumerate every possible trial at a sparse learned state. Studentized
    # Monte Carlo errors are misleading for actions almost never sampled.
    g_logits = runs['joint']['g'][3, :1]
    r_logits = runs['joint']['r'][3, :1]
    channel = worlds['channel'][:1]
    g, r = softmax(g_logits)[0], softmax(r_logits)[0]
    dg, dr = gradient(g_logits, r_logits, channel)
    baseline = evaluate(g[None], r[None], channel)[0, 0]
    exact_g, exact_r = np.zeros_like(g), np.zeros_like(r)
    for z, a, y, inferred in product(range(4), range(20), range(12), range(4)):
        probability = g[z, a]*channel[0, a, y]*r[y, inferred]/4
        feedback = float(z == inferred)-baseline
        exact_g[z] += probability*feedback*(np.eye(20)[a]-g[z])
        exact_r[y] += probability*feedback*(np.eye(4)[inferred]-r[y])
    enumeration_error = float(max(np.max(np.abs(exact_g-dg[0])), np.max(np.abs(exact_r-dr[0]))))
    assert enumeration_error < 1e-14
    sparse_minimum_expected_count = float(g.min()*819200/4)
    # Check random sampling separately where every action has adequate mass.
    rng = np.random.default_rng(129112)
    g_logits = worlds['initial_g'][:1]
    r_logits = rng.normal(0, .5, (1, 12, 4))
    g, r = softmax(g_logits)[0], softmax(r_logits)[0]
    dg, dr = gradient(g_logits, r_logits, channel)
    expected = np.r_[dg.ravel(), dr.ravel()]
    baseline = evaluate(g[None], r[None], channel)[0, 0]
    draws, batch = [], 8192
    for _ in trange(100, desc='Sampled local-rule audit'):
        z = rng.integers(4, size=batch)
        a = (rng.random((batch, 1)) > g[z].cumsum(axis=-1)).sum(axis=-1)
        y = (rng.random((batch, 1)) > channel[0, a].cumsum(axis=-1)).sum(axis=-1)
        inferred = (rng.random((batch, 1)) > r[y].cumsum(axis=-1)).sum(axis=-1)
        reward = (z == inferred).astype(float)-baseline
        sampled_g, sampled_r = np.zeros_like(g), np.zeros_like(r)
        np.add.at(sampled_g, z, reward[:, None]*(np.eye(20)[a]-g[z]))
        np.add.at(sampled_r, y, reward[:, None]*(np.eye(4)[inferred]-r[y]))
        draws.append(np.r_[sampled_g.ravel(), sampled_r.ravel()]/batch)
    draws = np.array(draws)
    zscores = (draws.mean(axis=0)-expected)/(draws.std(axis=0, ddof=1)/np.sqrt(len(draws)))
    assert np.max(np.abs(zscores)) < 5, {'max_zscore': float(np.max(np.abs(zscores))), 'worst_coordinate': int(np.abs(zscores).argmax()), 'expected': expected.tolist(), 'estimated': draws.mean(axis=0).tolist(), 'zscores': zscores.tolist()}
    # A finer numerical integration checks that the result is not a large-step artifact.
    g_half, r_half = worlds['initial_g'][:4].copy(), worlds['initial_r'][:4].copy()
    for _ in trange(int(runs['joint']['steps'][-1])*2, desc='Half-step integration audit'):
        g_half, r_half = step(g_half, r_half, worlds['channel'][:4], 5., 'joint')
    refined = evaluate(softmax(g_half), softmax(r_half), worlds['channel'][:4])
    integration_error = float(np.max(np.abs(refined-runs['joint']['records'][-1, :4, :2])))
    assert integration_error < 1e-4
    comparisons = {}
    rng = np.random.default_rng(128311)
    generator = runs['joint']['records'][-1, :, 1]
    for name in ['spectral', 'greedy', 'greedy_swap']:
        difference = baselines[name][:, 0]-generator
        bootstrap = difference[rng.integers(len(difference), size=(20000,len(difference)))].mean(axis=1)
        comparisons[name] = {'mean_direct_minus_joint': float(difference.mean()), 'paired_world_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(), 'direct_wins': int(np.sum(difference > 0))}
    results = {'posterior_predictive_max_absolute_error': posterior_error, 'enumerated_sparse_state_local_update_error': enumeration_error, 'sparse_state_minimum_expected_action_count_in_mc_audit': sparse_minimum_expected_count, 'sampled_local_update_max_abs_zscore': float(np.max(np.abs(zscores))), 'half_step_integration_max_metric_error': integration_error, 'minimum_recorded_generator_accuracy_change': float(np.diff(runs['joint']['records'][:, :, 1], axis=0).min()), 'comparisons': comparisons, 'interpretation': 'The SVD construction is a direct algebraic baseline; its mean is close but this interval does not establish equivalence. Greedy and one-swap are short deterministic searches, not reconstruction fits. Their superiority disproves an efficiency advantage for this screen, not the possibility that code search itself counts as practice. The initial sparse-state studentized Monte Carlo audit failed because one action had roughly one expected occurrence; its exact trial enumeration passes. Diagnostic output is retained.'}
    (base/'audit.json').write_text(json.dumps(results, indent=2)+'\n')
    shutil.copy2(__file__, base/'audit_source.py')
    plot_runs(runs, baselines)
    print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
