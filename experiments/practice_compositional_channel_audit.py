"""Posterior, trial-update and integration checks for the Boolean channel."""

import argparse
from itertools import product
import json
from pathlib import Path
import shutil

import numpy as np
from tqdm.auto import trange

from practice_compositional_channel import ACTION, BETA, CAUSE, FY, FZ, distributions, evaluate, field, step, tangent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    data = dict(np.load(args.base/'worlds.npz'))
    initial = dict(np.load(args.base/'initial_fit.npz'))
    joint = dict(np.load(args.base/'joint.npz'))
    reconstruction = dict(np.load(args.base/'reconstruction.npz'))
    baseline = dict(np.load(args.base/'baselines.npz'))
    # Recompute the gate posterior by enumerating all 16 truth tables and
    # using each raw observation, rather than the experiment's count formula.
    truth_tables = np.array(list(product([0, 1], repeat=4)))
    max_error = 0.
    for s in trange(len(data['channel']), desc='Full gate posterior audit'):
        posterior = []
        for j, (left, right) in enumerate(data['wiring'][s]):
            local = (2*ACTION[data['inputs'][s], left]+ACTION[data['inputs'][s], right]).astype(int)
            probabilities = .1+.8*truth_tables[:, local]
            observed = data['observed'][s, :, j]
            ll = (observed*np.log(probabilities)+(1-observed)*np.log1p(-probabilities)).sum(axis=1)
            mass = np.exp(ll-ll.max()); mass /= mass.sum()
            posterior.append(mass @ truth_tables)
        max_error = max(max_error, float(np.max(np.abs(np.array(posterior)-data['posterior'][s]))))
    assert max_error < 1e-13, max_error
    # Independent full enumeration of rewarded local Bernoulli residuals.
    w, v = joint['w'][-1, :1], joint['v'][-1, :1]
    small = {name: value[:1] for name, value in data.items()}
    g, r, pg, pr = [value[0] for value in distributions(w, v)]
    gw, gv = np.zeros_like(w[0]), np.zeros_like(v[0])
    reward_baseline = evaluate(w, v, small)[0, 0]
    for z, a in product(range(4), range(64)):
        for inferred in range(4):
            probability = g[z, a]*small['channel'][0, a]*r[:, inferred]/4
            rewarded = probability*(float(z == inferred)-reward_baseline)
            gw += rewarded.sum()*BETA*np.outer(ACTION[a]-pg[z], FZ[z])
            gv += BETA*np.einsum('y,yk,yn->kn', rewarded, CAUSE[inferred]-pr, FY)
    target_w, target_v = field(w, v, small, 'joint')
    enumeration_error = float(max(np.max(np.abs(tangent(gw[None], w)-target_w)), np.max(np.abs(tangent(gv[None], v)-target_v))))
    assert enumeration_error < 1e-13, enumeration_error
    small = {name: value[:4] for name, value in data.items()}
    w, v = initial['w'][-1, :4].copy(), initial['v'][-1, :4].copy()
    rate = json.loads((args.base/'protocol.json').read_text())['rate']
    for _ in trange(int(joint['steps'][-1])*2, desc='Half-step practice audit'):
        w, v = step(w, v, small, 'joint', rate/2)
    integration_error = float(np.max(np.abs(evaluate(w, v, small)-joint['records'][-1, :4])))
    assert integration_error < 1e-4, integration_error
    rng = np.random.default_rng(491223)
    final = joint['records'][-1, :, 1]
    comparisons = {}
    controls = {name: result[:, 0] for name, result in baseline.items()}
    controls['continued_reconstruction'] = reconstruction['records'][-1, :, 1]
    controls['frozen_recognizer'] = np.load(args.base/'frozen_recognizer.npz')['records'][-1, :, 1]
    controls['best_spectral_per_world'] = np.stack([baseline[name][:, 0] for name in ['raw_pair_svd', 'channel_covariance_svd', 'channel_spectral']]).max(axis=0)
    for name, control in controls.items():
        difference = final-control
        bootstrap = difference[rng.integers(len(final), size=(20000, len(final)))].mean(axis=1)
        comparisons[name] = {'practice_minus_control_mean': float(difference.mean()), 'paired_world_bootstrap_95ci': np.quantile(bootstrap, [.025, .975]).tolist(), 'practice_wins': int(np.sum(difference > 0))}
    result = {'posterior_error': max_error, 'local_trial_enumeration_error': enumeration_error, 'half_step_max_metric_error': integration_error, 'comparisons': comparisons, 'minimum_per_world_generator_increment': float(np.diff(joint['records'][:, :, 1], axis=0).min()), 'interpretation': 'Conditional Bayes joint-risk theorem; generator-only improvement is measured, not guaranteed. This is a finite baseline screen, not proof that no analytic shortcut exists. Covariance SVD was the selected primary shortcut comparator for the fresh 64-world follow-up. Best-of-spectral comparison is exploratory. No equivalence margin is assumed.'}
    (args.base/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    shutil.copy2(__file__, args.base/'audit_source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
