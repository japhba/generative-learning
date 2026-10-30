"""Independently compute paired-gradient covariance using absorbed path moments."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np


def absorbed_outcomes(theta, world, start, goal, beta, horizon):
    """Stop the recursion at arrival; do not append dummy post-arrival coins."""
    output = []
    def visit(state, depth, mass, score):
        if state == goal or depth == horizon:
            output.append((mass, float(state == goal), score.copy()))
            return
        delta = theta[goal] - theta[state]
        probability = 1 / (1 + np.exp(-beta * np.sin(delta)))
        for action in [0, 1]:
            following_score = score.copy()
            eligibility = (action - probability) * beta * np.cos(delta)
            following_score[goal] += eligibility
            following_score[state] -= eligibility
            visit(world[state, 1 - action], depth + 1, mass * (probability if action else 1 - probability), following_score)
    visit(start, 0, 1., np.zeros(len(theta)))
    return output


def main():
    load_dotenv(Path.home() / '.env')
    base = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/orientation'
    out = base / 'audit'; out.mkdir(parents=True, exist_ok=True)
    data = np.load(base / 'trajectories.npz')
    reference = np.load(base / 'step_audit/enumeration.npz')
    summary = json.loads((base / 'step_audit/summary.json').read_text())
    theta, worlds, posterior, queries, hessian = data['initial'][0], data['worlds'], data['posterior'][0], data['queries'], data['hessian']
    mean_queries, second_queries, outcomes_per_query = [], [], []
    score_mean_errors = []
    for start, goal in queries:
        rows = [(weight * mass, reward, score) for world, weight in zip(worlds, posterior) for mass, reward, score in absorbed_outcomes(theta, world, start, goal, 2., 3)]
        mass = np.array([row[0] for row in rows])
        reward = np.array([row[1] for row in rows])
        score = np.array([row[2] for row in rows])
        assert abs(mass.sum() - 1) < 1e-14
        score_mean_errors.append(float(np.linalg.norm(mass @ score)))
        assert score_mean_errors[-1] < 1e-14
        p = float(mass @ reward)
        a = (mass * reward) @ score
        c = np.einsum('i,ij,ik->jk', mass, score, score)
        d = np.einsum('i,ij,ik->jk', mass * reward, score, score)
        # For independent same-query trajectories, g=(R1-R2)(S1-S2)/2:
        # E[g]=a and E[gg^T]=[(1-2p)D+pC]/2+aa^T, since E[S]=0.
        second_queries.append(.5 * ((1 - 2 * p) * d + p * c) + np.outer(a, a))
        mean_queries.append(a)
        outcomes_per_query.append(len(rows))
    mean = np.mean(mean_queries, axis=0)
    covariance = (np.mean(second_queries, axis=0) - np.outer(mean, mean)) / 32
    covariance_error = float(np.max(abs(covariance - reference['batch_covariance'])))
    assert covariance_error < 1e-15
    variance, curvature = float(np.trace(covariance)), float(np.trace(hessian @ covariance))
    assert abs(variance - summary['trace_covariance']) < 1e-15
    assert abs(curvature - summary['trace_hessian_covariance']) < 1e-15
    assert np.linalg.norm(mean) < 1e-14
    beta, horizon = 2., 3
    a1 = beta * np.sqrt(2)
    a2 = beta ** 2 / 2 + 2 * beta
    a3_loose = beta ** 3 / np.sqrt(2) + 3 * beta ** 2 / np.sqrt(2) + 2 * np.sqrt(2) * beta
    a3_sharp = 2 * np.sqrt(2) * beta ** 3 / (6 * np.sqrt(3)) + 3 * beta ** 2 / np.sqrt(2) + 2 * np.sqrt(2) * beta
    third = (horizon * a1) ** 3 + 3 * (horizon * a1) * (horizon * a2) + horizon * a3_loose
    sharp_third = (horizon * a1) ** 3 + 3 * (horizon * a1) * (horizon * a2) + horizon * a3_sharp
    gradient_bound = horizon * a1
    threshold = 3 * curvature / (third * gradient_bound * variance)
    eta = 1e-5
    lower = eta ** 2 * curvature / 2 - eta ** 3 * third * gradient_bound * variance / 6
    assert abs(third - summary['global_third_directional_derivative_bound']) < 1e-10
    assert abs(threshold - summary['positive_gain_rate_upper']) < 1e-16
    assert abs(lower - summary['example_expected_gain_lower']) < 1e-25
    # The same first-step sign is not evidence that reward feedback is necessary:
    # equal-energy isotropic zero-sum perturbations also see positive curvature.
    projector = np.eye(7) - np.ones((7, 7)) / 7
    isotropic_covariance = variance * projector / 6
    isotropic_curvature = float(np.trace(hessian @ isotropic_covariance))
    result = {'uk_time': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'independent_method': 'Recursively enumerate variable-length absorbed paths, then use single-trajectory moment identities for the paired covariance. This does not reuse the256pair enumeration or its post-arrival dummy-coin convention.', 'absorbed_outcome_counts_per_query': outcomes_per_query, 'score_mean_max_norm': max(score_mean_errors), 'full_mean_norm': float(np.linalg.norm(mean)), 'batch_covariance_max_difference': covariance_error, 'trace_covariance': variance, 'trace_hessian_covariance': curvature, 'global_third_directional_bound_conservative': float(third), 'global_third_directional_bound_sharper_logistic_constant': float(sharp_third), 'gradient_norm_bound': float(gradient_bound), 'sufficient_first_step_rate_upper': float(threshold), 'example_rate': eta, 'certified_first_expected_gain_lower': float(lower), 'matched_trace_isotropic_zero_sum_curvature': isotropic_curvature, 'proof_scope': 'The derivatives are global directional bounds at fixed beta2. At the symmetric initial policy, exact cyclic/gauge symmetry makes the mean gradient zero. Taylor plus E||g||^3<=G trCov proves positive expected posterior success for this first update and sufficiently small eta. The practical eta.1, later updates, per-world improvement, convergence, and necessity of reward-specific sampling are not covered. Indeed matched-energy isotropic perturbations have positive initial curvature too.'}
    sources = [Path(__file__), base / 'step_audit/source.py']
    result['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (out / 'covariance_results.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'independent_covariance.npz', batch_covariance=covariance, query_means=mean_queries, query_second_moments=second_queries, mean=mean, isotropic_covariance=isotropic_covariance)
    shutil.copy2(__file__, out / 'covariance_source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
