"""Interval arithmetic certificate for the first uncertain-orientation practice step."""

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from mpmath import iv
import numpy as np
from tqdm.auto import tqdm


def zeros(): return [[iv.mpf(0) for _ in range(7)] for _ in range(7)]


def bounds(value): return {'lower': str(value.a), 'upper': str(value.b)}


def main():
    load_dotenv(Path.home() / '.env')
    iv.dps = 40
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/orientation/interval_audit'
    out.mkdir(parents=True, exist_ok=True)
    theta = [4 * iv.pi * i / 7 for i in range(7)]
    queries = [(s, g) for s in range(7) for g in range(7) if s != g]
    hessian, pair_second = zeros(), zeros()
    mean = [iv.mpf(0) for _ in range(7)]
    success = iv.mpf(0)
    for start, goal in tqdm(queries, desc='Interval path sums'):
        rho = iv.mpf(0)
        m0, m1 = [iv.mpf(0) for _ in range(7)], [iv.mpf(0) for _ in range(7)]
        M0, M1 = zeros(), zeros()
        mass = iv.mpf(0)
        for orientation, weight in [(1, iv.mpf(9) / 10), (-1, iv.mpf(1) / 10)]:
            for actions in product([0, 1], repeat=3):
                state, probability = start, weight
                score, log_hessian = [iv.mpf(0) for _ in range(7)], zeros()
                for action in actions:
                    if state == goal:
                        probability /= 2
                        continue
                    difference = theta[goal] - theta[state]
                    sine, cosine = iv.sin(difference), iv.cos(difference)
                    p = 1 / (1 + iv.exp(-2 * sine))
                    local = (action - p) * 2 * cosine
                    second = -4 * p * (1 - p) * cosine**2 - 2 * (action - p) * sine
                    score[goal] += local; score[state] -= local
                    for i, sign_i in [(goal, 1), (state, -1)]:
                        for j, sign_j in [(goal, 1), (state, -1)]: log_hessian[i][j] += sign_i * sign_j * second
                    probability *= p if action else 1 - p
                    state = (state + orientation * (2 * action - 1)) % 7
                reward = state == goal
                mass += probability
                if reward: rho += probability
                vector, matrix = (m1, M1) if reward else (m0, M0)
                for i in range(7):
                    vector[i] += probability * score[i]
                    for j in range(7):
                        matrix[i][j] += probability * score[i] * score[j]
                        if reward: hessian[i][j] += probability * (score[i] * score[j] + log_hessian[i][j]) / 42
        assert mass.a <= 1 <= mass.b
        success += rho / 42
        for i in range(7):
            mean[i] += ((1 - rho) * m1[i] - rho * m0[i]) / 42
            for j in range(7): pair_second[i][j] += ((1 - rho) * M1[i][j] + rho * M0[i][j] - m1[i] * m0[j] - m0[i] * m1[j]) / 84
    for value in mean: assert value.a <= 0 <= value.b
    covariance = [[(pair_second[i][j] - mean[i] * mean[j]) / 32 for j in range(7)] for i in range(7)]
    trace = sum(covariance[i][i] for i in range(7))
    curvature = sum(hessian[i][j] * covariance[j][i] for i in range(7) for j in range(7))
    assert curvature > iv.mpf('0.00041833733264')
    assert trace < iv.mpf('0.00906688904')
    # Conservative uniform derivative bounds; sigmoid derivative factors <= 1/4.
    first = 6 * iv.sqrt(2)
    second = iv.mpf(18)
    third = 42 * iv.sqrt(2)
    third_bound = first**3 + 3 * first * second + third
    rate = iv.mpf(1) / 100000
    lower = rate**2 * curvature / 2 - rate**3 * third_bound * first * trace / 6
    assert lower > iv.mpf('6.44e-15')
    threshold = 3 * curvature / (third_bound * first * trace)
    assert threshold > iv.mpf('0.0000144545787')
    saved = np.load(out.parent / 'step_audit/enumeration.npz')
    numeric_hessian = np.array([[float(value.mid) for value in row] for row in hessian])
    numeric_covariance = np.array([[float(value.mid) for value in row] for row in covariance])
    hessian_error = float(abs(numeric_hessian - saved['hessian']).max())
    covariance_error = float(abs(numeric_covariance - saved['batch_covariance']).max())
    assert hessian_error < 1e-14 and covariance_error < 1e-14
    result = {'uk_verified': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'arithmetic': 'mpmath.iv,40decimal digits; rational posterior(.9,.1), exact interval pi, analytic log-path Hessians, independently aggregated reward-conditioned score moments.', 'queries': 42, 'worlds': 2, 'paths_per_world': 8, 'independent_pairs': 32, 'success_interval': bounds(success), 'mean_gradient_intervals_contain_zero': True, 'trace_covariance_interval': bounds(trace), 'trace_hessian_covariance_interval': bounds(curvature), 'third_derivative_bound_interval': bounds(third_bound), 'strictly_positive_rate_threshold_interval': bounds(threshold), 'example_rate': '0.00001', 'expected_first_step_gain_lower_interval': bounds(lower), 'saved_hessian_max_error': hessian_error, 'saved_covariance_max_error': covariance_error, 'scope': 'Certificate only for the first stochastic update from the exact posterior atomic optimum. Cyclic symmetry proves the exact mean gradient zero. The tiny conservative rate is not the empirical rate0.1. No guarantee of repeated-step monotonicity, convergence, or improvement in every true orientation.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, out / 'source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
