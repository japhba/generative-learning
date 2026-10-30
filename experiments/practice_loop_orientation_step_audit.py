"""Enumerate the first paired practice update and bound its expected gain."""

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np

from practice_loop_ring import sigmoid


def main():
    load_dotenv(Path.home() / '.env')
    folder = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/orientation'
    out = folder / 'step_audit'; out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, out / 'source.py')
    data = np.load(folder / 'trajectories.npz')
    theta, worlds, posterior, queries, hessian = data['initial'][0], data['worlds'], data['posterior'][0], data['queries'], data['hessian']
    states, horizon, beta, batch = 7, 3, 2., 64
    probabilities, rewards, scores = [], [], []
    means, seconds = [], []
    for start, goal in queries:
        rows = []
        for world, weight in zip(worlds, posterior):
            for actions in product([0, 1], repeat=horizon):
                state, probability, score = int(start), float(weight), np.zeros(states)
                for action in actions:
                    if state == goal:
                        probability *= .5
                        continue
                    difference = theta[goal] - theta[state]
                    p = float(sigmoid(beta * np.sin(difference)))
                    local = (action - p) * beta * np.cos(difference)
                    score[goal] += local; score[state] -= local
                    probability *= p if action else 1 - p
                    state = int(world[state, 1 - action])
                rows.append((probability, float(state == goal), score))
        probability = np.array([row[0] for row in rows])
        reward = np.array([row[1] for row in rows])
        score = np.array([row[2] for row in rows])
        assert abs(probability.sum() - 1) < 1e-12
        assert np.linalg.norm(probability @ score) < 1e-12
        pair = (reward[:, None] - reward[None, :])[:, :, None] * (score[:, None] - score[None, :]) / 2
        mass = probability[:, None] * probability[None, :]
        means.append(np.einsum('ij,ijk->k', mass, pair))
        seconds.append(np.einsum('ij,ijk,ijl->kl', mass, pair, pair))
        probabilities.append(probability); rewards.append(reward); scores.append(score)
    mean = np.mean(means, axis=0)
    covariance = (np.mean(seconds, axis=0) - np.outer(mean, mean)) / (batch / 2)
    trace_covariance = float(np.trace(covariance))
    curvature = float(np.trace(hessian @ covariance))
    assert curvature > 0 and np.linalg.norm(mean) < 1e-12
    first = horizon * beta * np.sqrt(2)
    second = 2 * horizon * (beta + beta**2 / 4)
    third = 2 * np.sqrt(2) * horizon * (beta + 3 * beta**2 / 4 + beta**3 / 4)
    third_bound = first**3 + 3 * first * second + third
    threshold = 3 * curvature / (third_bound * first * trace_covariance)
    rate = 1e-5
    certified_gain = .5 * rate**2 * curvature - rate**3 * third_bound * first * trace_covariance / 6
    assert certified_gain > 0 and rate < threshold
    covariance_modes = [float(np.real(np.exp(-2j * np.pi * k * np.arange(states) / states) @ covariance @ np.exp(2j * np.pi * k * np.arange(states) / states) / states)) for k in range(states)]
    result = {'timestamp_uk': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'states': states, 'horizon': horizon, 'gain': beta, 'batch_trajectories': batch, 'independent_pairs': batch // 2, 'enumerated_queries': len(queries), 'worlds_per_query': len(worlds), 'full_action_strings_per_world': 2**horizon, 'pair_combinations_per_query': (len(worlds) * 2**horizon)**2, 'posterior': posterior.tolist(), 'mean_gradient_norm': float(np.linalg.norm(mean)), 'trace_covariance': trace_covariance, 'trace_hessian_covariance': curvature, 'quadratic_expected_gain_coefficient': curvature / 2, 'covariance_fourier_eigenvalues': covariance_modes, 'path_first_log_derivative_bound': float(first), 'path_second_log_derivative_bound': float(second), 'path_third_log_derivative_bound': float(third), 'global_third_directional_derivative_bound': float(third_bound), 'batch_gradient_norm_bound': float(first), 'positive_gain_rate_upper': float(threshold), 'example_certified_rate': rate, 'example_expected_gain_lower': float(certified_gain), 'scope': 'Only the first update from the exactly symmetric posterior atomic optimum. The positive bound concerns expected posterior route success over practice randomness. It neither covers practical learning rate0.1 nor proves repeated-step monotonicity, each-world improvement, or convergence. Full paths include independent dummy fair coins after absorption, with zero eligibility, to retain exactly8strings/world without double counting.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'enumeration.npz', theta=theta, worlds=worlds, posterior=posterior, queries=queries, probabilities=probabilities, rewards=rewards, scores=scores, mean_gradient=mean, batch_covariance=covariance, hessian=hessian)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
