"""Exact finite-cycle separation and all-winding local stability; no training."""

from datetime import datetime
import json
from math import gcd
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import trange

from practice_loop_ring import atomic, exact, sigmoid
from practice_loop_ring_theory_audit import directional


def cycle(n):
    states = np.arange(n)
    transitions = np.stack([(states + 1) % n, (states - 1) % n], axis=1)
    queries = np.array([(s, g) for s in states for g in states if s != g and g not in transitions[s]])
    return transitions, queries


def greedy_success(n, k, horizon):
    residues = k * np.arange(n) % n
    probability = np.where(2 * residues < n, 1., 0.)
    probability[(residues == 0) | (2 * residues == n)] = .5
    value = np.zeros(n)
    value[0] = 1
    for _ in range(horizon):
        value = probability * np.roll(value, 1) + (1 - probability) * np.roll(value, -1)
        value[0] = 1
    return float(value[2:n - 1].mean())


def main():
    load_dotenv(Path.home() / '.env')
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_ring/separation_audit'
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for source in [Path(__file__), root / 'experiments/practice_loop_ring.py', root / 'experiments/practice_loop_ring_theory_audit.py']:
        shutil.copy2(source, out / source.name)
    beta = 6.
    rows = []
    rng = np.random.default_rng(104301)
    for n in trange(3, 41, desc='Exact cycle classification'):
        eigenvalues = np.sin(2 * np.pi * np.arange(n) / n)
        maxima = np.flatnonzero(np.isclose(eigenvalues, eigenvalues.max(), rtol=0, atol=1e-13))
        transitions, queries = cycle(n)
        lower = np.logaddexp(0, -beta * eigenvalues.max())
        random_angles = rng.uniform(-np.pi, np.pi, (100, n))
        random_loss = atomic(random_angles, np.broadcast_to(transitions, (100, n, 2)), beta)[0]
        assert random_loss.min() >= lower - 1e-12
        for k in maxima:
            theta = 2 * np.pi * k * np.arange(n) / n
            loss, gradient = atomic(theta[None], transitions[None], beta)
            assert abs(loss[0] - lower) < 1e-12
            assert np.linalg.norm(gradient) < 1e-12
            row = {'n': n, 'winding': int(k), 'embedding_period': n // gcd(n, int(k)), 'aliased': gcd(n, int(k)) > 1, 'atomic_nll': float(loss[0]), 'rayleigh_maximum': float(eigenvalues.max())}
            if n >= 4:
                horizon = n // 2
                success = float(exact(theta[None], transitions[None], queries[None], horizon, beta)[0])
                once = 2 * np.pi * np.arange(n) / n
                once_success = float(exact(once[None], transitions[None], queries[None], horizon, beta)[0])
                row.update(horizon=horizon, success=success, once_wrapped_success=once_success, greedy_success=greedy_success(n, int(k), horizon))
            if n >= 7:
                assert n / 6 < k < n / 3
                chi = 1. if 4 * k < n else .5 if 4 * k == n else 0.
                formula = 2 * chi / (n - 3)
                finite_upper = 2 * sigmoid(beta * np.sin(4 * np.pi * k / n)) / (n - 3) + horizon * sigmoid(beta * np.sin(6 * np.pi * k / n))
                minimum_margin = np.sin(2 * np.pi / n) if n % 2 == 0 else np.sin(np.pi / n)
                finite_lower = float(sigmoid(beta * minimum_margin) ** horizon)
                assert abs(row['greedy_success'] - formula) < 1e-12
                assert success <= finite_upper + 1e-12
                assert once_success >= finite_lower - 1e-12
                row.update(greedy_formula=formula, finite_upper=float(finite_upper), once_wrapped_lower=finite_lower)
            rows.append(row)
    n, horizon = 12, 6
    transitions, queries = cycle(n)
    basis = np.eye(n)
    spectra, arrays = [], {}
    for winding in trange(1, 6, desc='Winding Hessians'):
        theta = 2 * np.pi * winding * np.arange(n) / n
        diagonal = np.array([directional(theta, v, transitions, queries, horizon, beta)[2] for v in basis])
        hessian = np.diag(diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                joint = directional(theta, basis[i] + basis[j], transitions, queries, horizon, beta)[2]
                hessian[i, j] = hessian[j, i] = (joint - diagonal[i] - diagonal[j]) / 2
        success, gradient = exact(theta[None], transitions[None], queries[None], horizon, beta, True)
        epsilon = 1e-5
        numerical = np.column_stack([(exact((theta + epsilon * v)[None], transitions[None], queries[None], horizon, beta, True)[1][0] - exact((theta - epsilon * v)[None], transitions[None], queries[None], horizon, beta, True)[1][0]) / (2 * epsilon) for v in basis])
        error = float(abs(hessian - numerical).max())
        assert error < 1e-7
        assert np.linalg.norm(gradient) < 1e-12
        eigenvalues = np.linalg.eigvalsh(hessian)
        modes = []
        for k in range(n):
            vector = np.exp(2j * np.pi * k * np.arange(n) / n) / np.sqrt(n)
            value = float(np.real(vector.conj() @ hessian @ vector))
            assert np.linalg.norm(hessian @ vector - value * vector) < 1e-12
            modes.append(value)
        spectra.append({'winding': winding, 'success': float(success[0]), 'gradient_norm': float(np.linalg.norm(gradient)), 'finite_difference_error': error, 'fourier_eigenvalues': modes, 'positive_directions': int((eigenvalues > 1e-10).sum()), 'negative_directions': int((eigenvalues < -1e-10).sum()), 'zero_directions': int((abs(eigenvalues) <= 1e-10).sum())})
        arrays[f'hessian_winding_{winding}'] = hessian
        arrays[f'phase_winding_{winding}'] = theta
    identity_checks = []
    for n, k in [(7, 2), (9, 2), (10, 3), (11, 3), (13, 3)]:
        phase = 2 * np.pi * np.arange(n) / n
        unit = np.stack([np.cos(phase), np.sin(phase)])
        permutation = k * np.arange(n) % n
        reordered = unit[:, permutation]
        decoder = rng.normal(size=(n, 2))
        original_logits = decoder @ unit
        changed_logits = decoder[permutation] @ reordered
        ce = lambda logits: float((np.logaddexp.reduce(logits, axis=0) - np.diag(logits)).mean())
        ce_error = abs(ce(original_logits) - ce(changed_logits))
        norm_error = abs(np.linalg.norm(decoder) - np.linalg.norm(decoder[permutation]))
        assert ce_error < 1e-12 and norm_error < 1e-12
        for alpha in [1., 6., 50.]:
            logits = alpha * reordered.T @ reordered
            margin = 1 - np.cos(2 * np.pi / n)
            upper = np.log1p((n - 1) * np.exp(-alpha * margin))
            loss = ce(logits)
            assert (logits.argmax(axis=0) == np.arange(n)).all()
            assert loss <= upper + 1e-12
            identity_checks.append({'n': n, 'winding': k, 'gain': alpha, 'identity_ce': loss, 'identity_ce_upper': float(upper), 'classification_accuracy': 1., 'permutation_ce_error': ce_error, 'permutation_norm_error': float(norm_error)})
    visible, hidden, samples = 7, 2, 31
    U = rng.normal(size=(hidden, visible))
    M = rng.normal(size=(hidden, hidden)); M = (M + M.T) / 2
    nu = rng.normal(size=hidden)
    x = np.eye(visible)[rng.integers(visible, size=samples)].T
    dx = np.eye(visible)[rng.integers(visible, size=samples)].T - x
    def lagrangian(matrix):
        y, dy = matrix @ x, matrix @ dx
        return .5 * (dy * dy).sum() / samples + .5 * np.trace(M @ (np.eye(hidden) - y @ y.T / samples)) + nu @ y.mean(axis=1)
    y, dy = U @ x, U @ dx
    expected = (dy @ dx.T - (M @ y - nu[:, None]) @ x.T) / samples
    numerical = np.empty_like(U)
    for i in range(hidden):
        for j in range(visible):
            perturbation = np.zeros_like(U); perturbation[i, j] = 1e-5
            numerical[i, j] = (lagrangian(U + perturbation) - lagrangian(U - perturbation)) / 2e-5
    sfa_error = float(abs(expected - numerical).max())
    assert sfa_error < 1e-8
    result = {'timestamp_uk': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'beta': beta, 'classification': rows, 'winding_hessians': spectra, 'identity_checks': identity_checks, 'local_sfa_lagrangian_gradient_error': sfa_error, 'scope': 'Exact finite-horizon population calculations on an oriented cycle; no fitted world, no new training, no inference of global convergence from local Hessians or SFA saddle directions.'}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(out / 'hessians.npz', **arrays)
    print(json.dumps({'output': str(out), 'winding_hessians': spectra}, indent=2))


if __name__ == '__main__': main()
