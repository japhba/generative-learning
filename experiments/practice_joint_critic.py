"""A joint Hebbian critic/sampler: finite-data Bayes risk and reconstruction audit.

CPU-only; no new observations after the sufficient statistic is formed. The
mean-field theorem is distinct from the finite-step Monte Carlo experiment.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def transpose(a): return np.swapaxes(a, -1, -2)


def risk(w, c):
    s = w @ transpose(w)
    signs, logdet = np.linalg.slogdet(s)
    assert np.all(signs == 1)
    return .5 * (logdet + np.trace(np.linalg.solve(s, c), axis1=-2, axis2=-1))


def drift(w, b, c, damping):
    s = w @ transpose(w)
    return s @ b @ w, c - s - damping * b


def rk4(w, b, c, damping, dt):
    k1, l1 = drift(w, b, c, damping)
    k2, l2 = drift(w + dt*k1/2, b + dt*l1/2, c, damping)
    k3, l3 = drift(w + dt*k2/2, b + dt*l2/2, c, damping)
    k4, l4 = drift(w + dt*k3, b + dt*l3, c, damping)
    return w + dt*(k1 + 2*k2 + 2*k3 + k4)/6, b + dt*(l1 + 2*l2 + 2*l3 + l4)/6


def sampled_step(w, b, c, damping, dt, pairs, rng, scramble):
    z = rng.standard_normal((2, w.shape[0], pairs, w.shape[-1]))
    x = z @ transpose(w)[None]
    r = .5 * np.einsum('asni,sij,asnj->asn', x, b, x)
    if scramble:
        z_reward = rng.standard_normal(z.shape)
        x_reward = z_reward @ transpose(w)[None]
        r = .5 * np.einsum('asni,sij,asnj->asn', x_reward, b, x_reward)
    reward_difference = r[0] - r[1]
    dw = .5 * (np.einsum('sn,sni,snj->sij', reward_difference, x[0], z[0]) - np.einsum('sn,sni,snj->sij', reward_difference, x[1], z[1])) / pairs
    generated_covariance = np.einsum('asni,asnj->sij', x, x) / (2*pairs)
    db = c - generated_covariance - damping*b
    return w + dt*dw, b + dt*db


def make_worlds(seeds, d, n, seed_start):
    rng = np.random.default_rng(seed_start)
    nu = 2*d + 4
    psi = (nu-d-1)*np.eye(d)
    normal = rng.standard_normal((seeds, nu, d)) / np.sqrt(nu-d-1)
    true_cov = np.linalg.inv(transpose(normal) @ normal)
    data = rng.standard_normal((seeds, n, d)) @ transpose(np.linalg.cholesky(true_cov))
    c = (psi + transpose(data) @ data) / (nu+n-d-1)
    left, _ = np.linalg.qr(rng.normal(size=(seeds, d, d)))
    right, _ = np.linalg.qr(rng.normal(size=(seeds, d, d)))
    w = (left * np.linspace(.65, 1.35, d)) @ transpose(right)
    return data, true_cov, c, w, nu, psi


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--seed-start', type=int, default=191026)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--dt', type=float, default=.005)
    parser.add_argument('--pairs', type=int, default=128)
    parser.add_argument('--damping', type=float, default=2.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).read_bytes()
    shutil.copy2(__file__, args.output/'source.py')
    data, true_cov, c, w0, nu, psi = make_worlds(args.seeds, 4, 48, args.seed_start)
    config = vars(args) | {'output': str(args.output), 'd': 4, 'n': 48, 'prior_df': nu, 'prior_scale': psi.tolist(), 'uk_started': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'source_sha256': hashlib.sha256(source).hexdigest(), 'target': 'Conditional Bayes Gaussian log loss under an inverse-Wishart prior; posterior predictive need not be Gaussian.', 'controls': ['mean-field RK4', 'sampled joint learning', 'independent reward control', 'zero-reconstruction baseline', 'direct covariance factorization'], 'selection': 'One exploratory mean-field 4D example preceded this script; these data and seeds are new. No test outcomes enter learning. No hyperparameter selection on this cohort.', 'stopping': 'Evaluate every 20 steps. Stop a condition when both excess Bayes risk and actual-world KL have log-log slopes of magnitude below .01 on the last 20 evaluations, after time 10. Nonfinite or tenfold population loss deterioration raises an error; retain all completed output.'}
    (args.output/'protocol.json').write_text(json.dumps(config, indent=2)+'\n')
    np.savez_compressed(args.output/'worlds.npz', data=data, true_cov=true_cov, c=c, w0=w0)
    optimal = .5*(np.linalg.slogdet(c)[1]+4)
    true_optimal = .5*(np.linalg.slogdet(true_cov)[1]+4)
    initial = risk(w0, c)
    names = ['mean_field', 'sampled', 'independent_reward']
    histories = {}
    for j, name in enumerate(tqdm(names, desc='Conditions')):
        rng = np.random.default_rng(args.seed_start + 100000 + j)
        w = w0.copy(); b = np.zeros_like(w)
        records = []
        for step in tqdm(range(args.steps+1), desc=name, leave=False):
            if step % 20 == 0:
                loss = risk(w, c)
                actual = risk(w, true_cov)-true_optimal
                energy = loss + .5*np.sum(b*b, axis=(-2, -1))
                assert np.isfinite(loss).all() and np.isfinite(actual).all(), (name, step)
                assert np.all(actual < 10*(risk(w0, true_cov)-true_optimal+1)), (name, step, actual)
                records.append((step, w.copy(), b.copy(), loss-optimal, actual, energy-initial))
                np.savez_compressed(args.output/f'{name}_checkpoint.npz', step=step, w=w, b=b, rng_state=json.dumps(rng.bit_generator.state))
                if step*args.dt >= 10 and len(records) >= 21:
                    times = np.array([r[0]*args.dt for r in records[-20:]])
                    bayes = np.array([r[3].mean() for r in records[-20:]])
                    actuals = np.array([r[4].mean() for r in records[-20:]])
                    slopes = [float(np.polyfit(np.log(times), np.log(values), 1)[0]) for values in (bayes, actuals)]
                    if max(abs(s) for s in slopes) < .01: break
            if step == args.steps: break
            if name == 'mean_field': w, b = rk4(w, b, c, args.damping, args.dt)
            else: w, b = sampled_step(w, b, c, args.damping, args.dt, args.pairs, rng, name == 'independent_reward')
        arrays = {'step': np.array([r[0] for r in records]), 'w': np.array([r[1] for r in records]), 'b': np.array([r[2] for r in records]), 'bayes_excess': np.array([r[3] for r in records]), 'true_kl': np.array([r[4] for r in records]), 'energy_change': np.array([r[5] for r in records])}
        np.savez_compressed(args.output/f'{name}.npz', **arrays)
        histories[name] = arrays
    summary = {'prior_df': nu, 'n': 48, 'worlds': args.seeds, 'initial_bayes_excess': float((initial-optimal).mean()), 'initial_true_kl': float((risk(w0, true_cov)-true_optimal).mean()), 'direct_factorization_true_kl': float((risk(np.linalg.cholesky(c), true_cov)-true_optimal).mean())}
    for name, arrays in histories.items():
        w = arrays['w'][-1]
        covariance_error = np.linalg.norm(w@transpose(w)-c, axis=(-2, -1))/np.linalg.norm(c, axis=(-2, -1))
        reconstruction = np.mean((data - data@transpose(np.linalg.inv(w))@transpose(w))**2, axis=(-2, -1))
        gain = arrays['bayes_excess'][0]-arrays['bayes_excess'][-1]
        summary[name] = {'final_bayes_excess': float(arrays['bayes_excess'][-1].mean()), 'final_true_kl': float(arrays['true_kl'][-1].mean()), 'bayes_worse_worlds': int(np.sum(gain < 0)), 'actual_worse_worlds': int(np.sum(arrays['true_kl'][-1] > arrays['true_kl'][0])), 'mean_bayes_gain': float(gain.mean()), 'se_bayes_gain': float(gain.std(ddof=1)/np.sqrt(args.seeds)), 'relative_covariance_error': float(covariance_error.mean()), 'maximum_reconstruction_mse': float(reconstruction.max()), 'maximum_energy_increase_from_start': float(arrays['energy_change'].max()), 'final_time': float(arrays['step'][-1]*args.dt)}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    figures = root/'tex/archive/figures'
    fig, ax = plt.subplots(layout='constrained')
    for name, label in [('mean_field', 'Joint mean field'), ('sampled', 'Joint sampled'), ('independent_reward', 'Independent reward')]:
        a = histories[name]; values = a['bayes_excess']; mean = values.mean(axis=1); se = values.std(axis=1, ddof=1)/np.sqrt(args.seeds)
        line, = ax.plot(a['step']*args.dt, mean, label=label)
        ax.fill_between(a['step']*args.dt, mean-1.96*se, mean+1.96*se, color=line.get_color(), alpha=.15)
    ax.axhline(summary['initial_bayes_excess'], linestyle=':', label='Zero reconstruction / frozen zero critic')
    ax.axhline(0, linestyle='--', label='Direct covariance factorization')
    ax.set(xlabel='Offline learning time', ylabel='Excess conditional Bayes log loss (nats)', title='Joint learning after 48 unlabelled observations')
    ax.legend(); fig.savefig(figures/'practice-joint-critic-learning.png'); plt.close(fig)
    fig, ax = plt.subplots(layout='constrained')
    a = histories['sampled']; eigvec = np.linalg.eigh(c[0])[1][:, -1]
    overlaps = np.array([float(np.linalg.svd(w[0])[0][:, 0]@eigvec)**2 for w in a['w']])
    ax.plot(a['step']*args.dt, overlaps, label='Leading generated/target axis overlap')
    ax.plot(a['step']*args.dt, np.linalg.norm(a['b'][:, 0], axis=(-2, -1)), label='Learned critic norm')
    ax.set(xlabel='Offline learning time', ylabel='Diagnostic value', title='First predeclared world: critic and learned weight direction')
    ax.legend(); fig.savefig(figures/'practice-joint-critic-representation.png'); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
