"""One learned binary cause, twelve stochastic atoms, and a fixed majority circuit."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy.special import expit, logit
from tqdm.auto import tqdm

from model_search_majority import CATEGORIES, D, FEATURES, FLOOR, INVERSE, POWERS, VISIBLE, Z, render

ETA = .5


def law(pi, p):
    joint = np.array([1 - pi, pi])[:, None] * np.exp(np.log(p) @ Z.T + np.log1p(-p) @ (1 - Z).T)
    latent = joint.sum(axis=0)
    observed = np.bincount(INVERSE, weights=latent, minlength=len(CATEGORIES))
    return joint, latent, observed


def parameter_vector(pi, p):
    return np.concatenate([[logit(pi)], logit(p).ravel()])


def parameter_law(theta):
    pi, p = float(expit(theta[0])), expit(theta[1:]).reshape(2, D)
    joint, latent, observed = law(pi, p)
    score = np.zeros((len(CATEGORIES), 1 + 2 * D))
    score[:, 0] = np.bincount(INVERSE, weights=joint[1], minlength=len(CATEGORIES)) / observed - pi
    for h in range(2):
        for j in range(D):
            score[:, 1 + h * D + j] = np.bincount(INVERSE, weights=joint[h] * (Z[:, j] - p[h, j]), minlength=len(CATEGORIES)) / observed
    return observed, score


def inverse_one(target, rng):
    """Targeted local constraint search stops at its first complete explanation."""
    candidate = np.full(D, -1, dtype=int)
    windows = [[(i + shift) % D for shift in [-2, -1, 0, 1, 2]] for i in range(D)]
    checks = [[i for i, window in enumerate(windows) if max(window) == j] for j in range(D)]
    order = rng.integers(0, 2, D)

    def search(j):
        if j == D:
            return True
        for bit in [order[j], 1 - order[j]]:
            candidate[j] = bit
            compatible = True
            for i in checks[j]:
                bits = candidate[windows[i]]
                middle = np.array([bits[k:k + 3].sum() >= 2 for k in range(3)])
                if int(middle.sum() >= 2) != target[i]:
                    compatible = False
                    break
            if compatible and search(j + 1):
                return True
        candidate[j] = -1
        return False

    assert search(0)
    assert np.array_equal(render(candidate[None])[0], target)
    return int(candidate @ POWERS)


def counts(pi, p, frequencies, included):
    joint, latent, observed = law(pi, p)
    cache_mass = np.bincount(INVERSE, weights=latent * included, minlength=len(CATEGORIES))
    observed_mask = frequencies > 0
    assert np.all(cache_mass[observed_mask] > 0)
    weight = np.zeros_like(latent)
    weight[included] = frequencies[INVERSE[included]] / cache_mass[INVERSE[included]]
    posterior = joint * weight[None]
    mass = posterior.sum(axis=1)
    means = posterior @ Z
    bound = frequencies[observed_mask] @ np.log(cache_mass[observed_mask])
    assert abs(mass.sum() - 1) < 1e-10
    return means, mass, float(bound), observed


def row_step(p, means, mass):
    s = (p - FLOOR) / (1 - 2 * FLOOR)
    gradient = (1 - 2 * FLOOR) * s * (1 - s) * (means / p - (mass - means) / (1 - p))
    new = FLOOR + (1 - 2 * FLOOR) * expit(logit(s) + ETA * gradient)
    before = np.sum(means * np.log(p) + (mass - means) * np.log1p(-p))
    after = np.sum(means * np.log(new) + (mass - means) * np.log1p(-new))
    assert after - before >= ETA * (1 - ETA / 8) * np.sum(gradient**2) - 1e-10
    return new


def proposal_scores(v, w):
    ph = .02 + .96 * expit(FEATURES @ v)
    scores = []
    for h in range(2):
        phi = np.column_stack([FEATURES, np.full(len(FEATURES), 2 * h - 1)])
        pz = .02 + .96 * expit(phi @ w.T)
        terms = Z * np.log(pz[INVERSE]) + (1 - Z) * np.log1p(-pz[INVERSE])
        scores.append(terms.sum(axis=1) + np.log(ph[INVERSE] if h else 1 - ph[INVERSE]))
    return np.array(scores)


def initialize(seed, n):
    rng = np.random.default_rng(seed)
    teacher_pi = .3 + .4 * rng.random()
    bias = rng.normal(0, .5, D)
    contrast = rng.normal(0, 1.5, D)
    teacher_p = FLOOR + (1 - 2 * FLOOR) * expit(bias[None] + np.array([-1, 1])[:, None] * contrast)
    teacher_joint, _, truth = law(teacher_pi, teacher_p)
    data = rng.multinomial(n, truth)
    freq = data / n
    uniform_joint, uniform_latent, uniform_x = law(.5, np.full((2, D), .5))
    uniform_weight = freq[INVERSE] * uniform_latent / uniform_x[INVERSE]
    base = np.clip(uniform_weight @ Z, FLOOR + .01, 1 - FLOOR - .01)
    weak_contrast = rng.normal(0, .15, D)
    p = FLOOR + (1 - 2 * FLOOR) * expit(logit((base - FLOOR) / (1 - 2 * FLOOR))[None] + np.array([-1, 1])[:, None] * weak_contrast)
    pi = .5
    joint, latent, observed = law(pi, p)
    initial_draws = rng.choice(len(Z), 1024, p=latent)
    included = np.zeros(len(Z), dtype=bool)
    included[initial_draws] = True
    included &= data[INVERSE] > 0
    for category in np.flatnonzero(data):
        if not np.any(included & (INVERSE == category)):
            included[inverse_one(VISIBLE[category].astype(int), rng)] = True
    v = np.zeros(FEATURES.shape[1])
    w = np.zeros((D, FEATURES.shape[1] + 1))
    w[:, 0] = (logit(p[0]) + logit(p[1])) / 2
    w[:, -1] = (logit(p[1]) - logit(p[0])) / 2
    return dict(seed=seed, teacher_pi=teacher_pi, teacher_p=teacher_p, teacher_joint=teacher_joint, truth=truth, data=data, freq=freq, pi=pi, p=p, included=included, v=v, w=w)


def train(initial, mode, steps):
    rng = np.random.default_rng(initial['seed'] + 100000)
    pi, p = initial['pi'], initial['p'].copy()
    included, v, w = initial['included'].copy(), initial['v'].copy(), initial['w'].copy()
    freq, truth = initial['freq'], initial['truth']
    initial_ll = float(freq @ np.log(law(pi, p)[2]))
    history, old_bound = [], -np.inf
    for step in range(steps + 1):
        means, mass, bound, observed = counts(pi, p, freq, included)
        assert bound >= old_bound - 1e-10
        old_bound = bound
        joint, latent, _ = law(pi, p)
        if step % 20 == 0:
            scores = proposal_scores(v, w)
            model_ce = float(-np.sum(joint * scores))
            valid = np.bincount(INVERSE, weights=np.exp(scores).sum(axis=0), minlength=len(CATEGORIES))
            record = dict(step=step, kl=float(truth @ np.log(truth / observed)), nll=float(-freq @ np.log(observed)), bound=bound, certificate=bound - initial_ll, model_ce=model_ce, valid_proposal=float(truth @ valid), cached_causes=int(included.sum()), reconstruction=0., contrast_norm=float(np.linalg.norm((logit(p[1]) - logit(p[0])) / 2)), pi=float(pi))
            history.append(record)
            if step >= 800 and len(history) >= 16:
                recent = history[-15:]
                logt = np.log([r['step'] for r in recent])
                slopes = [np.polyfit(logt, np.log([-r['bound'] for r in recent]), 1)[0], np.polyfit(logt, np.log([r['model_ce'] for r in recent]), 1)[0]]
                if max(abs(s) for s in slopes) < .002:
                    break
        if step == steps:
            break
        p = row_step(p, means, mass[:, None])
        pi = float(row_step(np.array(pi), np.array(mass[1]), np.array(1.)))
        joint, latent, _ = law(pi, p)
        dream_h = rng.random(256) < pi
        dream_z = rng.random((256, D)) < p[dream_h.astype(int)]
        dream_ids = (dream_z @ POWERS).astype(int)
        dream_phi = FEATURES[INVERSE[dream_ids]]
        phi_z = np.column_stack([dream_phi, 2 * dream_h - 1])
        if mode == 'practice':
            v += .3 * dream_phi.T @ (dream_h - expit(dream_phi @ v)) / len(dream_ids)
            w += .3 * (dream_z - expit(phi_z @ w.T)).T @ phi_z / len(dream_ids)
        if mode != 'fixed_cache':
            targets = rng.choice(np.flatnonzero(initial['data']), 128 if mode == 'practice' else 384)
            phi_h = FEATURES[targets]
            proposed_h = rng.random(len(targets)) < .02 + .96 * expit(phi_h @ v)
            phi_z = np.column_stack([phi_h, 2 * proposed_h - 1])
            probability = .02 + .96 * expit(phi_z @ w.T)
            proposed_z = rng.random(probability.shape) < probability
            ids = (proposed_z @ POWERS).astype(int)
            accepted = INVERSE[ids] == targets
            included[ids[accepted]] = True
    return dict(seed=initial['seed'], mode=mode, initial_likelihood=initial_ll, pi=pi, p=p.tolist(), v=v.tolist(), w=w.tolist(), initial_cache=np.flatnonzero(initial['included']).tolist(), final_cache=np.flatnonzero(included).tolist(), history=history)


def direct_controls(initial):
    freq, truth = initial['freq'], initial['truth']
    all_causes = initial['data'][INVERSE] > 0
    means, mass, _, _ = counts(initial['pi'], initial['p'], freq, all_causes)
    one_pi = float(np.clip(mass[1], FLOOR, 1 - FLOOR))
    one_p = np.clip(means / mass[:, None], FLOOR, 1 - FLOOR)
    predictions = dict(one_posterior=law(one_pi, one_p)[2])
    pi, p = initial['pi'], initial['p'].copy()
    em_history = []
    for step in range(1001):
        means, mass, bound, observed = counts(pi, p, freq, all_causes)
        if step % 20 == 0:
            em_history.append(dict(step=step, kl=float(truth @ np.log(truth / observed)), nll=float(-freq @ np.log(observed))))
        pi = float(np.clip(mass[1], FLOOR, 1 - FLOOR))
        p = np.clip(means / mass[:, None], FLOOR, 1 - FLOOR)
    predictions['iterative_em'] = law(pi, p)[2]
    for alpha in [.1, 1, 10, 100]:
        predictions[f'histogram_{alpha}'] = (initial['data'] + alpha / len(CATEGORIES)) / (initial['data'].sum() + alpha)
    return dict(results={name: dict(kl=float(truth @ np.log(truth / pred)), nll=float(-freq @ np.log(pred))) for name, pred in predictions.items()}, one_pi=one_pi, one_p=one_p.tolist(), em_pi=pi, em_p=p.tolist(), em_history=em_history)


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20107000)
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--steps', type=int, default=4000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args) | dict(out=str(args.out), time_uk=datetime.now(ZoneInfo('Europe/London')).isoformat(), initialization='1024 model draws plus targeted local DFS until every observed x has an exact cause; all completed valid causes retained', primary='practice versus one exact full posterior M-step and fixed initial cause cache')
    (args.out / 'protocol.json').write_text(json.dumps(config, indent=2))
    summary = []
    for world in tqdm(range(args.worlds), desc='Hierarchical-majority worlds'):
        initial = initialize(args.seed + world, args.n)
        serial = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in initial.items()}
        (args.out / f'world-{world}-initial.json').write_text(json.dumps(serial))
        controls = direct_controls(initial)
        (args.out / f'world-{world}-controls.json').write_text(json.dumps(controls))
        for mode in tqdm(['practice', 'frozen_recognition', 'fixed_cache'], desc='arms', leave=False):
            result = train(initial, mode, args.steps)
            (args.out / f'world-{world}-{mode}.json').write_text(json.dumps(result))
            summary.append(dict(world=world, mode=mode, **result['history'][-1]))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(2, 2)
    grid = np.arange(0, args.steps + 1, 20)
    for mode in ['practice', 'frozen_recognition', 'fixed_cache']:
        all_rows = [json.loads((args.out / f'world-{world}-{mode}.json').read_text())['history'] for world in range(args.worlds)]
        for ax, key in zip(axes.flat, ['kl', 'certificate', 'valid_proposal', 'contrast_norm']):
            curves = [np.interp(grid, [r['step'] for r in rows], [r[key] for r in rows]) for rows in all_rows]
            ax.plot(grid, np.mean(curves, axis=0), label=mode.replace('_', ' '))
            ax.set(xlabel='Offline updates', ylabel=key.replace('_', ' '))
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(args.out / 'learning.png')
    report = {mode: {key: float(np.mean([r[key] for r in summary if r['mode'] == mode])) for key in ['kl', 'certificate', 'valid_proposal', 'contrast_norm']} for mode in ['practice', 'frozen_recognition', 'fixed_cache']}
    control_rows = [json.loads((args.out / f'world-{world}-controls.json').read_text())['results'] for world in range(args.worlds)]
    report['controls'] = {key: float(np.mean([r[key]['kl'] for r in control_rows])) for key in control_rows[0]}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
