"""One-shot local-curvature controls for the binary-cause majority model."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from tqdm.auto import tqdm

from model_search_hierarchical_majority import CATEGORIES, D, FLOOR, INVERSE, Z, counts, law, parameter_law, parameter_vector

RIDGES = [0, .01, .03, .1, .3, 1, 3]
SCALES = [.25, .5, .75, 1, 1.5, 2, 3]
FAMILIES = ['initial_fisher', 'initial_newton', 'posterior_fisher', 'posterior_newton']
MENU = [(family, ridge, scale) for family in FAMILIES for ridge in RIDGES for scale in SCALES]
UNIFORM = law(.5, np.full((2, D), .5))


def base_initial(data, weak):
    frequencies = data / data.sum()
    weight = frequencies[INVERSE] * UNIFORM[1] / UNIFORM[2][INVERSE]
    base = np.clip(weight @ Z, FLOOR + .01, 1 - FLOOR - .01)
    p = FLOOR + (1 - 2 * FLOOR) * expit(logit((base - FLOOR) / (1 - 2 * FLOOR))[None] + np.array([-1, 1])[:, None] * weak)
    return .5, p


def curvature(pi, p, frequencies, kind):
    theta = parameter_vector(pi, p)
    observed, score = parameter_law(theta)
    g = frequencies @ score
    if kind == 'fisher':
        return theta, g, score.T @ (observed[:, None] * score)
    joint, latent, _ = law(pi, p)
    joint_score = np.zeros((2, len(Z), len(theta)))
    joint_score[0, :, 0] = -pi
    joint_score[1, :, 0] = 1 - pi
    joint_score[0, :, 1:1 + D] = Z - p[0]
    joint_score[1, :, 1 + D:] = Z - p[1]
    posterior_weight = joint * (frequencies[INVERSE] / observed[INVERSE])[None]
    s = joint_score.reshape(-1, len(theta))
    second = s.T @ (posterior_weight.ravel()[:, None] * s)
    mass = posterior_weight.sum(axis=1)
    diagonal = np.concatenate([[pi * (1 - pi)], (mass[:, None] * p * (1 - p)).ravel()])
    negative_hessian = np.diag(diagonal) + score.T @ (frequencies[:, None] * score) - second
    return theta, g, negative_hessian


def candidates(data, weak):
    frequencies = data / data.sum()
    initial_pi, initial_p = base_initial(data, weak)
    means, mass, _, _ = counts(initial_pi, initial_p, frequencies, data[INVERSE] > 0)
    posterior_pi = float(np.clip(mass[1], FLOOR, 1 - FLOOR))
    posterior_p = np.clip(means / mass[:, None], FLOOR, 1 - FLOOR)
    predictions = []
    for family in FAMILIES:
        pi, p = (initial_pi, initial_p) if family.startswith('initial') else (posterior_pi, posterior_p)
        theta, gradient, matrix = curvature(pi, p, frequencies, family.split('_')[-1])
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, 0)
        magnitude = values.mean()
        for ridge in RIDGES:
            denominator = values + ridge * magnitude
            inverse = np.zeros_like(values)
            active = denominator > 1e-12
            inverse[active] = 1 / denominator[active]
            direction = vectors @ (inverse * (vectors.T @ gradient))
            for scale in SCALES:
                proposal = np.clip(theta + scale * direction, logit(FLOOR), logit(1 - FLOOR))
                pi, p = float(expit(proposal[0])), expit(proposal[1:]).reshape(2, D)
                predictions.append(law(pi, p)[2])
    return predictions


def audit():
    rng = np.random.default_rng(20107999)
    theta = rng.normal(0, .4, 1 + 2 * D)
    p, score = parameter_law(theta)
    freq = rng.dirichlet(np.ones(len(p)))
    pi, rows = float(expit(theta[0])), expit(theta[1:]).reshape(2, D)
    _, grad, negative_hessian = curvature(pi, rows, freq, 'newton')
    numerical_g, numerical_h = [], []
    for j in range(len(theta)):
        delta = np.zeros_like(theta)
        delta[j] = 1e-5
        pp, sp = parameter_law(theta + delta)
        pm, sm = parameter_law(theta - delta)
        numerical_g.append(freq @ (np.log(pp) - np.log(pm)) / 2e-5)
        numerical_h.append((freq @ sp - freq @ sm) / 2e-5)
    errors = dict(gradient=float(np.max(np.abs(grad - numerical_g))), hessian=float(np.max(np.abs(negative_hessian + np.array(numerical_h).T))))
    assert max(errors.values()) < 1e-7
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    errors = audit()
    records = []
    for file in tqdm(sorted(args.directory.glob('world-*-initial.json')), desc='Hierarchy Fisher controls'):
        initial = json.loads(file.read_text())
        data = np.array(initial['data'])
        p = np.array(initial['p'])
        weak = .5 * (logit((p[1] - FLOOR) / (1 - 2 * FLOOR)) - logit((p[0] - FLOOR) / (1 - 2 * FLOOR)))
        rng = np.random.default_rng(initial['seed'] + 400000)
        order = rng.permutation(np.repeat(np.arange(len(data)), data))
        scores = np.zeros(len(MENU))
        for heldout in np.array_split(order, 4):
            held = np.bincount(heldout, minlength=len(data))
            for j, prediction in enumerate(candidates(data - held, weak)):
                scores[j] += held @ np.log(prediction)
        truth = np.array(initial['truth'])
        risks = np.array([truth @ np.log(truth / prediction) for prediction in candidates(data, weak)])
        selected = int(np.argmax(scores))
        practice_file = file.with_name(file.name.replace('-initial', '-practice'))
        practice = json.loads(practice_file.read_text())
        records.append(dict(seed=initial['seed'], practice_kl=practice['history'][-1]['kl'], selected_kl=float(risks[selected]), selected_option=MENU[selected], oracle_kl=float(risks.min()), families={family: float(risks[[j for j, item in enumerate(MENU) if item[0] == family][np.argmax(scores[[j for j, item in enumerate(MENU) if item[0] == family]])]]) for family in FAMILIES}, all_risks=risks.tolist(), all_cv=scores.tolist()))
    differences = np.array([r['practice_kl'] - r['selected_kl'] for r in records])
    rng = np.random.default_rng(73387)
    boot = differences[rng.integers(0, len(records), (20000, len(records)))].mean(axis=1)
    result = dict(audit=errors, practice_kl=float(np.mean([r['practice_kl'] for r in records])), selected_kl=float(np.mean([r['selected_kl'] for r in records])), oracle_kl=float(np.mean([r['oracle_kl'] for r in records])), difference=float(differences.mean()), confidence_interval=np.quantile(boot, [.025, .975]).tolist(), family_means={family: float(np.mean([r['families'][family] for r in records])) for family in FAMILIES}, records=records)
    (args.directory / 'fisher-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2))


if __name__ == '__main__':
    main()
