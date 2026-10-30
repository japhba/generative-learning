"""A direct spectral program construction using finite-state matrix products.

Neither moment calculation nor candidate scoring enumerates program words.
The stored exhaustive channel is used only for an independent audit.
"""

from datetime import datetime
from itertools import product
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import trange

from practice_program_channel import BETA, FEATURES, softmax, unit


def predict(p, matrix, posterior):
    state = np.zeros((len(p), 4, len(posterior), 8)); state[..., 0] = 1.
    for t in range(6):
        transition = np.einsum('bza,haxy->bzhxy', p[:, :, t], matrix)
        state = np.einsum('bzhx,bzhxy->bzhy', state, transition)
    return .25/8+.75*np.einsum('h,bzhy->bzy', posterior, state)


def cross_moment(matrix, posterior):
    average = matrix.mean(axis=1)
    powers = [np.broadcast_to(np.eye(8), (len(matrix), 8, 8))]
    for _ in range(6): powers.append(powers[-1] @ average)
    uniform = .25/8+.75*np.einsum('h,hy->y', posterior, powers[6][:, 0])
    moments = []
    for t in range(6):
        future = np.einsum('hx,haxy,hyq->haq', powers[t][:, 0], matrix, powers[5-t])
        moments.append((.25/8+.75*np.einsum('h,hay->ay', posterior, future)-uniform)/3)
    return np.array(moments).reshape(18, 8)


def candidate_weights(moment):
    u, _, _ = np.linalg.svd(moment, full_matrices=False)
    tetrahedron = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/np.sqrt(3)
    candidates = []
    for angles in product(np.arange(4)*np.pi/8, repeat=3):
        rotation = np.eye(3)
        for (i, j), angle in zip([(0, 1), (1, 2), (0, 2)], angles):
            part = np.eye(3)
            part[i, i] = np.cos(angle); part[j, j] = np.cos(angle)
            part[i, j] = -np.sin(angle); part[j, i] = np.sin(angle)
            rotation = rotation @ part
        weights = (tetrahedron @ rotation @ u[:, :3].T).reshape(4, 6, 3)
        weights -= weights.mean(axis=-1, keepdims=True)
        weights[np.linalg.norm(weights, axis=-1) < 1e-12] = [1., 0., 0.]
        candidates.append(unit(weights))
    return np.array(candidates)


def main():
    base = Path('/var/tmp/jbauer/generative_practice_program_channel/pilot')
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'scope': 'Supplementary direct analytic-control screen after inspecting the program pilot. Form raw-pair or exact posterior word-token/output cross moments; use top three left singular vectors and a tetrahedron of four cause codes, with 64 fixed rotations; normalize each position to the identical centered unit-sphere generator parameterization. Score with exact six-step matrix products under the fixed posterior world. No enumeration or sampling of programs occurs in construction or scoring. The exhaustive stored channel is used only to audit the computation. No claim of fresh replication.'}
    (base/'spectral_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, base/'spectral_source.py')
    data = dict(np.load(base/'worlds.npz'))
    joint = np.load(base/'extended/joint.npz')['records'][-1, :, 1]
    results = {name: [] for name in ['raw_svd', 'posterior_svd']}
    weights = {name: [] for name in results}
    max_moment_error, max_prediction_error = 0., 0.
    for s in trange(len(joint), desc='Analytic spectral program control'):
        matrix = np.eye(8)[data['primitives'][s]]
        moment = cross_moment(matrix, data['posterior'][s])
        reference = (FEATURES.reshape(729, 18)-1/3).T @ data['channel'][s]/729
        max_moment_error = max(max_moment_error, float(np.max(np.abs(moment-reference))))
        empirical_f = FEATURES[data['inputs'][s]].reshape(5, 18)
        empirical_y = np.eye(8)[data['observed'][s]]
        raw = (empirical_f-empirical_f.mean(axis=0)).T @ empirical_y/5
        for name, m in [('raw_svd', raw), ('posterior_svd', moment)]:
            candidates = candidate_weights(m)
            p = softmax(BETA*candidates)
            prediction = predict(p, matrix, data['posterior'][s])
            score = prediction.max(axis=1).sum(axis=-1)/4
            choice = int(score.argmax())
            weights[name].append(candidates[choice])
            g = np.exp(np.einsum('ntk,ztk->zn', FEATURES, np.log(p[choice])))
            max_prediction_error = max(max_prediction_error, float(np.max(np.abs(prediction[choice]-g @ data['channel'][s]))))
            truth_score = (g @ data['truth'][s]).max(axis=0).sum()/4
            results[name].append([score[choice], truth_score])
    assert max_moment_error < 1e-13, max_moment_error
    assert max_prediction_error < 1e-13, max_prediction_error
    results = {name: np.array(value) for name, value in results.items()}
    rng = np.random.default_rng(120921)
    comparisons = {}
    for name, value in results.items():
        difference = joint-value[:, 0]
        boot = difference[rng.integers(len(joint), size=(20000, len(joint)))].mean(axis=1)
        comparisons[name] = {'baseline_mean_posterior_true': value.mean(axis=0).tolist(), 'practice_minus_control_mean': float(difference.mean()), 'paired_world_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()}
    summary = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'moment_error': max_moment_error, 'prediction_error': max_prediction_error, 'comparisons': comparisons}
    (base/'spectral_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(base/'spectral_baselines.npz', **results)
    np.savez_compressed(base/'spectral_weights.npz', **{name: np.array(value) for name, value in weights.items()})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
