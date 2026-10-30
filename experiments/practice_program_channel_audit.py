"""Continue unsettled pilot flows, then audit program risk independently."""

from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import trange

from practice_program_channel import BETA, FEATURES, WORDS, distributions, evaluate, field, fit, plot, softmax, step, tangent


def forward(p, primitives, posterior):
    matrix = np.eye(8)[primitives]
    state = np.zeros((len(p), 4, len(posterior[0]), 8)); state[..., 0] = 1.
    for t in range(6):
        transition = np.einsum('wza,whaxy->wzhxy', p[:, :, t], matrix)
        state = np.einsum('wzhx,wzhxy->wzhy', state, transition)
    return .25/8+.75*np.einsum('wh,wzhy->wzy', posterior, state)


def main():
    base = Path('/var/tmp/jbauer/generative_practice_program_channel/pilot')
    output = base/'extended'
    output.mkdir(exist_ok=False)
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'scope': 'The joint, frozen-recognition and recognition-only pilot flows hit the 4000-step cap without satisfying stationarity. Continue their final states for at most 12000 more steps with identical rates, objective and stopping threshold. Original pilot trajectories are retained. This is an exploratory same-world follow-up, not a replication.'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, output/'audit_source.py')
    shutil.copy2(Path(__file__).with_name('practice_program_channel.py'), output/'model_source.py')
    data = dict(np.load(base/'worlds.npz'))
    names = ['initial_fit', 'joint', 'reconstruction', 'frozen_recognizer', 'recognition_only']
    runs = {name: dict(np.load(base/f'{name}.npz')) for name in names}
    for name in ['joint', 'frozen_recognizer', 'recognition_only']:
        previous = runs[name]
        run = fit(previous['w'][-1], previous['v'][-1], data, name, 12000, .1, int(previous['steps'][-1]))
        combined = {key: np.concatenate([previous[key], run[key][1:]]) for key in ['records', 'steps', 'w', 'v']}
        combined['stationary'] = run['stationary']
        np.savez_compressed(output/f'{name}.npz', **combined)
        runs[name] = combined
    joint = runs['joint']
    g, r, p = distributions(joint['w'][-1], joint['v'][-1])
    received = forward(p, data['primitives'], data['posterior'])
    matrix_error = float(np.max(np.abs(received-g @ data['channel'])))
    assert matrix_error < 1e-13, matrix_error
    # Independent posterior likelihood: execute each original program under
    # every hypothesis, with no use of the stored endpoint table.
    posterior = []
    for s in range(len(g)):
        likelihood = np.ones(64)
        for program, observed in zip(WORDS[data['inputs'][s]], data['observed'][s]):
            states = np.zeros(64, dtype=int)
            for action in program: states = data['primitives'][s, np.arange(64), action, states]
            likelihood *= .25/8+.75*(states == observed)
        posterior.append(likelihood/likelihood.sum())
    posterior_error = float(np.max(np.abs(posterior-data['posterior'])))
    assert posterior_error < 1e-13, posterior_error
    # Exact expected local trial update, including a nonzero reward baseline.
    w, v = joint['w'][-1, :1], joint['v'][-1, :1]
    small = {name: value[:1] for name, value in data.items()}
    gs, rs, ps = [value[0] for value in distributions(w, v)]
    gw, gv = np.zeros_like(w[0]), np.zeros_like(v[0])
    b = evaluate(w, v, small)[0, 0]
    for z in range(4):
        for n in range(len(WORDS)):
            for inferred in range(4):
                probability = gs[z, n]*small['channel'][0, n]*rs[:, inferred]/4
                rewarded = probability*(float(z == inferred)-b)
                gw[z] += BETA*rewarded.sum()*(FEATURES[n]-ps[z])
                gv += BETA*rewarded[:, None]*(np.eye(4)[inferred]-rs)
    target_w, target_v = field(w, v, small, 'joint')
    local_error = float(max(np.max(np.abs(tangent(gw[None], w)-target_w)), np.max(np.abs(tangent(gv[None], v)-target_v))))
    assert local_error < 1e-13, local_error
    # Refinement over the first 100 offline steps, where changes are largest.
    small = {name: value[:4] for name, value in data.items()}
    w, v = runs['initial_fit']['w'][-1, :4].copy(), runs['initial_fit']['v'][-1, :4].copy()
    for _ in trange(200, desc='Half-step integration audit'): w, v = step(w, v, small, 'joint', .05)
    reference = np.flatnonzero(joint['steps'] == 100).item()
    integration_error = float(np.max(np.abs(evaluate(w, v, small)-joint['records'][reference, :4])))
    assert integration_error < 1e-4, integration_error
    baseline = dict(np.load(base/'baselines.npz'))
    final = joint['records'][-1, :, 1]
    controls = {name: value[:, 0] for name, value in baseline.items()}
    controls['frozen_recognizer'] = runs['frozen_recognizer']['records'][-1, :, 1]
    controls['continued_reconstruction'] = runs['reconstruction']['records'][-1, :, 1]
    rng = np.random.default_rng(2010269)
    comparisons = {}
    for name, control in controls.items():
        difference = final-control
        boot = difference[rng.integers(len(final), size=(20000, len(final)))].mean(axis=1)
        comparisons[name] = {'practice_minus_control_mean': float(difference.mean()), 'paired_world_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(), 'practice_wins': int(np.sum(difference > 0))}
    result = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'matrix_product_prediction_error': matrix_error, 'posterior_error': posterior_error, 'local_trial_update_error': local_error, 'half_step_metric_error': integration_error, 'runs': {name: {'last_step': int(run['steps'][-1]), 'stationary': bool(run['stationary']), 'final': run['records'][-1].mean(axis=0).tolist()} for name, run in runs.items()}, 'comparisons': comparisons}
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(runs, {name: baseline[name] for name in ['map_graph', 'mean_graph', 'short_periodic', 'enumerated_spectral']})
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
