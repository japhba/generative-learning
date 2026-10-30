"""Independent finite-sum and finite-difference checks for the routing candidate."""

from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

import practice_loop_candidate as candidate


root = Path('/var/tmp/jbauer/generative_practice_loop_candidate/checks')
root.mkdir(parents=True, exist_ok=True)
archive = np.load('/var/tmp/jbauer/generative_practice_loop_candidate/confirmation/curves.npz')
transition = archive['learned_0']
states = len(transition)
queries = candidate.all_queries(states)
horizon = 3
weights = candidate.init_weights(np.random.default_rng(631), states, 4)
results = {}

for tied in [False, True]:
    parameters = candidate.copy_weights(weights)
    if tied: parameters[0] = -parameters[1].copy()
    gradients = candidate.planning_gradient(parameters, transition, queries, horizon)
    maximum, maximum_derivative = 0.0, 0.0
    for group in range(1 if tied else 0, 5):
        for index in np.ndindex(parameters[group].shape):
            original = parameters[group][index]
            values = []
            for direction in [1, -1]:
                parameters[group][index] = original + direction * 1e-5
                if tied and group == 1: parameters[0][index] = -parameters[1][index]
                values.append(candidate.policy_values(parameters, transition, horizon)[1][-1].mean())
            parameters[group][index] = original
            if tied and group == 1: parameters[0][index] = -original
            numerical = (values[0] - values[1]) / 2e-5
            analytical = gradients[group][index] - gradients[0][index] if tied and group == 1 else gradients[group][index]
            maximum = max(maximum, abs(numerical - analytical))
            maximum_derivative = max(maximum_derivative, abs(numerical))
    assert maximum < 1e-8 and maximum_derivative > 1e-5
    results['tied' if tied else 'untied'] = {'gradient_max_error': maximum, 'maximum_derivative': maximum_derivative}

policy, probabilities, ph, pa, hidden = candidate.marginal(weights, queries)
enumerated = np.zeros((states, states))
for start, goal in queries:
    for sequence in itertools.product(range(2), repeat=horizon):
        probability, state = 1.0, int(start)
        for action in sequence:
            probability *= policy[state * states + goal, action]
            if state != goal: state = transition[state, action]
        enumerated[start, goal] += probability * (state == goal)
value_error = float(np.max(np.abs(enumerated - candidate.policy_values(weights, transition, horizon)[1][-1])))
assert value_error < 1e-14
results['value_enumeration_error'] = value_error

coefficient = np.random.default_rng(95).normal(size=(len(queries), 2))
local = candidate.zeros_like(weights)
expected_score = np.zeros((len(queries), 4 + 2))
for n, (state, goal) in enumerate(queries):
    for h_index, h in enumerate(hidden):
        for action in range(2):
            joint_probability = ph[n, h_index] * pa[h_index, action]
            hidden_score = h - probabilities[n]
            action_score = np.eye(2)[action] - pa[h_index]
            weighted = joint_probability * coefficient[n, action]
            local[0][:, state] += weighted * hidden_score
            local[1][:, goal] += weighted * hidden_score
            local[2] += weighted * hidden_score
            local[3] += weighted * np.outer(action_score, h)
            local[4] += weighted * action_score
            expected_score[n] += joint_probability * np.r_[hidden_score, action_score]
marginal = candidate.probability_gradient(weights, queries, coefficient)
local_error = float(max(np.max(np.abs(x - y)) for x, y in zip(local, marginal)))
zero_score_error = float(np.max(np.abs(expected_score)))
assert local_error < 1e-12 and zero_score_error < 1e-14
results['local_joint_marginal_identity_error'] = local_error
results['conditional_expected_score_error'] = zero_score_error
results['note'] = 'Conditional expected score zero proves the independent same-query rollout return is a valid baseline. The original tied-pilot three-query gradient check had zero objective and was vacuous; these checks use all queries and assert nonzero derivatives.'
results['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
results['shared_code_sha256'] = hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest()
(root / 'checks.json').write_text(json.dumps(results, indent=2) + '\n')
(root / 'run_source.py').write_bytes(Path(__file__).read_bytes())
(root / 'shared_source.py').write_bytes(Path(candidate.__file__).read_bytes())
print(json.dumps(results, indent=2))
