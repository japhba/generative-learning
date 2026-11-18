"""Reproduce the finite-budget tables in tex/practice_residual_depth.tex by summation.

Each three-level state is (early execution, middle execution, terminal probabilities
for operations 0, 1, 2). The teacher executes both binary moves and prefers terminal
operation 0. All sample counts include rejected strings and incomplete parses.
"""

from functools import cache
from itertools import product
from math import log

import numpy as np


def binary_curve(total, watch, step):
    """Distribution of the number of updates to the second binary gate."""
    counts = np.ones(1)
    for t in range(total + 1):
        first = 1 - 0.5 * (1 - step) ** t
        second = 1 - 0.5 * (1 - step) ** np.arange(t + 1)
        loss = -log(first) - counts @ np.log(second)
        root_accuracy = (1 + 5 * first * (counts @ second)) / 6
        yield loss, root_accuracy
        reach = first if t < watch else 1 - first * (1 - first)
        counts = np.convolve(counts, [1 - reach, reach])


def outcome_probabilities(reader, source):
    """Stop at gate one, stop at gate two, or judge one of three terminal moves."""
    reach_middle = 1 - source[..., 0] * (1 - reader[..., 0])
    reach_top = reach_middle * (1 - source[..., 1] * (1 - reader[..., 1]))
    terminal = reach_top[..., None] * source[..., 2:]
    return np.concatenate(((1 - reach_middle)[..., None], (reach_middle - reach_top)[..., None], terminal), axis=-1)


def successors(states, step):
    """Index the five possible successor states of each recipient, independent of source."""
    indices = {}
    destinations = []
    for early, middle, *terminal in states:
        first = early + step * (1 - early)
        second = middle + step * (1 - middle)
        outcomes = [(first, middle, *terminal), (first, second, *terminal)]
        row = np.array(terminal)
        for mark in range(3):
            if mark == 0:
                activity = np.array([1.0, 0.0, 0.0])
            else:
                activity = row / (1 - row[mark])
                activity[mark] = 0
            updated = (1 - step) * row + step * activity
            outcomes.append((first, second, *updated))
        destinations.append([indices.setdefault(state, len(indices)) for state in outcomes])
    return np.array(list(indices)), np.array(destinations)


def population_loss(states):
    return -np.log(states[:, :3]).sum(axis=1)


def marginal_curves(initial, step, budget):
    """Own, exactly KL-matched depth swap, and fresh wake share a state grid."""
    states = np.array([initial])
    masses = [np.ones(1) for _ in range(3)]
    teacher = np.array([1, 1, 1, 0, 0])
    yield [mass @ population_loss(states) for mass in masses]
    for _ in range(budget):
        following, destinations = successors(states, step)
        sources = (states, states[:, [1, 0, 2, 3, 4]], teacher)
        for i, source in enumerate(sources):
            transitions = masses[i][:, None] * outcome_probabilities(states, source)
            masses[i] = np.bincount(destinations.ravel(), weights=transitions.ravel(), minlength=len(following))
        states = following
        yield [mass @ population_loss(states) for mass in masses]


def live_twin_curve(first, second, step, budget):
    """Sum the joint law when each learner reads the other learner's next string."""
    left, right = np.array([first]), np.array([second])
    joint = np.ones((1, 1))
    yield population_loss(left)[0], population_loss(right)[0]
    for _ in range(budget):
        next_left, left_index = successors(left, step)
        next_right, right_index = successors(right, step)
        left_events = outcome_probabilities(left[:, None, :], right[None, :, :])
        right_events = outcome_probabilities(right[None, :, :], left[:, None, :])
        following = np.zeros((len(next_left), len(next_right)))
        for i, j in product(range(5), repeat=2):
            destinations = left_index[:, i, None], right_index[None, :, j]
            np.add.at(following, destinations, joint * left_events[..., i] * right_events[..., j])
        joint, left, right = following, next_left, next_right
        np.testing.assert_allclose(joint.sum(), 1, rtol=1e-12)
        yield population_loss(left) @ joint.sum(axis=1), population_loss(right) @ joint.sum(axis=0)


def best_wake_sleep(initial, step, horizon):
    """Every sleep receives ideal teacher-direction supervision on both reader gates."""
    def trained(p, hits):
        return 1 - (1 - p) * (1 - step) ** hits

    @cache
    def value(t, wakes, middle_hits, terminal_hits):
        if t == horizon:
            return -sum(log(trained(p, n)) for p, n in zip(initial[:3], (wakes, middle_hits, terminal_hits)))
        r1 = trained(initial[0], t)
        r2 = trained(initial[1], middle_hits + t - wakes)
        sleep = value(t + 1, wakes, middle_hits, terminal_hits)
        wake = ((1 - r1) * value(t + 1, wakes + 1, middle_hits, terminal_hits)
                + r1 * (1 - r2) * value(t + 1, wakes + 1, middle_hits + 1, terminal_hits)
                + r1 * r2 * value(t + 1, wakes + 1, middle_hits + 1, terminal_hits + 1))
        return min(wake, sleep)

    return value(0, 0, 0, 0)


if __name__ == "__main__":
    own, fresh = (list(binary_curve(100, n, 0.01)) for n in (50, 100))
    print("strings  own_KL  fresh_KL  own_root_accuracy  fresh_root_accuracy")
    for t in (0, 50, 75, 100):
        print(t, *(f"{x:.5f}" for x in (own[t][0], fresh[t][0], own[t][1], fresh[t][1])))

    profiles = [(1/3, 5/6, 1/3, 1/3, 1/3), (5/6, 1/3, 1/3, 1/3, 1/3)]
    live = list(live_twin_curve(*profiles, step=0.1, budget=6))
    print("early  middle  strings  own_KL  live_twin_KL  matched_KL  best_wake_sleep_KL")
    for i, profile in enumerate(profiles):
        curves = list(marginal_curves(profile, 0.1, 6))
        for t in (0, 1, 5, 6):
            own_loss, matched_loss, wake_loss = curves[t]
            baseline = best_wake_sleep(profile, 0.1, t)
            np.testing.assert_allclose(baseline, wake_loss, rtol=1e-12)
            print(*profile[:2], t, *(f"{x:.5f}" for x in (own_loss, live[t][i], matched_loss, baseline)))
