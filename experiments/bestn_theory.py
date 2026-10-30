"""Population dynamics for likelihood distillation of binary best-of-N outputs."""

import numpy as np
from numpy.polynomial.hermite import hermgauss


NODES, MASS = hermgauss(160)
NODES, MASS = np.sqrt(2) * NODES, MASS / np.sqrt(np.pi)


def sigmoid(x): return np.exp(-np.logaddexp(0, -x))


def multiplier(p, n): return sum((1 - p)**k for k in range(n - 1))


def expectation(w, signal, function):
    values = signal * w[0] + np.linalg.norm(w) * NODES
    value, derivative = function(values)
    result = w * np.dot(MASS, derivative)
    result[0] += signal * np.dot(MASS, value)
    return result


def selected_mean(s, n):
    u = sigmoid(-2 * s)
    return 1 - 2 * u**n, 4 * n * u**n * (1 - u)


def current_residual(s, n):
    u = sigmoid(-2 * s)
    return 2 * (u - u**n), -4 * u * (1 - u) * (1 - n * u**(n - 1))


def ordinary_mean(s):
    value = np.tanh(s)
    return value, 1 - value**2


def drift(w, initial, signal, n, source):
    if source == "current": return expectation(w, signal, lambda s: current_residual(s, n))
    assert source == "frozen"
    return expectation(initial, signal, lambda s: selected_mean(s, n)) - expectation(w, signal, ordinary_mean)


def success(w, signal): return float(np.dot(MASS, sigmoid(2 * (signal * w[0] + np.linalg.norm(w) * NODES))))


def overlap_velocity(w, signal, n):
    norm = np.linalg.norm(w)
    rho = w[0] / norm
    a = np.dot(MASS, current_residual(signal * w[0] + norm * NODES, n)[0])
    return signal / norm * (1 - rho**2) * a
