"""Stable population scaling calculations for the existing conditional RBM."""

from math import erf, pi, sqrt

import numpy as np
from numpy.polynomial.legendre import leggauss

from bestn_theory import sigmoid


NODES, MASS = leggauss(256)
V, MASS = 16 * (NODES + 1), 16 * MASS
SECH2 = 4 * sigmoid(2 * V) * sigmoid(-2 * V)


def phi(x): return np.exp(-np.asarray(x)**2 / 2) / sqrt(2 * pi)


def normal_cdf(x): return (1 + erf(x / sqrt(2))) / 2


def rbm_metrics(norm, rho, signal=1.5):
    """Integrate near the decision boundary; fixed Hermite rules alias at large norm."""
    delta, z = signal * rho, V / norm
    positive, negative = phi(z - delta), phi(z + delta)
    difference = negative * np.expm1(2 * z * delta)
    soft_excess = np.dot(MASS, sigmoid(-2 * V) * difference) / norm
    radial = .5 * np.dot(MASS, V * SECH2 * difference) / norm**2
    transverse = .5 * np.dot(MASS, SECH2 * (positive + negative)) / norm
    overlap_velocity = signal * (1 - rho**2) * transverse / norm
    gradient_squared = radial**2 + signal**2 * (1 - rho**2) * transverse**2
    error = normal_cdf(-delta) + soft_excess
    return {"norm": float(norm), "rho": float(rho), "error": float(error),
            "excess_error": float(error - normal_cdf(-signal)), "soft_excess": float(soft_excess),
            "radial_velocity": float(radial), "overlap_velocity": float(overlap_velocity),
            "gradient_squared": float(gradient_squared)}


def rbm_population(times, initial=(1., .3), signal=1.5):
    state, time, rows = np.array(initial, dtype=float), 0., []
    def velocity(value):
        row = rbm_metrics(*value, signal)
        return np.array([row["radial_velocity"], row["overlap_velocity"]])
    for endpoint in times:
        while time < endpoint:
            dt = min(endpoint - time, max(.02, .01 * (time + 1)))
            k1 = velocity(state)
            k2 = velocity(state + .5 * dt * k1)
            k3 = velocity(state + .5 * dt * k2)
            k4 = velocity(state + dt * k3)
            state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            time += dt
        rows.append({"time": float(endpoint), **rbm_metrics(*state, signal)})
    return rows


def gate_time(probability, initial=.6):
    """Integral of dp/[p^2(1-p)^2] for a unit-logit success gate."""
    def primitive(p): return -1 / p + 1 / (1 - p) + 2 * (np.log(p) - np.log1p(-p))
    return primitive(np.asarray(probability)) - primitive(initial)


def asymptotic_constant(signal=1.5): return signal * float(phi(signal)) * pi**2 / 24
