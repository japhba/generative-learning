"""Numerical checks of gradients, covariance controls, and metric geometry."""

import unittest
import numpy as np
from numpy.polynomial.hermite import hermgauss
from experiments.simulate_finite_sample import gradients, initial_fisher, initial_weight, metric_hessians, metrics


def finite_hessian(function, w, epsilon=1e-4):
    eye = np.eye(len(w)) * epsilon
    return np.array([[(function(w + u + v) - function(w + u - v) - function(w - u + v) + function(w - u - v)) / (4 * epsilon**2)
                      for v in eye] for u in eye])


class FiniteSampleTests(unittest.TestCase):
    def test_score_and_bernoulli_moments(self):
        w, x = np.array([.3, -.7]), np.array([1.2, .4])
        mean = np.tanh(w @ x)
        p = (1 + mean) / 2
        scores = np.stack([x * (1 - mean), x * (-1 - mean)])
        np.testing.assert_allclose(np.array([p, 1-p]) @ scores, 0, atol=1e-15)
        covariance = np.einsum("n,ni,nj->ij", [p, 1-p], scores, scores)
        np.testing.assert_allclose(covariance, (1 - mean**2) * np.outer(x, x), atol=1e-15)
        eps = 1e-6
        likelihood = lambda theta: theta @ x - np.logaddexp(theta @ x, -theta @ x)
        numerical = np.array([(likelihood(w + e) - likelihood(w - e)) / (2*eps) for e in eps * np.eye(2)])
        np.testing.assert_allclose(numerical, scores[0], atol=1e-9)

    def test_metric_hessians(self):
        w = np.array([.4, .7, -.3])
        for name, analytic in zip(("overlap", "error"), metric_hessians(w)):
            numerical = finite_hessian(lambda theta: metrics(theta[None])[name][0], w)
            np.testing.assert_allclose(numerical, analytic, rtol=2e-6, atol=2e-7)

    def test_quadrature_convergence(self):
        for overlap in (.3, .8):
            w = initial_weight(8, overlap)
            np.testing.assert_allclose(initial_fisher(w, 80), initial_fisher(w, 120), atol=2e-8, rtol=2e-7)

    def test_covariance_controls(self):
        rng = np.random.default_rng(9173)
        repetitions = 150000
        w0 = np.array([.3, -.4])
        x0 = np.array([[1.1, .2], [-.3, 1.2], [.5, -.7]])
        w = np.broadcast_to(w0, (3, repetitions, 2))
        x = np.broadcast_to(x0, (repetitions, 3, 2))
        draws, trace = gradients(w, x, rng)
        v = 1 - np.tanh(x0 @ w0)**2
        expected = np.einsum("b,bi,bj->ij", v, x0, x0) / 9
        for arm in range(3):
            target = expected if arm < 2 else np.eye(2) * np.trace(expected) / 2
            np.testing.assert_allclose(np.cov(draws[arm].T), target, rtol=.03, atol=.002)
            self.assertLess(np.linalg.norm(draws[arm].mean(axis=0)), .005)
            np.testing.assert_allclose(trace[arm], np.trace(expected), atol=1e-15)

    def test_scalar_exact_one_step(self):
        z, mass = hermgauss(32)
        mass = mass / np.sqrt(np.pi)
        for initial_mean in (.1, 1.):
            theta, eta = np.sqrt(initial_mean), .1
            new_theta = theta + eta * 2 * theta * np.sqrt(2) * z
            u = 4 * eta**2
            expected_error = initial_mean**2 * (1+6*u+3*u*u) - 2*initial_mean*(1+u) + 1
            np.testing.assert_allclose(mass @ new_theta, theta, atol=1e-15)
            np.testing.assert_allclose(mass @ ((new_theta**2 - 1)**2), expected_error, atol=1e-15)


if __name__ == "__main__": unittest.main()
