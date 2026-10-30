import unittest

import numpy as np
from numpy.polynomial.hermite import hermgauss

from bestn_theory import current_residual, sigmoid
from scaling_theory import asymptotic_constant, gate_time, normal_cdf, rbm_metrics


class ScalingTheoryTest(unittest.TestCase):
    def test_boundary_quadrature_against_independent_hermite(self):
        nodes, mass = hermgauss(320)
        nodes, mass = np.sqrt(2)*nodes, mass/np.sqrt(np.pi)
        for norm, rho in ((1., .3), (1., .8), (2., .5)):
            w = norm * np.array([rho, np.sqrt(1-rho**2)])
            row = rbm_metrics(norm, rho)
            values = 1.5*w[0]+norm*nodes
            value, derivative = current_residual(values, 2)
            gradient = w*np.dot(mass, derivative)
            gradient[0] += 1.5*np.dot(mass, value)
            self.assertAlmostEqual(row["error"], 1-np.dot(mass, sigmoid(2*values)), places=8)
            self.assertAlmostEqual(row["radial_velocity"], np.dot(w/norm, gradient), places=7)
            self.assertAlmostEqual(row["gradient_squared"], np.dot(gradient, gradient), places=7)

    def test_asymptotic_coefficients(self):
        a = asymptotic_constant()
        row = rbm_metrics(1000., 1.)
        self.assertAlmostEqual(row["soft_excess"] * 1000**2 / a, 1., places=5)
        self.assertAlmostEqual(row["radial_velocity"] * 1000**3 / (2*a), 1., places=5)

    def test_success_gradient_identity(self):
        for norm, rho in ((1., .3), (5., .8), (100., 1.)):
            row = rbm_metrics(norm, rho)
            h = 1e-4
            updated = rbm_metrics(norm+h*row["radial_velocity"], rho+h*row["overlap_velocity"])
            measured = (row["error"] - updated["error"]) / h
            np.testing.assert_allclose(measured, row["gradient_squared"], rtol=2e-4, atol=1e-12)

    def test_gate_time_derivative(self):
        p, h = .8, 1e-6
        derivative = (gate_time(p+h)-gate_time(p-h))/(2*h)
        self.assertAlmostEqual(derivative*p**2*(1-p)**2, 1., places=8)


if __name__ == "__main__": unittest.main()
