"""Independent numerical checks of normalization, cumulants, and inversion.

Run with: python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pta_distribution as pta


def compensated_exp(z):
    """e^-z - 1 + z, stable for small positive z."""
    if z < 1e-4:
        return z*z*(0.5-z/6+z*z/24-z**3/120)
    return np.expm1(-z)+z


class DistributionChecks(unittest.TestCase):
    def test_all_source_means_and_bright_tails_match(self):
        # r=u^-1/2 transforms int u rho du into a benign radial integral.
        for model in pta.model_names:
            with self.subTest(model=model):
                upper = 2*np.sqrt(np.pi) if model == "sphere" else np.inf
                mean = 2*pta.TAIL_AMPLITUDE*quad(
                    lambda r: pta.source_shape(r**-2, model), 0, upper,
                    epsabs=2e-11, epsrel=2e-11)[0]
                self.assertAlmostEqual(mean, 1.0, places=9)
                self.assertAlmostEqual(pta.source_shape(1e16, model), 1.0,
                                       delta=6e-9)

    def test_closed_exponents_match_direct_source_integrals(self):
        for model in pta.model_names:
            for q in (0.01, 0.3):
                with self.subTest(model=model, q=q):
                    upper = 2*np.sqrt(np.pi) if model == "sphere" else np.inf
                    def integrand(r):
                        return (2*pta.TAIL_AMPLITUDE*r*r
                                *pta.source_shape(r**-2, model)
                                *compensated_exp(q/(r*r)))
                    direct = quad(integrand, 0, upper, epsabs=3e-12,
                                  epsrel=3e-12, limit=250)[0]
                    self.assertAlmostEqual(pta.psi(q, model).real, direct,
                                           delta=2e-11)

    def test_characteristic_normalization_and_conjugacy(self):
        t = np.array([0.0, 0.001, 0.5, 4.0, 40.0])
        for model in pta.model_names:
            with self.subTest(model=model):
                phi = pta.characteristic(t, 10.0, model)
                self.assertEqual(phi[0], 1.0)
                np.testing.assert_allclose(pta.characteristic(-t, 10, model),
                                           phi.conjugate(), atol=1e-13)
                self.assertTrue(np.all(np.abs(phi) <= 1+2e-13))
                for order in (0, 1, 2, 3):
                    self.assertEqual(pta.asymptotic_characteristic(0, 10, model,
                                                                 order), 1.0)

    def test_paper_reference_densities(self):
        # These values were obtained independently by 60-digit exponent
        # evaluation and adaptive Fourier quadrature in the derivation.
        references = {-1.0: 0.23564771, 0.0: 0.52904555, 1.0: 0.08413935}
        for x, expected in references.items():
            self.assertAlmostEqual(pta.density_quad(x, 1000, "paper"), expected,
                                   delta=6e-9)

    def test_full_correction_reference_densities(self):
        # Independent earlier inversion checks the cross terms in the
        # expansion, not just agreement between two numerical integrators.
        references = {-1.0: 0.23742475, 0.0: 0.52931352, 1.0: 0.08413914}
        for x, expected in references.items():
            self.assertAlmostEqual(pta.density_quad(x, 1000, "paper", order=3),
                                   expected, delta=6e-9)

    def test_asymptotic_characteristic_approaches_exact(self):
        # The exact exponents come from the source integrals, independently
        # of the A/B/C symbols used by the characteristic expansion.
        for model in pta.model_names:
            with self.subTest(model=model):
                err1 = abs(pta.characteristic(1, 1000, model)
                           - pta.asymptotic_characteristic(1, 1000, model, 3))
                err2 = abs(pta.characteristic(1, 10000, model)
                           - pta.asymptotic_characteristic(1, 10000, model, 3))
                self.assertLess(err1, 4e-7)
                self.assertLess(err2, err1/10)

    def test_plot_grid_against_independent_quadrature(self):
        xs = np.array([-1.0, -0.5, 0.0, 1.0, 3.0])
        for model in pta.model_names:
            with self.subTest(model=model):
                curves = pta.density_curves(xs, 10, model, orders=(None, 3),
                                            tmax=160, dt=0.02)
                for order in (None, 3):
                    refs = [pta.density_quad(x, 10, model, order, tmax=200)
                            for x in xs]
                    np.testing.assert_allclose(curves[order], refs,
                                               atol=2e-7, rtol=0)

    def test_fourier_cutoff_and_step_convergence(self):
        xs = np.array([-1.4, 0.2, 2.2])
        for model in pta.model_names:
            with self.subTest(model=model):
                coarse = pta.density_grid(xs, 10, model, tmax=160, dt=0.02)
                fine = pta.density_grid(xs, 10, model, tmax=240, dt=0.01)
                np.testing.assert_allclose(coarse, fine, atol=2e-7, rtol=0)

    def test_high_N_error_curve_resolves_smallest_correction(self):
        # The convergence plot reaches errors of a few 1e-9.  Exact and
        # expanded densities share the leading quadrature bias, so check
        # their difference against separate adaptive full-CF integrations.
        x, N = -1.05, 1e5
        curves = pta.density_curves([x], N, "sphere", orders=(None, 3),
                                    tmax=160, dt=0.02)
        grid_error = curves[3][0]-curves[None][0]
        reference = (pta.density_quad(x, N, "sphere", 3, tmax=240, epsabs=2e-12)
                     - pta.density_quad(x, N, "sphere", None,
                                        tmax=240, epsabs=2e-12))
        self.assertGreater(abs(reference), 3e-9)
        self.assertAlmostEqual(grid_error, reference, delta=5e-13)

    def test_stable_airy_matches_its_fourier_transform(self):
        for x in (-1.0, 0.0, 1.0, 4.0):
            self.assertAlmostEqual(pta.stable_density(x),
                                   pta.density_quad(x, 100, order=0),
                                   delta=2e-10)

    def test_zero_source_atom(self):
        self.assertEqual(pta.zero_source_probability(10, "paper"), 0)
        self.assertLess(pta.zero_source_probability(10, "sphere"), 6.5e-19)
        self.assertAlmostEqual(pta.zero_source_probability(1, "gaussian"),
                               np.exp(-8), places=14)


if __name__ == "__main__":
    unittest.main()
