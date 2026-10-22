"""Fast smoke tests for fluctuation-energy analysis."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('pipe_analysis', Path(__file__).parents[1] / 'scripts' / 'analyze.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Tests(unittest.TestCase):
    def test_fluctuation_energy_zero_for_laminar(self):
        uzp = np.zeros((4, 5, 6))
        ur = np.zeros_like(uzp)
        uth = np.zeros_like(uzp)
        k = a.fluctuation_energy_density(uzp, ur, uth)
        np.testing.assert_allclose(k, 0.0)

    def test_fluctuation_energy_formula(self):
        uzp = np.ones((2, 2, 2)) * 2.0
        ur = np.ones_like(uzp) * 1.0
        uth = np.ones_like(uzp) * 3.0
        k = a.fluctuation_energy_density(uzp, ur, uth)
        np.testing.assert_allclose(k, 0.5 * (4 + 1 + 9))

    def test_section_mean_energy(self):
        r = np.array([0.2, 0.6])
        k = np.zeros((2, 3, 4))
        k[0, :, :] = 2.0
        k[1, :, :] = 6.0
        w = a.radial_weights(r)
        self.assertAlmostEqual(a.section_mean_energy(k, r), float(np.dot(w, [2.0, 6.0])))


if __name__ == '__main__':
    unittest.main()
