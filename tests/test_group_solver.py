"""Test that the convex formulation matches the original discrete score."""

import itertools
import unittest
import numpy as np
from baselines.algorithms import cardinality_kernel, solve_subset


class GroupSolverTests(unittest.TestCase):
    def test_reformulation_preserves_all_subsets_and_is_psd(self):
        rng = np.random.default_rng(9)
        features = rng.normal(size=(7, 4))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        for pair in [1 - features @ features.T, rng.normal(size=(7, 7))]:
            kernel, shift = cardinality_kernel(pair)
            self.assertGreaterEqual(np.linalg.eigvalsh(kernel)[0], -1e-9)
            for k in range(1, 8):
                for selected in itertools.combinations(range(7), k):
                    x = np.zeros(7)
                    x[list(selected)] = 1
                    original = sum(pair[i, j] for i, j in itertools.combinations(selected, 2))
                    revised = 0.5 * (k * k + shift * k - x @ kernel @ x)
                    self.assertAlmostEqual(original, revised, places=10)

    def test_optima_match_exhaustive_for_multiple_cardinalities_and_weights(self):
        rng = np.random.default_rng(32)
        u = rng.normal(size=8)
        pair = rng.normal(size=(8, 8))
        for k in [1, 3, 7, 8]:
            for weight in [0.0, 0.7]:
                value = lambda subset: sum(u[i] for i in subset) + weight * sum(
                    pair[i, j] for i, j in itertools.combinations(subset, 2)
                )
                ids, report = solve_subset(u, pair, k, weight, 10)
                expected = max(value(v) for v in itertools.combinations(range(8), k))
                self.assertAlmostEqual(value(ids), expected, places=7)
                self.assertAlmostEqual(report["objective"], expected, places=7)
                self.assertTrue(report["optimal"])
                self.assertEqual(report["formulation"], "cardinality-psd-v1")

    def test_invalid_solver_inputs_are_rejected(self):
        for count, seconds in [(True, 1), (0, 1), (4, 1), (1, 0), (1, float("nan"))]:
            with self.assertRaises(ValueError):
                solve_subset(np.ones(3), np.zeros((3, 3)), count, 1, seconds)


if __name__ == "__main__":
    unittest.main()
