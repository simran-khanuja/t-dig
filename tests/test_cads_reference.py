"""Independent NumPy oracle for CADS equations 1--4 (arXiv:2310.17347v3).

The oracle imports no production math. Torch supplies only the fixed noise; NumPy float64
computes the perturbation and population statistics.
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from baselines.algorithms import corrupt, gamma


def reference(y, noise, a, scale, mix):
    y, noise = np.asarray(y, dtype=np.float64), np.asarray(noise, dtype=np.float64)
    z = np.sqrt(a) * y + scale * np.sqrt(1 - a) * noise
    axes = tuple(range(1, y.ndim))
    if mix:
        standardized = (z - z.mean(axis=axes, keepdims=True)) / z.std(
            axis=axes, keepdims=True, ddof=0
        )
        rescaled = standardized * y.std(axis=axes, keepdims=True, ddof=0)
        rescaled += y.mean(axis=axes, keepdims=True)
        z = mix * rescaled + (1 - mix) * z
    return z


class CadsReferenceTests(unittest.TestCase):
    def test_schedule_against_piecewise_definition(self):
        for lower, upper in [(0.6, 0.9), (0.1, 0.3), (0, 1)]:
            for t in list(np.linspace(0, 1, 101)) + [lower, upper]:
                expected = 1 if t <= lower else 0 if t >= upper else (upper - t) / (upper - lower)
                self.assertAlmostEqual(gamma(float(t), lower, upper), expected, places=14)

    def test_perturbation_against_independent_numpy(self):
        rng = np.random.default_rng(1729)
        for shape in [(1, 7, 9), (2, 7, 9), (1, 13)]:
            y = torch.tensor(rng.normal(size=shape), dtype=torch.float32)
            noise = torch.tensor(rng.normal(size=shape), dtype=torch.float32)
            # Distinct row statistics expose accidental cross-batch rescaling.
            if shape[0] == 2:
                y[1] = y[1] * 3 + 7
            for a in [0, 0.17, 0.5, 0.93]:
                for scale in [0.1, 0.25, 1.0]:
                    for mix in [0, 0.3, 1]:
                        with self.subTest(shape=shape, a=a, scale=scale, mix=mix):
                            expected = reference(y.numpy(), noise.numpy(), a, scale, mix)
                            before = y.clone()
                            with patch('torch.randn', return_value=noise.clone()) as draw:
                                actual = corrupt(y, a, scale, mix, torch.Generator())
                            draw.assert_called_once()
                            np.testing.assert_allclose(
                                actual.numpy(), expected, rtol=3e-6, atol=3e-6
                            )
                            self.assertTrue(torch.equal(before, y))

    def test_identity_does_not_draw_noise(self):
        y = torch.arange(24).reshape(2, 3, 4).float()
        with patch('torch.randn', side_effect=AssertionError('identity consumed noise')):
            self.assertIs(corrupt(y, 1, 0.25, 1, torch.Generator()), y)

    def test_zero_noise_scale_is_not_disabled_cads(self):
        y = torch.arange(24).reshape(2, 3, 4).float()
        actual = corrupt(y, 0.25, 0, 0, torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(actual, y * 0.5))
        self.assertFalse(torch.equal(actual, y))


if __name__ == '__main__':
    unittest.main()
