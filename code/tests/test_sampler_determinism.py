"""Seeded determinism: same seed produces same output."""
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sampler import samplerz


SEED = 42
N = 100


class TestSamplerDeterminism:

    def _generate(self, mu, sigma, seed, n):
        random.seed(seed)
        return [samplerz(mu, sigma) for _ in range(n)]

    def test_centered(self):
        a = self._generate(0.0, 1.55, SEED, N)
        b = self._generate(0.0, 1.55, SEED, N)
        assert a == b

    def test_fractional_center(self):
        a = self._generate(0.5, 1.55, SEED, N)
        b = self._generate(0.5, 1.55, SEED, N)
        assert a == b

    def test_different_seeds_differ(self):
        a = self._generate(0.0, 1.55, SEED, N)
        b = self._generate(0.0, 1.55, SEED + 1, N)
        assert a != b
