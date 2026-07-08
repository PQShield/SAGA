"""Tests for the analytic Renyi divergence certification layer."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

mpmath = pytest.importorskip("mpmath")

from certification import (
    certify_base_sampler,
    certify_full_sampler,
    compute_base_sampler_distribution,
    compute_ideal_halfgaussian,
    renyi_divergence,
    total_variation,
)


class TestBaseSampler:
    """Verify base sampler divergence is small (informational)."""

    def test_base_rd_is_small(self):
        result = certify_base_sampler()
        rd = result["renyi_divergences"]
        assert rd["R_2"] < 1e-20
        assert rd["R_inf"] < 1e-15

    def test_base_tv_is_small(self):
        result = certify_base_sampler()
        assert result["total_variation"] < 1e-15

    def test_base_distributions_sum_to_one(self):
        _, P = compute_base_sampler_distribution()
        _, Q = compute_ideal_halfgaussian()
        assert abs(sum(P) - 1) < 1e-30
        assert abs(sum(Q) - 1) < 1e-30


class TestFullSampler:
    """Verify full sampler meets security targets."""

    PARAMS = [
        (0.0, 1.55),
        (0.5, 1.55),
        (0.25, 1.55),
        (0.0, 1.3),
        (0.0, 1.7),
        (3.7, 1.55),
    ]

    @pytest.mark.parametrize("mu,sigma", PARAMS,
                             ids=[f"mu={m}_sig={s}" for m, s in PARAMS])
    def test_falcon_512_passes(self, mu, sigma):
        result = certify_full_sampler(sigma=sigma, mu=mu)
        check = result["security_check"]["falcon-512"]
        assert check["passes"], (
            f"Full sampler at mu={mu}, sigma={sigma} fails Falcon-512 "
            f"certification: R_128-1 = {check['R_a_minus_1']:.2e}"
        )

    @pytest.mark.parametrize("mu,sigma", PARAMS,
                             ids=[f"mu={m}_sig={s}" for m, s in PARAMS])
    def test_falcon_1024_passes(self, mu, sigma):
        result = certify_full_sampler(sigma=sigma, mu=mu)
        check = result["security_check"]["falcon-1024"]
        assert check["passes"], (
            f"Full sampler at mu={mu}, sigma={sigma} fails Falcon-1024 "
            f"certification: R_256-1 = {check['R_a_minus_1']:.2e}"
        )


class TestRenyiDivergence:
    """Unit tests for the divergence computation."""

    def test_identical_distributions(self):
        P = [mpmath.mpf("0.25")] * 4
        Q = [mpmath.mpf("0.25")] * 4
        assert abs(renyi_divergence(P, Q, 2)) < 1e-50

    def test_known_value(self):
        P = [mpmath.mpf("0.5"), mpmath.mpf("0.5")]
        Q = [mpmath.mpf("0.4"), mpmath.mpf("0.6")]
        r2 = renyi_divergence(P, Q, 2)
        expected = mpmath.log(
            mpmath.mpf("0.5") ** 2 / mpmath.mpf("0.4")
            + mpmath.mpf("0.5") ** 2 / mpmath.mpf("0.6")
        )
        assert abs(r2 - expected) < 1e-50

    def test_r_inf(self):
        P = [mpmath.mpf("0.5"), mpmath.mpf("0.5")]
        Q = [mpmath.mpf("0.3"), mpmath.mpf("0.7")]
        r_inf = renyi_divergence(P, Q, 'inf')
        expected = mpmath.log(mpmath.mpf("0.5") / mpmath.mpf("0.3"))
        assert abs(r_inf - expected) < 1e-50

    def test_tv_identical(self):
        P = [mpmath.mpf("0.5"), mpmath.mpf("0.5")]
        assert total_variation(P, P) == 0
