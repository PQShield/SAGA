"""Test SAGA multivariate analysis against generated test vectors."""
import json
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from saga import MultivariateSamples
from multivariate_tests import squared_norm_test, henze_zirkler_test
from generate_test_vectors import multivariate_discrete_gaussian


class TestSquaredNormCalibration:
    """squared_norm must be calibrated on integer data, and keep its power.

    ||x||^2 lives on a lattice; comparing it to a continuous chi2(dim) made the
    test strongly anti-conservative at small sigma (mean p ~ 0.27 instead of
    0.5, 20% of p-values below 0.05 on *correct* data). It now runs a binned
    chi-square against the exact pmf of ||x||^2 -- an analytic null, so unlike
    a Monte-Carlo calibration it keeps the power to reject at alpha.
    """

    DIM, SIGMA, N, REPS = 32, 1.55, 1000, 30

    def _correct(self, seed):
        rng = np.random.default_rng(seed)
        return multivariate_discrete_gaussian(rng, self.DIM, self.SIGMA, self.N)

    def test_calibrated_on_correct_integer_data(self):
        p = np.array([squared_norm_test(self.SIGMA, self._correct(2000 + i))["pvalue"]
                      for i in range(self.REPS)])
        assert p.mean() > 0.35, (
            f"squared_norm is anti-conservative on correct integer data: "
            f"mean p = {p.mean():.3f} (expected ~0.5)")
        assert (p < 0.05).mean() <= 0.20, (
            f"{(p < 0.05).mean():.0%} of p-values below 0.05 on correct data "
            f"(expected ~5%)")

    def test_power_preserved_on_integer_data(self):
        """A wrong sigma must still be rejected -- an MC null would floor at
        1/(B+1) and silently lose this."""
        rng = np.random.default_rng(2100)
        bad = multivariate_discrete_gaussian(rng, self.DIM, self.SIGMA * 1.05, self.N)
        res = squared_norm_test(self.SIGMA, bad)
        assert not res["passes"] and res["pvalue"] < 1e-3, res["pvalue"]

    def test_null_routing(self):
        """Integer data uses the exact discrete null; continuous data the chi2."""
        assert squared_norm_test(self.SIGMA, self._correct(2200))["null"] \
            == "exact-discrete-chi2"
        cont = np.random.default_rng(2300).normal(0, self.SIGMA,
                                                  size=(self.N, self.DIM))
        assert squared_norm_test(self.SIGMA, cont)["null"] == "chi2-analytic"


class TestHenzeZirklerPValue:
    def test_pvalue_floor_is_respected(self):
        """The MC p-value must be (1 + #{null >= obs}) / (B + 1).

        The naive mean() estimator returns exactly 0 whenever the observed
        statistic beats every null draw -- and 0 is below *any* alpha, so it
        rejected correct samplers about 1/(mc_B+1) of the time. Feed it data
        far from Gaussian so the observed statistic certainly beats all nulls;
        the p-value must then be the floor 1/(B+1), never 0.
        """
        MC_B = 20
        rng = np.random.default_rng(2400)
        far_from_gaussian = rng.uniform(-3, 3, size=(600, 16))
        res = henze_zirkler_test(1.55, far_from_gaussian, mc_B=MC_B)
        assert res["pvalue"] == pytest.approx(1.0 / (MC_B + 1)), res["pvalue"]


class TestNonIntegerFallback:
    """All real vectors are integer (Falcon signatures are integer vectors).
    Continuous input is not a supported case, but must degrade gracefully:
    skip the discrete per-coordinate channel, not crash or emit nan."""

    def test_continuous_skips_per_coordinate(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1.55, size=(1500, 32)).tolist()
        mv = MultivariateSamples(1.55, data)
        assert mv.is_integer is False
        assert mv.nb_gaussian_coord is None
        assert mv.univariates is None
        # covariance-based tests still run and produce finite p-values
        assert 0.0 <= mv.PO <= 1.0
        ext = mv.run_multivariate_battery()
        assert "fisher_bh" not in ext  # per-coordinate channel dropped


class TestGoodMultivariate:
    """iid Gaussian samples must pass all tests."""

    def test_doornik_hansen_passes(self, good_multivariate_vector):
        v = good_multivariate_vector
        mv = MultivariateSamples(v["params"]["sigma"], v["samples"])
        assert mv.PO > 0.001, f"DH p-value {mv.PO:.6f} too low for good vector"

    def test_diagcov_passes(self, good_multivariate_vector):
        v = good_multivariate_vector
        mv = MultivariateSamples(v["params"]["sigma"], v["samples"])
        assert mv.dc_pvalue > 0.001, f"diagcov p-value {mv.dc_pvalue:.6f} too low"

    def test_extended_battery_passes(self, good_multivariate_vector):
        v = good_multivariate_vector
        mv = MultivariateSamples(v["params"]["sigma"], v["samples"])
        ext = mv.run_multivariate_battery()
        assert ext["all_pass"], (
            f"Good multivariate vector {v['label']} fails extended battery: "
            + ", ".join(k for k, r in ext.items()
                        if isinstance(r, dict) and not r.get("passes", True))
        )

    def test_to_json_after_battery(self, good_multivariate_vector):
        """to_dict()/to_json() must survive the battery's numpy scalars."""
        v = good_multivariate_vector
        mv = MultivariateSamples(v["params"]["sigma"], v["samples"])
        mv.run_multivariate_battery()
        parsed = json.loads(mv.to_json())
        assert parsed["test"] == "multivariate"
        assert parsed["extended_tests"]["all_pass"] is True
        json.dumps(mv.to_dict())  # a user dumping to_dict() directly also works

    def test_per_coordinate_channel(self, good_multivariate_vector):
        """Integer (discrete-Gaussian) good vectors must light up the
        per-coordinate discrete channel; continuous vectors skip it."""
        v = good_multivariate_vector
        mv = MultivariateSamples(v["params"]["sigma"], v["samples"])
        if mv.is_integer:
            # essentially all coordinates should read as Gaussian
            assert mv.nb_gaussian_coord >= mv.dim - 2, (
                f"integer good vector {v['label']}: only "
                f"{mv.nb_gaussian_coord}/{mv.dim} coords Gaussian")
        else:
            assert mv.nb_gaussian_coord is None


class TestBadMultivariate:
    """Flawed multivariate distributions should be detected."""

    def test_at_least_one_fails(self, bad_multivariate_vector):
        v = bad_multivariate_vector
        mv = MultivariateSamples(v["params"]["sigma"], v["samples"])

        dh_fails = mv.PO <= 0.001
        dc_fails = mv.dc_pvalue <= 0.001
        # Per-coordinate channel only applies to integer data (None otherwise).
        if mv.nb_gaussian_coord is not None:
            coord_fail_rate = 1 - (mv.nb_gaussian_coord / mv.dim)
        else:
            coord_fail_rate = 0.0

        detected_basic = dh_fails or dc_fails or coord_fail_rate > 0.1

        if detected_basic:
            return  # caught by basic tests

        # Run extended battery
        ext = mv.run_multivariate_battery()
        assert not ext["all_pass"], (
            f"Bad multivariate vector {v['label']} not detected "
            f"(DH_p={mv.PO:.4f}, dc_p={mv.dc_pvalue:.4f}, "
            f"coord_fail_rate={coord_fail_rate:.2%}, "
            f"extended all_pass={ext['all_pass']})"
        )
