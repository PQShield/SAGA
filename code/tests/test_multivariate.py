"""Test SAGA multivariate analysis against generated test vectors."""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from saga import MultivariateSamples


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
