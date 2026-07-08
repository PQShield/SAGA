"""Test SAGA multivariate analysis against generated test vectors."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from saga import MultivariateSamples


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


class TestBadMultivariate:
    """Flawed multivariate distributions should be detected."""

    def test_at_least_one_fails(self, bad_multivariate_vector):
        v = bad_multivariate_vector
        mv = MultivariateSamples(v["params"]["sigma"], v["samples"])

        dh_fails = mv.PO <= 0.001
        dc_fails = mv.dc_pvalue <= 0.001
        coord_fail_rate = 1 - (mv.nb_gaussian_coord / mv.dim)

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
