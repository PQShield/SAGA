"""Test SAGA univariate analysis against generated test vectors."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from saga import UnivariateSamples


class TestGoodVectors:
    """Perfect discrete Gaussian samples must pass all tests."""

    def test_is_valid(self, good_univariate_vector):
        v = good_univariate_vector
        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])
        assert uv.is_valid, (
            f"Good vector {v['label']} flagged as invalid "
            f"(chi2_p={uv.chi2_pvalue:.6f}, outliers={uv.outlier})"
        )

    def test_no_outliers(self, good_univariate_vector):
        v = good_univariate_vector
        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])
        assert uv.outlier == 0

    def test_extended_battery_passes(self, good_univariate_vector):
        v = good_univariate_vector
        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])
        ext = uv.run_extended_battery(samples=v["samples"], mc_B=200)
        assert ext["all_pass"], (
            f"Good vector {v['label']} fails extended battery: "
            + ", ".join(k for k, r in ext.items()
                        if isinstance(r, dict) and not r.get("passes", True))
        )


class TestBadVectors:
    """Flawed distributions must be detected by chi-square or extended battery."""

    def test_detected(self, bad_univariate_vector):
        v = bad_univariate_vector
        flaw = v["flaw"]["type"]

        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])

        # First check chi-square
        if not uv.is_valid:
            return  # detected by chi-square, good

        # Chi-square missed it — run extended battery (includes sequence tests)
        ext = uv.run_extended_battery(samples=v["samples"], mc_B=200)
        assert not ext["all_pass"], (
            f"Bad vector {v['label']} (flaw={flaw}) not detected by "
            f"chi-square or extended battery"
        )


class TestMediocreVectors:
    """Borderline vectors — document sensitivity without hard pass/fail."""

    def test_runs(self, mediocre_univariate_vector):
        v = mediocre_univariate_vector
        uv = UnivariateSamples(v["params"]["mu"], v["params"]["sigma"], v["samples"])
        assert uv.chi2_pvalue is not None
