"""Tests for suite calibration: p-value uniformity meta-test."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calibration import pvalue_uniformity


class TestPvalueUniformity:
    """Chi-square p-values under H0 should be U[0,1]."""

    def test_chi2_pvalues_uniform(self):
        result = pvalue_uniformity(mu=0.0, sigma=1.55,
                                   n=10000, reps=50, seed=3000)
        assert result["passes"], (
            f"p-value uniformity failed: KS p={result['ks_pvalue']:.4f}, "
            f"mean_p={result['mean_pvalue']:.3f}"
        )

    def test_mean_pvalue_near_half(self):
        result = pvalue_uniformity(mu=0.0, sigma=1.55,
                                   n=10000, reps=50, seed=3000)
        assert 0.2 < result["mean_pvalue"] < 0.8, (
            f"Mean p-value {result['mean_pvalue']:.3f} far from 0.5"
        )
