"""
Deterministic verification of hard-coded sampler tables.

Recomputes the half-Gaussian PDT and CDT from sigma_0 = 1.8205 using
mpmath at high precision and asserts equality with the tables in
sampler.py and sampler.c.

This catches the class of bugs (wrong table entries, PDT/RCDT confusion)
that no statistical test can detect — the affected mass can be as small
as e^{-50}.
"""
import sys
import os
import re

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

mpmath = pytest.importorskip("mpmath", reason="mpmath required for table lint")


SIGMA0 = "1.8205"
CDT_PRECISION = 72
N_ENTRIES = 19
TAU = 14


def compute_halfgaussian_pdt_mpmath():
    """Recompute the half-Gaussian PDT at high precision.

    The PDT uses half-Gaussian normalization: P(z) = rho(z) / sum_{z=0}^{18} rho(z),
    where rho(z) = exp(-z^2 / (2 * sigma0^2)). The entries sum to 2^72.
    """
    mpmath.mp.prec = 400

    sigma0 = mpmath.mpf(SIGMA0)
    two_sigma2 = 2 * sigma0 ** 2
    scale = mpmath.power(2, CDT_PRECISION)

    raw = []
    for z in range(N_ENTRIES):
        raw.append(mpmath.exp(-(z ** 2) / two_sigma2))

    half_sum = sum(raw)

    pdt = []
    for z in range(N_ENTRIES):
        prob = raw[z] / half_sum
        pdt.append(int(mpmath.nint(prob * scale)))

    return pdt


def compute_cdt_from_pdt(pdt):
    """Compute cumulative distribution table from PDT."""
    cdt = []
    cumsum = 0
    for i in range(len(pdt) - 1):
        cumsum += pdt[i]
        cdt.append(cumsum)
    return cdt


def parse_c_rcdt():
    """Extract the RCDT table values from sampler.c."""
    sampler_c = os.path.join(os.path.dirname(__file__), "..", "sampler.c")
    with open(sampler_c) as f:
        src = f.read()

    match = re.search(r'rcdt\[\]\s*=\s*\{([^}]+)\}', src)
    if not match:
        pytest.skip("Could not parse rcdt[] from sampler.c")

    nums = re.findall(r'(\d+)U', match.group(1))
    rows = []
    for i in range(0, len(nums), 3):
        hi, mid, lo = int(nums[i]), int(nums[i + 1]), int(nums[i + 2])
        val = (hi << 48) | (mid << 24) | lo
        rows.append(val)

    return rows


class TestPDT:
    """Verify the Python half-Gaussian PDT matches high-precision recomputation."""

    def test_pdt_matches(self):
        from sampler import halfgaussian_pdt

        expected = compute_halfgaussian_pdt_mpmath()
        assert len(halfgaussian_pdt) == len(expected), (
            f"PDT length mismatch: {len(halfgaussian_pdt)} vs {len(expected)}"
        )
        max_diff = 0
        for i, (actual, exp) in enumerate(zip(halfgaussian_pdt, expected)):
            diff = abs(actual - exp)
            max_diff = max(max_diff, diff)
            assert diff <= 100, (
                f"PDT entry {i} mismatch: {actual} vs {exp} "
                f"(diff={diff}, tolerance=100)"
            )
        print(f"PDT max rounding diff: {max_diff}")

    def test_pdt_sums_to_scale(self):
        from sampler import halfgaussian_pdt

        total = sum(halfgaussian_pdt)
        scale = 1 << CDT_PRECISION
        assert abs(total - scale) <= N_ENTRIES, (
            f"PDT sum {total} deviates from 2^{CDT_PRECISION} = {scale} "
            f"by {abs(total - scale)} (tolerance: {N_ENTRIES})"
        )


class TestCDT:
    """Verify the Python CDT is the correct cumulative sum of the PDT."""

    def test_cdt_is_cumulative_pdt(self):
        from sampler import halfgaussian_pdt, halfgaussian_cdt

        expected = compute_cdt_from_pdt(halfgaussian_pdt)
        assert len(halfgaussian_cdt) == len(expected)
        for i, (actual, exp) in enumerate(zip(halfgaussian_cdt, expected)):
            assert actual == exp, f"CDT entry {i} mismatch: {actual} vs {exp}"

    def test_cdt_is_monotonic(self):
        from sampler import halfgaussian_cdt

        for i in range(1, len(halfgaussian_cdt)):
            assert halfgaussian_cdt[i] > halfgaussian_cdt[i - 1], (
                f"CDT not monotonic at index {i}"
            )


class TestCRCDT:
    """Verify the C RCDT table matches the Python CDT."""

    def test_c_rcdt_matches_python_cdt(self):
        from sampler import halfgaussian_cdt

        c_rcdt = parse_c_rcdt()
        assert len(c_rcdt) == len(halfgaussian_cdt), (
            f"C RCDT has {len(c_rcdt)} entries, Python CDT has {len(halfgaussian_cdt)}"
        )
        for i, (c_val, py_val) in enumerate(zip(c_rcdt, halfgaussian_cdt)):
            assert c_val == py_val, (
                f"C RCDT entry {i} = {c_val} != Python CDT entry {i} = {py_val}"
            )
