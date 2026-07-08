"""
Compile sampler.c, run it, and chi2-test the output against the folded PDT.

This closes the blind spot that let the sampler.c mirror bug through:
test_tables.py validates the *table*, but not the *comparison logic*. An
ascending CDT paired with the RCDT "v < row" count samples 18 - z0 (mass
at |z| ~ 18) yet leaves the table lint green. Testing the actual compiled
output against the target distribution is the only thing that catches it.

The folded PDT is the distribution of z = b + (2b-1)*z0, where z0 follows
the half-Gaussian PDT and b is a uniform sign bit:
    z = 0      <- z0=0, b=0        P(z0=0)/2
    z = 1      <- z0=0, b=1        P(z0=0)/2
    z = -k     <- z0=k, b=0  (k>0) P(z0=k)/2
    z = k+1    <- z0=k, b=1  (k>0) P(z0=k)/2
"""
import os
import sys
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
from scipy.stats import chisquare

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sampler import halfgaussian_pdt, cdt_precision

CODE_DIR = os.path.join(os.path.dirname(__file__), "..")
SAMPLER_C = os.path.join(CODE_DIR, "sampler.c")

N_SAMPLES = 500_000


def _compiler():
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return cc
    return None


def folded_pdt():
    """Return (support, probs) for z = b + (2b-1)*z0 over the full range."""
    scale = float(sum(halfgaussian_pdt))  # normalizes; PDT sums to 2^72
    probs = {}
    for z0, count in enumerate(halfgaussian_pdt):
        p = count / scale
        if z0 == 0:
            probs[0] = probs.get(0, 0.0) + p / 2      # b=0
            probs[1] = probs.get(1, 0.0) + p / 2      # b=1
        else:
            probs[-z0] = probs.get(-z0, 0.0) + p / 2      # b=0
            probs[z0 + 1] = probs.get(z0 + 1, 0.0) + p / 2  # b=1
    support = sorted(probs)
    return support, np.array([probs[z] for z in support])


@pytest.fixture(scope="module")
def c_samples():
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler (cc/gcc/clang) available")

    with tempfile.TemporaryDirectory() as tmp:
        binpath = os.path.join(tmp, "sampler_bin")
        compile_proc = subprocess.run(
            [cc, "-O2", SAMPLER_C, "-o", binpath],
            capture_output=True, text=True)
        assert compile_proc.returncode == 0, (
            f"sampler.c failed to compile:\n{compile_proc.stderr}")

        run_proc = subprocess.run(
            [binpath, str(N_SAMPLES)],
            capture_output=True, text=True, cwd=tmp)
        assert run_proc.returncode == 0, (
            f"sampler binary crashed:\n{run_proc.stderr}")

        samples = np.array([int(line) for line in
                            run_proc.stdout.split() if line])
    return samples


class TestCompiledSampler:
    """The compiled C sampler must match the folded PDT."""

    def test_sample_count(self, c_samples):
        assert len(c_samples) == N_SAMPLES

    def test_not_mirrored(self, c_samples):
        # Direct guard on the exact bug: the correct distribution has ~90%
        # of its mass in |z| <= 3; the mirrored one piles it at |z| ~ 18.
        frac_central = np.mean(np.abs(c_samples) <= 3)
        assert frac_central > 0.85, (
            f"only {frac_central:.1%} of mass in |z|<=3 -- sampler looks "
            f"mirrored or misconfigured (mean={c_samples.mean():.2f})")

    def test_mean_near_half(self, c_samples):
        # E[z] = 0.5 by the b/(1-b) symmetry around the 0/1 split.
        assert abs(c_samples.mean() - 0.5) < 0.05, (
            f"sample mean {c_samples.mean():.3f} far from 0.5")

    def test_chi2_against_folded_pdt(self, c_samples):
        support, probs = folded_pdt()
        n = len(c_samples)
        idx = {z: i for i, z in enumerate(support)}

        observed = np.zeros(len(support))
        for z in c_samples:
            if z in idx:
                observed[idx[z]] += 1
            # out-of-support samples (should be ~none) fall through and
            # will surface as a count mismatch below.

        expected = probs * n

        # Merge bins with expected < 5 into their neighbours (chi2 validity).
        obs_m, exp_m = [], []
        acc_o, acc_e = 0.0, 0.0
        for o, e in zip(observed, expected):
            acc_o += o
            acc_e += e
            if acc_e >= 5:
                obs_m.append(acc_o)
                exp_m.append(acc_e)
                acc_o, acc_e = 0.0, 0.0
        if acc_e > 0:
            obs_m[-1] += acc_o
            exp_m[-1] += acc_e

        obs_m = np.array(obs_m)
        exp_m = np.array(exp_m)
        # Rescale expected to match observed total (out-of-support drift).
        exp_m *= obs_m.sum() / exp_m.sum()

        stat, pval = chisquare(obs_m, f_exp=exp_m)
        assert pval > 0.001, (
            f"compiled sampler output rejects the folded PDT: "
            f"chi2={stat:.1f}, p={pval:.2e}")
