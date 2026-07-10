"""
End-to-end distribution test of samplerz(), the sampler proposed in [HPRR20].

sampler.c gets its output chi2-tested against the folded PDT
(test_sampler_c.py), but that is only the *base* sampler (half-Gaussian plus
sign bit, no BerExp). samplerz() -- the full proposed sampler, with the BerExp
rejection step and the ccs scaling -- was previously covered only by
test_sampler_determinism.py, which checks reproducibility and never the output
distribution. This closes that gap: the artifact's headline deliverable now has
a distribution test.

The test is seeded, so it is deterministic and never flaky. That makes it a
regression guard on a fixed stream rather than a statistical assessment of the
sampler in general -- the same trade-off test_sampler_c.py makes. Observed
Fisher p-values run from 0.11 to 0.82, two orders of magnitude above the
alpha = 0.001 threshold, so there is ample margin.
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sampler import samplerz, sigmin
from saga import UnivariateSamples

SEED = 1234
N = 20_000
MC_B = 200

# Boundaries that matter: centered, the worst-case fractional center for the
# sign/BerExp recombination, both ends of the permitted sigma range, and a
# large center.
PARAMS = [
    (0.0, 1.55),        # typical
    (0.5, 1.55),        # hostile fractional mu
    (0.0, sigmin),      # lower sigma bound (discreteness is strongest here)
    (0.0, 1.80),        # near the upper bound sigma0 = 1.8205
    (3.7, 1.55),        # nonzero center
]


def _draw(mu, sigma, seed=SEED):
    random.seed(seed)
    return [samplerz(mu, sigma) for _ in range(N)]


@pytest.mark.parametrize("mu,sigma", PARAMS,
                         ids=[f"mu={m}_sigma={s:.4f}" for m, s in PARAMS])
def test_samplerz_passes_battery(mu, sigma):
    """samplerz output must pass the chi-square and the extended battery."""
    samples = _draw(mu, sigma)
    uv = UnivariateSamples(mu, sigma, samples)
    ext = uv.run_extended_battery(mc_B=MC_B)

    assert uv.outlier == 0, f"samplerz({mu}, {sigma}): {uv.outlier} outliers"
    assert uv.is_valid, (
        f"samplerz({mu}, {sigma}) fails chi-square: p={uv.chi2_pvalue:.6f}")
    assert ext["all_pass"], (
        f"samplerz({mu}, {sigma}) fails the extended battery "
        f"(global p={ext['global_pvalue']:.6f}); rejecting: "
        + ", ".join(ext["bonferroni_rejects"] or ["<none, global Fisher only>"]))


def test_battery_rejects_a_wrong_sigma():
    """Negative control: the battery must have teeth.

    Test samplerz(0, 1.55) output against the *wrong* sigma. If this passed,
    test_samplerz_passes_battery above would be vacuous.
    """
    samples = _draw(0.0, 1.55)
    uv = UnivariateSamples(0.0, 1.55 * 1.15, samples)   # declared sigma is wrong
    ext = uv.run_extended_battery(mc_B=MC_B)
    assert not (uv.is_valid and ext["all_pass"]), (
        "the battery accepted samples drawn at the wrong sigma -- "
        "test_samplerz_passes_battery would be vacuous")
