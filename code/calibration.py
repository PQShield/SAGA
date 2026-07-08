"""
Suite calibration: flaw-injection library, power matrix, and p-value
uniformity meta-test.

The power matrix measures each test's detection power against each
flaw type at various strengths and sample sizes. This decides which
tests earn a spot in the default battery and tells users what N buys.

Usage:
    python calibration.py [--n 10000] [--reps 100]
"""

import argparse
import json
import numpy as np
from scipy.stats import kstest

from saga import UnivariateSamples, make_gaussian_pdt
from univariate_tests import (
    tail_exceedance, sign_halfgaussian, discrete_anderson_darling,
    higher_criticism, ljung_box, runs_test, block_homogeneity,
)


# ---------------------------------------------------------------------------
# Flaw generators (distribution-level, cheap at large N)
# ---------------------------------------------------------------------------

def _pdt_arrays(mu, sigma, tau=14):
    pdt = make_gaussian_pdt(mu, sigma)
    support = np.array(sorted(pdt.keys()))
    probs = np.array([pdt[z] for z in support])
    return support, probs


def gen_perfect(rng, mu, sigma, n):
    s, p = _pdt_arrays(mu, sigma)
    return rng.choice(s, size=n, p=p).tolist()


def gen_sigma_shift(rng, mu, sigma, n, epsilon):
    s, p = _pdt_arrays(mu, sigma * (1 + epsilon))
    return rng.choice(s, size=n, p=p).tolist()


def gen_mu_shift(rng, mu, sigma, n, delta):
    s, p = _pdt_arrays(mu + delta, sigma)
    return rng.choice(s, size=n, p=p).tolist()


def gen_sign_coupling(rng, mu, sigma, n, bias):
    s, p = _pdt_arrays(mu, sigma)
    p_biased = p.copy()
    for i, z in enumerate(s):
        if z > mu:
            p_biased[i] *= bias / 0.5
        elif z < mu:
            p_biased[i] *= (1 - bias) / 0.5
    p_biased /= p_biased.sum()
    return rng.choice(s, size=n, p=p_biased).tolist()


def gen_tail_truncation(rng, mu, sigma, n, k_sigma):
    s, p = _pdt_arrays(mu, sigma)
    mask = np.abs(s - mu) <= k_sigma * sigma
    p_trunc = p * mask
    p_trunc /= p_trunc.sum()
    return rng.choice(s, size=n, p=p_trunc).tolist()


def gen_table_error(rng, mu, sigma, n, factor):
    s, p = _pdt_arrays(mu, sigma)
    mode_idx = np.argmax(p)
    p_err = p.copy()
    p_err[mode_idx + 2] *= factor
    p_err /= p_err.sum()
    return rng.choice(s, size=n, p=p_err).tolist()


def gen_contamination(rng, mu, sigma, n, epsilon):
    s, p = _pdt_arrays(mu, sigma)
    p_cont = (1 - epsilon) * p + epsilon / len(p)
    p_cont /= p_cont.sum()
    return rng.choice(s, size=n, p=p_cont).tolist()


def gen_markov(rng, mu, sigma, n, rho):
    s, p = _pdt_arrays(mu, sigma)
    samples = [rng.choice(s, p=p)]
    for _ in range(n - 1):
        if rng.random() < rho:
            samples.append(samples[-1])
        else:
            samples.append(rng.choice(s, p=p))
    return [int(x) for x in samples]


# ---------------------------------------------------------------------------
# Flaw library
# ---------------------------------------------------------------------------

FLAW_LIBRARY = [
    ("sigma+1%", lambda rng, mu, s, n: gen_sigma_shift(rng, mu, s, n, 0.01)),
    ("sigma+5%", lambda rng, mu, s, n: gen_sigma_shift(rng, mu, s, n, 0.05)),
    ("sigma+10%", lambda rng, mu, s, n: gen_sigma_shift(rng, mu, s, n, 0.10)),
    ("mu+0.1", lambda rng, mu, s, n: gen_mu_shift(rng, mu, s, n, 0.1)),
    ("mu+0.5", lambda rng, mu, s, n: gen_mu_shift(rng, mu, s, n, 0.5)),
    ("sign_52/48", lambda rng, mu, s, n: gen_sign_coupling(rng, mu, s, n, 0.52)),
    ("sign_55/45", lambda rng, mu, s, n: gen_sign_coupling(rng, mu, s, n, 0.55)),
    ("sign_60/40", lambda rng, mu, s, n: gen_sign_coupling(rng, mu, s, n, 0.60)),
    ("trunc_3sig", lambda rng, mu, s, n: gen_tail_truncation(rng, mu, s, n, 3)),
    ("trunc_4sig", lambda rng, mu, s, n: gen_tail_truncation(rng, mu, s, n, 4)),
    ("table_2x", lambda rng, mu, s, n: gen_table_error(rng, mu, s, n, 2.0)),
    ("table_3x", lambda rng, mu, s, n: gen_table_error(rng, mu, s, n, 3.0)),
    ("contam_1%", lambda rng, mu, s, n: gen_contamination(rng, mu, s, n, 0.01)),
    ("contam_5%", lambda rng, mu, s, n: gen_contamination(rng, mu, s, n, 0.05)),
    ("markov_0.1", lambda rng, mu, s, n: gen_markov(rng, mu, s, n, 0.1)),
    ("markov_0.3", lambda rng, mu, s, n: gen_markov(rng, mu, s, n, 0.3)),
]


# ---------------------------------------------------------------------------
# Test runners (return True if flaw detected)
# ---------------------------------------------------------------------------

def _run_test(test_name, mu, sigma, samples, alpha=0.001):
    """Run a single test, return True if it rejects (detects flaw)."""
    try:
        if test_name == "chi2":
            uv = UnivariateSamples(mu, sigma, samples)
            return not uv.is_valid
        elif test_name == "tail_exceedance":
            r = tail_exceedance(mu, sigma, samples, alpha=alpha)
            return not r["passes"]
        elif test_name == "sign_halfgauss":
            r = sign_halfgaussian(mu, sigma, samples, alpha=alpha)
            return not r["passes"]
        elif test_name == "discrete_ad":
            r = discrete_anderson_darling(mu, sigma, samples,
                                          alpha=alpha, mc_B=200)
            return not r["passes"]
        elif test_name == "higher_crit":
            r = higher_criticism(mu, sigma, samples,
                                 alpha=alpha, mc_B=200)
            return not r["passes"]
        elif test_name == "ljung_box":
            r = ljung_box(samples, alpha=alpha)
            return not r["passes"]
        elif test_name == "runs_test":
            r = runs_test(samples, alpha=alpha)
            return not r["passes"]
        elif test_name == "block_homog":
            r = block_homogeneity(mu, sigma, samples, alpha=alpha)
            return not r["passes"]
    except Exception:
        return False
    return False


TEST_NAMES = [
    "chi2", "tail_exceedance", "sign_halfgauss", "discrete_ad",
    "higher_crit", "ljung_box", "runs_test", "block_homog",
]


# ---------------------------------------------------------------------------
# Power matrix computation
# ---------------------------------------------------------------------------

def compute_power_matrix(mu=0.0, sigma=1.55, n=10000, reps=50,
                         alpha=0.001, seed=2026):
    """
    Compute the power matrix: for each (test, flaw), the detection
    rate over `reps` replicates.

    Returns:
        dict with flaw names as keys, each mapping to a dict of
        test_name -> detection_rate.
    """
    matrix = {}

    # Enumerate index for the seed offset: hash(flaw_name) is salted per
    # process (PYTHONHASHSEED) and would make the matrix non-reproducible.
    for flaw_idx, (flaw_name, flaw_gen) in enumerate(FLAW_LIBRARY):
        row = {}
        for test_name in TEST_NAMES:
            detections = 0
            for r in range(reps):
                rng = np.random.default_rng(seed + r * 1000 +
                                            flaw_idx * 1_000_000)
                samples = flaw_gen(rng, mu, sigma, n)
                if _run_test(test_name, mu, sigma, samples, alpha):
                    detections += 1
            row[test_name] = detections / reps
        matrix[flaw_name] = row

    return matrix


# ---------------------------------------------------------------------------
# p-value uniformity meta-test
# ---------------------------------------------------------------------------

def pvalue_uniformity(mu=0.0, sigma=1.55, n=10000, reps=100,
                      seed=3000):
    """
    Generate reps perfect samples, collect chi-square p-values,
    test their uniformity on [0,1] via KS.

    If the suite is well-calibrated, p-values under H0 should be
    U[0,1]. This meta-test detects miscalibration of the suite itself.
    """
    support, probs = _pdt_arrays(mu, sigma)
    pvalues = []

    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        samples = rng.choice(support, size=n, p=probs).tolist()
        uv = UnivariateSamples(mu, sigma, samples)
        pvalues.append(float(uv.chi2_pvalue))

    pvalues = np.array(pvalues)
    ks_stat, ks_pval = kstest(pvalues, 'uniform')

    return {
        "test": "pvalue_uniformity",
        "n_replicates": reps,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "mean_pvalue": float(pvalues.mean()),
        "passes": ks_pval > 0.05,
    }


# ---------------------------------------------------------------------------
# Pretty-print the power matrix
# ---------------------------------------------------------------------------

def print_power_matrix(matrix):
    """Print the power matrix as a formatted table."""
    header = f"{'Flaw':<16}" + "".join(f"{t:>14}" for t in TEST_NAMES)
    print(header)
    print("-" * len(header))

    for flaw_name in matrix:
        row = matrix[flaw_name]
        line = f"{flaw_name:<16}"
        for t in TEST_NAMES:
            val = row[t]
            if val >= 0.8:
                line += f"{'%.0f%%' % (val*100):>14}"
            elif val > 0:
                line += f"{'%.0f%%' % (val*100):>14}"
            else:
                line += f"{'—':>14}"
        print(line)


def select_battery(matrix):
    """
    Select the default battery: keep tests that win on at least one
    flaw (highest detection rate for that flaw among all tests).
    """
    winners = set()
    for flaw_name, row in matrix.items():
        best_test = max(row, key=row.get)
        if row[best_test] > 0:
            winners.add(best_test)

    niche = {}
    for t in winners:
        flaws_won = []
        for flaw_name, row in matrix.items():
            best_rate = max(row.values())
            if row[t] == best_rate and best_rate > 0:
                flaws_won.append(flaw_name)
        niche[t] = flaws_won

    return {"battery": sorted(winners), "niches": niche}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAGA suite calibration: power matrix")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--reps", type=int, default=20,
                        help="Replicates per (test, flaw) cell")
    parser.add_argument("--output", default="power_matrix.json")
    args = parser.parse_args()

    print(f"Computing power matrix (n={args.n}, reps={args.reps})...\n")
    matrix = compute_power_matrix(n=args.n, reps=args.reps)
    print_power_matrix(matrix)

    print("\n--- p-value uniformity meta-test ---")
    pvu = pvalue_uniformity(n=args.n, reps=50)
    print(f"KS stat={pvu['ks_stat']:.4f}, p={pvu['ks_pvalue']:.4f}, "
          f"mean_p={pvu['mean_pvalue']:.3f}, "
          f"{'PASS' if pvu['passes'] else 'FAIL'}")

    print("\n--- Default battery selection ---")
    battery = select_battery(matrix)
    print(f"Tests in default battery: {battery['battery']}")
    for t, flaws in battery["niches"].items():
        print(f"  {t}: wins on {', '.join(flaws)}")

    class SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    with open(args.output, 'w') as f:
        json.dump({
            "power_matrix": matrix,
            "pvalue_uniformity": pvu,
            "battery_selection": battery,
            "params": {"n": args.n, "reps": args.reps},
        }, f, indent=2, cls=SafeEncoder)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
