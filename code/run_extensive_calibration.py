"""
Extensive calibration: power matrix across multiple (mu, sigma) pairs,
fine-grained flaw gradients, and large replicate counts.

Usage:
    python run_extensive_calibration.py [--reps 100] [--n 10000]

Runtime estimate: 30-90 minutes depending on hardware.
"""

import argparse
import json
import time
import sys
import numpy as np
from scipy.stats import kstest

from saga import UnivariateSamples, make_gaussian_pdt
from univariate_tests import (
    tail_exceedance, sign_halfgaussian, discrete_anderson_darling,
    higher_criticism, ljung_box, runs_test, block_homogeneity,
)


# ---------------------------------------------------------------------------
# PDT helpers
# ---------------------------------------------------------------------------

def _pdt_arrays(mu, sigma, tau=14):
    pdt = make_gaussian_pdt(mu, sigma)
    support = np.array(sorted(pdt.keys()))
    probs = np.array([pdt[z] for z in support])
    return support, probs


# ---------------------------------------------------------------------------
# Flaw generators
# ---------------------------------------------------------------------------

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
    idx = min(mode_idx + 2, len(p) - 1)
    p_err = p.copy()
    p_err[idx] *= factor
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


def gen_bimodal(rng, mu, sigma, n, separation):
    """Mix two shifted Gaussians — creates excess kurtosis."""
    s1, p1 = _pdt_arrays(mu - separation, sigma * 0.8)
    s2, p2 = _pdt_arrays(mu + separation, sigma * 0.8)
    half = n // 2
    a = rng.choice(s1, size=half, p=p1).tolist()
    b = rng.choice(s2, size=n - half, p=p2).tolist()
    combined = a + b
    rng.shuffle(combined)
    return [int(x) for x in combined]


def gen_periodic(rng, mu, sigma, n, period):
    """Periodic pattern in the sample stream."""
    s, p = _pdt_arrays(mu, sigma)
    samples = rng.choice(s, size=n, p=p).tolist()
    for i in range(0, n, period):
        samples[i] = int(round(mu))
    return samples


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

PARAM_GRID = [
    # (mu, sigma, label)
    (0.0, 1.55, "base_sampler"),           # BaseSampler sigma
    (0.0, 1.2778, "sigmin"),               # sigmin
    (0.25, 1.55, "mu_quarter"),            # hostile fractional mu
    (0.5, 1.55, "mu_half"),                # worst-case: exactly between integers
    (0.499, 1.55, "mu_near_half"),         # near-boundary
    (0.0, 4.05, "keygen_512"),             # Falcon-512 keygen sigma
    (0.0, 2.87, "keygen_1024"),            # Falcon-1024 keygen sigma
]

# ---------------------------------------------------------------------------
# Flaw gradient library
# ---------------------------------------------------------------------------

def build_flaw_library():
    """Build flaws at 10+ gradient levels per type."""
    flaws = []

    # Sigma shifts: 0.1% to 20%
    for eps in [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        flaws.append((f"sigma+{eps*100:.1f}%",
                       lambda rng, mu, s, n, e=eps: gen_sigma_shift(rng, mu, s, n, e)))

    # Mu shifts: 0.01 to 1.0
    for delta in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
        flaws.append((f"mu+{delta}",
                       lambda rng, mu, s, n, d=delta: gen_mu_shift(rng, mu, s, n, d)))

    # Sign coupling: 50.5/49.5 to 65/35
    for bias in [0.505, 0.51, 0.52, 0.53, 0.55, 0.57, 0.60, 0.65]:
        flaws.append((f"sign_{int(bias*100)}/{int((1-bias)*100)}",
                       lambda rng, mu, s, n, b=bias: gen_sign_coupling(rng, mu, s, n, b)))

    # Tail truncation: 2.5-sigma to 6-sigma
    for k in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        flaws.append((f"trunc_{k}sig",
                       lambda rng, mu, s, n, kk=k: gen_tail_truncation(rng, mu, s, n, kk)))

    # Table error: 1.1x to 5x
    for factor in [1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        flaws.append((f"table_{factor}x",
                       lambda rng, mu, s, n, f=factor: gen_table_error(rng, mu, s, n, f)))

    # Contamination: 0.1% to 10%
    for eps in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        flaws.append((f"contam_{eps*100:.1f}%",
                       lambda rng, mu, s, n, e=eps: gen_contamination(rng, mu, s, n, e)))

    # Markov coupling: 0.01 to 0.5
    for rho in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        flaws.append((f"markov_{rho}",
                       lambda rng, mu, s, n, r=rho: gen_markov(rng, mu, s, n, r)))

    # Bimodal contamination
    for sep in [1.0, 2.0, 3.0]:
        flaws.append((f"bimodal_sep{sep}",
                       lambda rng, mu, s, n, sp=sep: gen_bimodal(rng, mu, s, n, sp)))

    # Periodic pattern
    for period in [10, 50, 100]:
        flaws.append((f"periodic_{period}",
                       lambda rng, mu, s, n, p=period: gen_periodic(rng, mu, s, n, p)))

    return flaws


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

TEST_NAMES = [
    "chi2", "tail_exceedance", "sign_halfgauss", "discrete_ad",
    "higher_crit", "ljung_box", "runs_test", "block_homog",
]


def _run_test(test_name, mu, sigma, samples, alpha=0.001):
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


# ---------------------------------------------------------------------------
# Power matrix per parameter point
# ---------------------------------------------------------------------------

def compute_power_matrix_single(mu, sigma, flaws, n, reps, alpha, seed):
    matrix = {}
    # Enumerate index for the seed offset: hash(flaw_name) is salted per
    # process (PYTHONHASHSEED) and would make the matrix non-reproducible.
    for flaw_idx, (flaw_name, flaw_gen) in enumerate(flaws):
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
# False alarm rate
# ---------------------------------------------------------------------------

def compute_false_alarm_rates(mu, sigma, n, reps, alpha, seed):
    rates = {}
    for test_name in TEST_NAMES:
        false_alarms = 0
        for r in range(reps):
            rng = np.random.default_rng(seed + r * 7919)
            samples = gen_perfect(rng, mu, sigma, n)
            if _run_test(test_name, mu, sigma, samples, alpha):
                false_alarms += 1
        rates[test_name] = false_alarms / reps
    return rates


# ---------------------------------------------------------------------------
# p-value uniformity
# ---------------------------------------------------------------------------

def pvalue_uniformity(mu, sigma, n, reps, seed):
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
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "mean_pvalue": float(pvalues.mean()),
        "passes": ks_pval > 0.05,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_power_matrix(matrix, param_label):
    print(f"\n{'='*120}")
    print(f"  Power matrix for {param_label}")
    print(f"{'='*120}")
    header = f"{'Flaw':<20}" + "".join(f"{t:>14}" for t in TEST_NAMES)
    print(header)
    print("-" * len(header))

    for flaw_name in matrix:
        row = matrix[flaw_name]
        line = f"{flaw_name:<20}"
        for t in TEST_NAMES:
            val = row[t]
            if val >= 0.8:
                line += f"\033[92m{'%3.0f%%' % (val*100):>14}\033[0m"
            elif val > 0.2:
                line += f"\033[93m{'%3.0f%%' % (val*100):>14}\033[0m"
            elif val > 0:
                line += f"\033[91m{'%3.0f%%' % (val*100):>14}\033[0m"
            else:
                line += f"{'—':>14}"
        print(line)


def print_summary(all_results):
    print(f"\n{'='*120}")
    print("  SUMMARY")
    print(f"{'='*120}")

    # Detection threshold analysis: per flaw type, what's the weakest
    # strength detectable at 80% power by any test?
    print("\n--- Detection thresholds (weakest flaw detected at >=80% by any test) ---\n")

    flaw_families = {}
    for param_label, data in all_results.items():
        matrix = data["power_matrix"]
        for flaw_name, row in matrix.items():
            best_power = max(row.values())
            family = flaw_name.rstrip('0123456789.%x_').rstrip('_')
            if family not in flaw_families:
                flaw_families[family] = []
            flaw_families[family].append({
                "flaw": flaw_name,
                "param": param_label,
                "best_power": best_power,
                "best_test": max(row, key=row.get),
            })

    for family, entries in sorted(flaw_families.items()):
        detected = [e for e in entries if e["best_power"] >= 0.8]
        if detected:
            weakest = min(detected, key=lambda e: e["best_power"])
            print(f"  {family:<20} threshold ~ {weakest['flaw']:<20} "
                  f"(power {weakest['best_power']:.0%} by {weakest['best_test']} "
                  f"at {weakest['param']})")
        else:
            strongest = max(entries, key=lambda e: e["best_power"])
            print(f"  {family:<20} NOT detected at 80%  "
                  f"(best {strongest['best_power']:.0%} for {strongest['flaw']} "
                  f"at {strongest['param']})")

    # False alarm summary
    print("\n--- False alarm rates ---\n")
    for param_label, data in all_results.items():
        if "false_alarm_rates" in data:
            far = data["false_alarm_rates"]
            total = sum(far.values())
            print(f"  {param_label}: {' | '.join(f'{t}={v:.1%}' for t, v in far.items())}")

    # p-value uniformity
    print("\n--- p-value uniformity (chi2 under H0 -> U[0,1]) ---\n")
    for param_label, data in all_results.items():
        if "pvalue_uniformity" in data:
            pvu = data["pvalue_uniformity"]
            status = "PASS" if pvu["passes"] else "FAIL"
            print(f"  {param_label}: KS={pvu['ks_stat']:.4f}, "
                  f"p={pvu['ks_pvalue']:.4f}, "
                  f"mean_p={pvu['mean_pvalue']:.3f} [{status}]")

    # Battery selection: tests that win on at least one flaw across all params
    print("\n--- Battery selection (tests that win on at least one flaw) ---\n")
    winners_global = {}
    for param_label, data in all_results.items():
        matrix = data["power_matrix"]
        for flaw_name, row in matrix.items():
            best_test = max(row, key=row.get)
            best_val = row[best_test]
            if best_val > 0:
                if best_test not in winners_global:
                    winners_global[best_test] = []
                winners_global[best_test].append(f"{flaw_name}@{param_label}")

    for test in sorted(winners_global.keys()):
        niches = winners_global[test]
        print(f"  {test}: wins on {len(niches)} (flaw, param) combos")
        for niche in niches[:5]:
            print(f"    - {niche}")
        if len(niches) > 5:
            print(f"    ... and {len(niches)-5} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extensive SAGA calibration")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--reps", type=int, default=100,
                        help="Replicates per (test, flaw) cell")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--output", default="extensive_calibration.json")
    parser.add_argument("--params", nargs="*", default=None,
                        help="Subset of param labels to run (default: all)")
    args = parser.parse_args()

    flaws = build_flaw_library()
    n_flaws = len(flaws)
    n_tests = len(TEST_NAMES)
    n_params = len(PARAM_GRID) if args.params is None else len(args.params)

    total_cells = n_params * n_flaws * n_tests * args.reps
    print(f"Extensive calibration:")
    print(f"  {n_params} parameter points")
    print(f"  {n_flaws} flaw types")
    print(f"  {n_tests} tests")
    print(f"  {args.reps} replicates per cell")
    print(f"  n = {args.n} samples per replicate")
    print(f"  Total cells: {total_cells:,}")
    print(f"  + {n_params} false-alarm sweeps ({args.reps} reps each)")
    print(f"  + {n_params} p-value uniformity checks (50 reps each)")
    print()

    all_results = {}
    t_start = time.time()

    for mu, sigma, label in PARAM_GRID:
        if args.params is not None and label not in args.params:
            continue

        t_param = time.time()
        print(f"--- {label} (mu={mu}, sigma={sigma}) ---")

        # Power matrix
        print(f"  Computing power matrix ({n_flaws} flaws x {n_tests} tests x {args.reps} reps)...")
        sys.stdout.flush()
        matrix = compute_power_matrix_single(
            mu, sigma, flaws, args.n, args.reps, args.alpha, seed=2026)
        print_power_matrix(matrix, label)

        # False alarm rates
        print(f"  Computing false alarm rates ({args.reps} reps)...")
        sys.stdout.flush()
        far = compute_false_alarm_rates(
            mu, sigma, args.n, args.reps, args.alpha, seed=5000)
        print(f"  False alarms: {far}")

        # p-value uniformity
        print(f"  p-value uniformity check...")
        sys.stdout.flush()
        pvu = pvalue_uniformity(mu, sigma, args.n, reps=50, seed=3000)
        status = "PASS" if pvu["passes"] else "FAIL"
        print(f"  KS={pvu['ks_stat']:.4f}, p={pvu['ks_pvalue']:.4f}, "
              f"mean_p={pvu['mean_pvalue']:.3f} [{status}]")

        elapsed = time.time() - t_param
        print(f"  Done in {elapsed:.1f}s\n")

        all_results[label] = {
            "mu": mu,
            "sigma": sigma,
            "power_matrix": matrix,
            "false_alarm_rates": far,
            "pvalue_uniformity": pvu,
        }

    # Summary
    print_summary(all_results)

    # Save
    total_time = time.time() - t_start

    class SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output = {
        "params": {
            "n": args.n,
            "reps": args.reps,
            "alpha": args.alpha,
        },
        "n_flaws": n_flaws,
        "n_tests": n_tests,
        "n_param_points": len(all_results),
        "total_time_seconds": total_time,
        "results": all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, cls=SafeEncoder)

    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
