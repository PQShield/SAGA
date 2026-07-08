"""
Extended univariate test battery for discrete Gaussian samplers.

Tests:
    - tail_exceedance: exact binomial tests at 3-6 sigma thresholds
    - sign_halfgaussian: factorization into sign and magnitude
    - moment_cis: confidence intervals on sample moments
    - discrete_anderson_darling: tail-weighted EDF test (MC-calibrated)
    - higher_criticism: max standardized CDF deviation (MC-calibrated)

All tests take (mu, sigma, samples, pdt) and return a result dict.
The MC calibration helper enables correct p-values for tests without
textbook null distributions.
"""

import numpy as np
from math import floor, sqrt
from scipy.stats import binomtest

from saga import make_gaussian_pdt


# ---------------------------------------------------------------------------
# MC calibration helper
# ---------------------------------------------------------------------------

def mc_calibrate(statistic_fn, pdt_support, pdt_probs, n, B=1000, seed=0):
    """
    Monte Carlo calibration: compute the null distribution of a test
    statistic by sampling B replicates of size n from the exact PDT.

    Args:
        statistic_fn: callable(samples) -> float (the test statistic)
        pdt_support: array of support values
        pdt_probs: array of probabilities (same length)
        n: sample size
        B: number of MC replicates
        seed: RNG seed for reproducibility

    Returns:
        null_distribution: sorted array of B statistic values under H0
    """
    rng = np.random.default_rng(seed)
    null_stats = np.empty(B)
    for i in range(B):
        samples = rng.choice(pdt_support, size=n, p=pdt_probs)
        null_stats[i] = statistic_fn(samples)
    null_stats.sort()
    return null_stats


def mc_pvalue(observed, null_distribution):
    """
    MC p-value: (1 + #{null >= observed}) / (B + 1).

    The +1 correction (Davison & Hinkley) keeps the test valid at finite
    B -- the naive #{null >= obs}/B can return 0, making the achievable
    size coarser than the nominal alpha. For a target alpha, use
    B >= 10/alpha replicates so 1/(B+1) sits comfortably below alpha.
    """
    null_distribution = np.asarray(null_distribution)
    B = len(null_distribution)
    return (1 + int(np.sum(null_distribution >= observed))) / (B + 1)


# ---------------------------------------------------------------------------
# Helper: extract PDT as arrays
# ---------------------------------------------------------------------------

def pdt_arrays(mu, sigma, tau=14):
    """Convert make_gaussian_pdt dict to aligned arrays."""
    pdt = make_gaussian_pdt(mu, sigma)
    support = np.array(sorted(pdt.keys()))
    probs = np.array([pdt[z] for z in support])
    return support, probs


# ---------------------------------------------------------------------------
# 3.1 Tail exceedance
# ---------------------------------------------------------------------------

def tail_exceedance(mu, sigma, samples, tau=14, k_values=(3, 4, 5, 6),
                    alpha=0.001):
    """
    Exact binomial test on tail counts at k*sigma thresholds.

    For each k, count samples with |z - mu| > k*sigma and test
    against the exact tail mass from the PDT.
    """
    support, probs = pdt_arrays(mu, sigma, tau)
    samples = np.asarray(samples)
    n = len(samples)
    results = {}

    for k in k_values:
        threshold = k * sigma
        tail_mask = np.abs(support - mu) > threshold
        expected_mass = probs[tail_mask].sum()

        observed_count = int(np.sum(np.abs(samples - mu) > threshold))

        if expected_mass > 0 and expected_mass < 1:
            bt = binomtest(observed_count, n, expected_mass)
            pval = bt.pvalue
        else:
            pval = 1.0

        results[f"{k}sigma"] = {
            "observed": observed_count,
            "expected": round(expected_mass * n, 2),
            "expected_mass": float(expected_mass),
            "pvalue": float(pval),
            "passes": pval > alpha,
        }

    overall_passes = all(r["passes"] for r in results.values())
    return {
        "test": "tail_exceedance",
        "thresholds": results,
        "passes": overall_passes,
    }


# ---------------------------------------------------------------------------
# 3.2 Sign / half-Gaussian factorization
# ---------------------------------------------------------------------------

def sign_halfgaussian(mu, sigma, samples, tau=14, alpha=0.001):
    """
    Test the sign/magnitude factorization of the sampler output.

    (a) Chi-square of |z| against the folded PDT
    (b) Per-|z| binomial test of sign balance (for near-integer mu)
    (c) Joint (sign, |z|) chi-square vs expected
    """
    support, probs = pdt_arrays(mu, sigma, tau)
    samples = np.asarray(samples)
    n = len(samples)
    c0 = mu - floor(mu)

    # Build folded PDT: P(|z-floor(mu)| = k)
    z_offsets = samples - floor(mu)
    abs_offsets = np.abs(z_offsets)

    # Support centered at floor(mu)
    max_abs = int(np.max(abs_offsets)) + 1
    folded_support = list(range(max_abs + 1))

    # Compute expected folded probabilities from the full PDT
    pdt = make_gaussian_pdt(mu, sigma)
    folded_expected = {}
    for k in folded_support:
        z_pos = floor(mu) + k
        z_neg = floor(mu) - k
        p = pdt.get(z_pos, 0)
        if k > 0:
            p += pdt.get(z_neg, 0)
        folded_expected[k] = p

    # (a) Chi-square on |z| distribution
    obs_counts = np.bincount(abs_offsets.astype(int),
                             minlength=max_abs + 1)
    exp_counts = np.array([folded_expected.get(k, 0) * n
                           for k in range(len(obs_counts))])

    # Merge bins with expected < 5
    obs_merged, exp_merged = [], []
    obs_acc, exp_acc = 0, 0.0
    for o, e in zip(obs_counts, exp_counts):
        obs_acc += o
        exp_acc += e
        if exp_acc >= 5:
            obs_merged.append(obs_acc)
            exp_merged.append(exp_acc)
            obs_acc, exp_acc = 0, 0.0
    if obs_acc > 0 or exp_acc > 0:
        if len(obs_merged) > 0:
            obs_merged[-1] += obs_acc
            exp_merged[-1] += exp_acc
        else:
            obs_merged.append(obs_acc)
            exp_merged.append(exp_acc)

    if len(obs_merged) > 1:
        from scipy.stats import chisquare
        obs_arr = np.array(obs_merged, dtype=float)
        exp_arr = np.array(exp_merged, dtype=float)
        diff = obs_arr.sum() - exp_arr.sum()
        exp_arr[len(exp_arr) // 2] += diff
        chi_stat, chi_p = chisquare(obs_arr, f_exp=exp_arr)
        magnitude_passes = chi_p > alpha
    else:
        chi_stat, chi_p = 0.0, 1.0
        magnitude_passes = True

    # (b) Sign balance (only for near-integer mu where sign should be ~50/50)
    is_near_integer = abs(c0) < 0.01 or abs(c0 - 1) < 0.01
    sign_results = {}
    if is_near_integer:
        for k in range(1, min(7, max_abs + 1)):
            mask = abs_offsets == k
            count_k = int(mask.sum())
            if count_k < 10:
                continue
            positive = int((z_offsets[mask] > 0).sum())
            bt = binomtest(positive, count_k, 0.5)
            sign_results[k] = {
                "positive": positive,
                "total": count_k,
                "pvalue": float(bt.pvalue),
            }

    sign_passes = all(r["pvalue"] > alpha for r in sign_results.values()) \
        if sign_results else True

    return {
        "test": "sign_halfgaussian",
        "magnitude_chi2": {"stat": float(chi_stat), "pvalue": float(chi_p),
                           "passes": magnitude_passes},
        "sign_balance": sign_results,
        "sign_passes": sign_passes,
        "is_near_integer_mu": is_near_integer,
        "passes": magnitude_passes and sign_passes,
    }


# ---------------------------------------------------------------------------
# 3.3 Moment confidence intervals
# ---------------------------------------------------------------------------

def moment_cis(mu, sigma, samples, confidence=0.95):
    """
    95% confidence intervals on sample moments.
    Informational — no hard pass/fail.
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    z = 1.96  # 95% CI

    emp_mean = np.mean(samples)
    emp_std = np.std(samples, ddof=1)
    emp_skew = float(np.mean(((samples - emp_mean) / emp_std) ** 3))
    emp_kurt = float(np.mean(((samples - emp_mean) / emp_std) ** 4) - 3)

    return {
        "test": "moment_cis",
        "mean": {
            "empirical": float(emp_mean),
            "expected": mu,
            "ci_low": float(emp_mean - z * sigma / sqrt(n)),
            "ci_high": float(emp_mean + z * sigma / sqrt(n)),
            "within_ci": abs(emp_mean - mu) <= z * sigma / sqrt(n),
        },
        "stdev": {
            "empirical": float(emp_std),
            "expected": sigma,
        },
        "skewness": {
            "empirical": emp_skew,
            "expected": 0,
            "se": sqrt(6.0 / n),
            "within_ci": abs(emp_skew) <= z * sqrt(6.0 / n),
        },
        "kurtosis": {
            "empirical": emp_kurt,
            "expected": 0,
            "se": sqrt(24.0 / n),
            "within_ci": abs(emp_kurt) <= z * sqrt(24.0 / n),
        },
    }


# ---------------------------------------------------------------------------
# 3.5 Discrete Anderson-Darling
# ---------------------------------------------------------------------------

def _ad_statistic(samples, support, probs):
    """
    Choulakian-Lockhart-Stephens discrete AD statistic.

    A^2 = (1/n) * sum_j Z_j^2 * p_{j+1} / (H_j * (1 - H_j))

    where Z_j = S_j - T_j (cumulative observed - cumulative expected),
    H_j = cumulative probability.
    """
    n = len(samples)
    counts = np.zeros(len(support))
    sample_counts = dict(zip(*np.unique(samples, return_counts=True)))
    for i, z in enumerate(support):
        counts[i] = sample_counts.get(z, 0)

    cum_obs = np.cumsum(counts)
    cum_exp = np.cumsum(probs) * n
    cum_prob = np.cumsum(probs)

    A2 = 0.0
    for j in range(len(support) - 1):
        Hj = cum_prob[j]
        if Hj > 0 and Hj < 1:
            Zj = cum_obs[j] - cum_exp[j]
            A2 += Zj ** 2 * probs[j + 1] / (Hj * (1 - Hj))

    return A2 / n


def discrete_anderson_darling(mu, sigma, samples, tau=14, alpha=0.001,
                              mc_B=1000, mc_seed=42):
    """
    Discrete Anderson-Darling test with MC-calibrated null.
    Tail-weighted: fills the blind spot of chi-square's bucket aggregation.
    """
    support, probs = pdt_arrays(mu, sigma, tau)
    samples = np.asarray(samples)
    n = len(samples)

    observed_stat = _ad_statistic(samples, support, probs)

    null_dist = mc_calibrate(
        lambda s: _ad_statistic(s, support, probs),
        support, probs, n, B=mc_B, seed=mc_seed
    )
    pvalue = mc_pvalue(observed_stat, null_dist)

    return {
        "test": "discrete_anderson_darling",
        "statistic": float(observed_stat),
        "pvalue": float(pvalue),
        "mc_replicates": mc_B,
        "passes": pvalue > alpha,
    }


# ---------------------------------------------------------------------------
# 3.6 Higher criticism
# ---------------------------------------------------------------------------

def _hc_statistic(samples, support, probs):
    """
    Higher criticism: max standardized EDF deviation.

    HC = max_j sqrt(n) * |F_hat(j) - F(j)| / sqrt(F(j) * (1 - F(j)))

    Near-optimal against sparse alternatives (single wrong table entry).
    """
    n = len(samples)
    counts = np.zeros(len(support))
    sample_counts = dict(zip(*np.unique(samples, return_counts=True)))
    for i, z in enumerate(support):
        counts[i] = sample_counts.get(z, 0)

    cum_obs = np.cumsum(counts) / n
    cum_prob = np.cumsum(probs)

    hc = 0.0
    for j in range(len(support) - 1):
        Fj = cum_prob[j]
        if Fj > 0.01 and Fj < 0.99:
            deviation = sqrt(n) * abs(cum_obs[j] - Fj) / sqrt(Fj * (1 - Fj))
            hc = max(hc, deviation)

    return hc


def higher_criticism(mu, sigma, samples, tau=14, alpha=0.001,
                     mc_B=1000, mc_seed=43):
    """
    Higher criticism test with MC-calibrated null.
    Near-optimal against sparse alternatives — a single wrong table entry.
    """
    support, probs = pdt_arrays(mu, sigma, tau)
    samples = np.asarray(samples)
    n = len(samples)

    observed_stat = _hc_statistic(samples, support, probs)

    null_dist = mc_calibrate(
        lambda s: _hc_statistic(s, support, probs),
        support, probs, n, B=mc_B, seed=mc_seed
    )
    pvalue = mc_pvalue(observed_stat, null_dist)

    return {
        "test": "higher_criticism",
        "statistic": float(observed_stat),
        "pvalue": float(pvalue),
        "mc_replicates": mc_B,
        "passes": pvalue > alpha,
    }


# ---------------------------------------------------------------------------
# 4.1 Ljung-Box autocorrelation test
# ---------------------------------------------------------------------------

def ljung_box(samples, max_lag=20, alpha=0.001):
    """
    Ljung-Box test for serial autocorrelation.

    Q(h) = n(n+2) * sum_{k=1}^{h} r_k^2 / (n-k) ~ chi2(h)

    Detects serial dependence that all distributional tests miss
    (e.g., Markov-coupled samplers with perfect marginals).
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    h = min(max_lag, n // 5)

    mean = np.mean(samples)
    centered = samples - mean
    var = np.sum(centered ** 2)

    if var == 0:
        return {"test": "ljung_box", "statistic": 0.0,
                "pvalue": 1.0, "lags": h, "passes": True}

    Q = 0.0
    lag_details = {}
    for k in range(1, h + 1):
        rk = np.sum(centered[k:] * centered[:-k]) / var
        Q += rk ** 2 / (n - k)
        lag_details[k] = float(rk)

    Q *= n * (n + 2)

    from scipy.stats import chi2 as chi2_dist
    pvalue = 1 - chi2_dist.cdf(Q, h)

    return {
        "test": "ljung_box",
        "statistic": float(Q),
        "pvalue": float(pvalue),
        "lags": h,
        "passes": pvalue > alpha,
        "autocorrelations": {str(k): v for k, v in
                             list(lag_details.items())[:5]},
    }


# ---------------------------------------------------------------------------
# 4.2 Wald-Wolfowitz runs test
# ---------------------------------------------------------------------------

def runs_test(samples, alpha=0.001):
    """
    Wald-Wolfowitz runs test.

    Tests whether the sequence of above/below-median values forms
    a random pattern. Too few runs = positive autocorrelation,
    too many = negative autocorrelation.
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    median = np.median(samples)

    above = samples > median
    n1 = int(np.sum(above))
    n2 = n - n1

    if n1 == 0 or n2 == 0:
        return {"test": "runs_test", "statistic": 0.0,
                "pvalue": 1.0, "passes": True}

    runs = 1 + int(np.sum(above[1:] != above[:-1]))

    E_R = 1 + 2 * n1 * n2 / (n1 + n2)
    Var_R = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
            ((n1 + n2) ** 2 * (n1 + n2 - 1))

    if Var_R <= 0:
        return {"test": "runs_test", "statistic": 0.0,
                "pvalue": 1.0, "passes": True}

    Z = (runs - E_R) / sqrt(Var_R)
    from scipy.stats import norm
    pvalue = 2 * (1 - norm.cdf(abs(Z)))

    return {
        "test": "runs_test",
        "statistic": float(Z),
        "pvalue": float(pvalue),
        "runs": runs,
        "expected_runs": float(E_R),
        "passes": pvalue > alpha,
    }


# ---------------------------------------------------------------------------
# 4.3 Block homogeneity (drift detection)
# ---------------------------------------------------------------------------

def block_homogeneity(mu, sigma, samples, n_blocks=10, tau=14,
                      alpha=0.001):
    """
    Split the stream into blocks and test distributional homogeneity.

    Chi-square homogeneity test across blocks x buckets. Catches
    mid-run drift or state corruption invisible to pooled tests.
    """
    samples = np.asarray(samples)
    n = len(samples)
    block_size = n // n_blocks
    if block_size < 50:
        return {"test": "block_homogeneity", "statistic": 0.0,
                "pvalue": 1.0, "passes": True,
                "note": "insufficient samples for block test"}

    support, probs = pdt_arrays(mu, sigma, tau)

    zmax = int(np.ceil(tau * sigma))
    lo = int(np.floor(mu)) - zmax
    nbins = min(10, len(support))
    bin_edges = np.linspace(lo, lo + len(support), nbins + 1)

    observed = np.zeros((n_blocks, nbins))
    for b in range(n_blocks):
        block = samples[b * block_size:(b + 1) * block_size]
        hist, _ = np.histogram(block, bins=bin_edges)
        observed[b] = hist

    col_totals = observed.sum(axis=0)
    row_totals = observed.sum(axis=1)
    grand_total = observed.sum()

    if grand_total == 0:
        return {"test": "block_homogeneity", "statistic": 0.0,
                "pvalue": 1.0, "passes": True}

    chi2_stat = 0.0
    for b in range(n_blocks):
        for j in range(nbins):
            expected = row_totals[b] * col_totals[j] / grand_total
            if expected > 0:
                chi2_stat += (observed[b][j] - expected) ** 2 / expected

    df = (n_blocks - 1) * (nbins - 1)
    if df <= 0:
        return {"test": "block_homogeneity", "statistic": 0.0,
                "pvalue": 1.0, "passes": True}

    from scipy.stats import chi2 as chi2_dist
    pvalue = 1 - chi2_dist.cdf(chi2_stat, df)

    return {
        "test": "block_homogeneity",
        "statistic": float(chi2_stat),
        "pvalue": float(pvalue),
        "df": df,
        "n_blocks": n_blocks,
        "passes": pvalue > alpha,
    }


# ---------------------------------------------------------------------------
# Extended battery runner
# ---------------------------------------------------------------------------

def run_extended_battery(mu, sigma, samples, tau=14, alpha=0.001,
                         mc_B=1000):
    """
    Run all extended univariate tests.

    Returns dict with results from each test and an overall verdict.
    """
    results = {}
    results["tail_exceedance"] = tail_exceedance(mu, sigma, samples,
                                                  tau=tau, alpha=alpha)
    results["sign_halfgaussian"] = sign_halfgaussian(mu, sigma, samples,
                                                      tau=tau, alpha=alpha)
    results["moment_cis"] = moment_cis(mu, sigma, samples)
    results["discrete_ad"] = discrete_anderson_darling(
        mu, sigma, samples, tau=tau, alpha=alpha, mc_B=mc_B)
    results["higher_criticism"] = higher_criticism(
        mu, sigma, samples, tau=tau, alpha=alpha, mc_B=mc_B)
    results["ljung_box"] = ljung_box(samples, alpha=alpha)
    results["runs_test"] = runs_test(samples, alpha=alpha)
    results["block_homogeneity"] = block_homogeneity(
        mu, sigma, samples, tau=tau, alpha=alpha)

    # Single family verdict via Fisher's method on the per-test p-values,
    # rather than AND-ing 7 tests each at alpha (which inflates the family
    # false-alarm rate to ~1-(1-alpha)^7 ~ 0.7% at alpha=1e-3). One global
    # p keeps the family-wise rate at alpha. Individual results are still
    # reported above (report-first), so localized flaws remain visible.
    component_pvalues = {
        "tail_exceedance": min(
            (t["pvalue"] for t in results["tail_exceedance"]["thresholds"].values()),
            default=1.0),
        "sign_halfgaussian": results["sign_halfgaussian"]["magnitude_chi2"]["pvalue"],
        "discrete_ad": results["discrete_ad"]["pvalue"],
        "higher_criticism": results["higher_criticism"]["pvalue"],
        "ljung_box": results["ljung_box"]["pvalue"],
        "runs_test": results["runs_test"]["pvalue"],
        "block_homogeneity": results["block_homogeneity"]["pvalue"],
    }
    pvals = np.clip(np.array(list(component_pvalues.values()), dtype=float),
                    1e-300, 1.0)
    from scipy.stats import chi2 as _chi2
    fisher_stat = float(-2 * np.sum(np.log(pvals)))
    global_pvalue = float(_chi2.sf(fisher_stat, 2 * len(pvals)))

    results["component_pvalues"] = {k: float(v)
                                    for k, v in component_pvalues.items()}
    results["fisher_stat"] = fisher_stat
    results["global_pvalue"] = global_pvalue
    # Also expose which individual tests reject at a Bonferroni-corrected
    # threshold, so a single strong localized flaw is never masked.
    k = len(pvals)
    results["bonferroni_rejects"] = [
        name for name, p in component_pvalues.items() if p <= alpha / k]
    results["all_pass"] = (global_pvalue > alpha
                           and not results["bonferroni_rejects"])

    return results
