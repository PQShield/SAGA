"""
SAGA test vector generator — known-good, known-bad, and borderline distributions.

Run BEFORE any improvements to establish a baseline, then after each phase
to confirm that new/fixed tests catch what they should.

Usage:
    python generate_test_vectors.py [--outdir test_vectors] [--n 10000] [--seed 2026]

Each vector is a JSON file containing:
    {
        "label": "good_centered" | "bad_wrong_sigma" | ...,
        "tier": "good" | "bad" | "mediocre",
        "description": "...",
        "params": {"mu": ..., "sigma": ..., "n": ...},
        "flaw": {"type": "none" | "sigma_scale" | ..., ...},
        "expected_detection": {"chi2": true/false, "AD": ..., ...},
        "samples": [...]
    }
"""
import argparse
import json
import os
import sys
from math import ceil, exp, floor, sqrt
import numpy as np


# ---------------------------------------------------------------------------
# PDT construction (mirrors saga.py's make_gaussian_pdt)
# ---------------------------------------------------------------------------

def gaussian(x, mu, sigma):
    return exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def make_pdt(mu, sigma, tau=14):
    """Return (support, probabilities) as aligned numpy arrays."""
    zmax = ceil(tau * sigma)
    lo = int(floor(mu)) - zmax
    hi = int(ceil(mu)) + zmax
    support = list(range(lo, hi))
    raw = np.array([gaussian(z, mu, sigma) for z in support])
    probs = raw / raw.sum()
    return np.array(support), probs


# ---------------------------------------------------------------------------
# Samplers: one per flaw type
# ---------------------------------------------------------------------------

def sample_perfect(rng, support, probs, n):
    """Draw n iid samples from the exact discrete Gaussian PDT."""
    return rng.choice(support, size=n, p=probs).tolist()


def sample_wrong_sigma(rng, mu, true_sigma, factor, n, tau=14):
    """Sample from a Gaussian with sigma scaled by `factor`."""
    s, p = make_pdt(mu, true_sigma * factor, tau)
    return rng.choice(s, size=n, p=p).tolist()


def sample_wrong_mu(rng, mu, sigma, shift, n, tau=14):
    """Sample from a Gaussian with mu shifted by `shift`."""
    s, p = make_pdt(mu + shift, sigma, tau)
    return rng.choice(s, size=n, p=p).tolist()


def sample_truncated_tails(rng, support, probs, n, tail_sigma, mu, sigma):
    """Zero out probability beyond tail_sigma * sigma from mu."""
    mask = np.abs(support - mu) <= tail_sigma * sigma
    p = probs * mask
    if p.sum() == 0:
        raise ValueError("Truncation killed all mass")
    p = p / p.sum()
    return rng.choice(support, size=n, p=p).tolist()


def sample_biased_sign(rng, support, probs, n, bias, mu):
    """
    Bias the sign: P(z > mu) is inflated by `bias` (e.g. 0.55 = 55/45 split).
    Redistributes mass while keeping |z - mu| distribution unchanged.
    """
    p = probs.copy()
    center_idx = np.argmin(np.abs(support - mu))
    for i, z in enumerate(support):
        if z > mu:
            p[i] *= bias / 0.5
        elif z < mu:
            p[i] *= (1 - bias) / 0.5
    p = p / p.sum()
    return rng.choice(support, size=n, p=p).tolist()


def sample_single_table_error(rng, support, probs, n, error_idx, error_factor):
    """
    Multiply one entry in the PDT by error_factor, renormalize.
    Simulates a single wrong CDT/RCDT entry.
    """
    p = probs.copy()
    idx = error_idx % len(p)
    p[idx] *= error_factor
    p = p / p.sum()
    return rng.choice(support, size=n, p=p).tolist()


def sample_quantized_pdt(rng, support, probs, n, bits):
    """
    Quantize PDT to `bits` bits of precision (simulates low-precision table).
    """
    scale = 2 ** bits
    q = np.round(probs * scale) / scale
    q = np.maximum(q, 1.0 / scale)  # no zeros
    q = q / q.sum()
    return rng.choice(support, size=n, p=q).tolist()


def sample_contaminated(rng, support, probs, n, epsilon):
    """
    (1 - epsilon) from the true distribution + epsilon from uniform on support.
    """
    p = (1 - epsilon) * probs + epsilon * np.ones_like(probs) / len(probs)
    p = p / p.sum()
    return rng.choice(support, size=n, p=p).tolist()


def sample_correlated_markov(rng, support, probs, n, rho):
    """
    Lag-1 Markov chain: with probability rho, repeat previous sample;
    with probability (1 - rho), draw fresh from PDT. Marginal is exact
    but samples are serially correlated.
    """
    samples = [rng.choice(support, p=probs)]
    for _ in range(n - 1):
        if rng.random() < rho:
            samples.append(samples[-1])
        else:
            samples.append(rng.choice(support, p=probs))
    return [int(x) for x in samples]


def sample_uniform(rng, support, n):
    """Uniform on the support — maximally wrong shape."""
    return rng.choice(support, size=n).tolist()


def sample_drift(rng, support, probs, n, mu, sigma, drift_rate, tau=14):
    """
    Sigma drifts linearly over the stream: sigma(t) = sigma * (1 + drift_rate * t/n).
    Marginal pooled distribution looks roughly correct; block homogeneity catches it.
    """
    samples = []
    for i in range(n):
        s_t = sigma * (1 + drift_rate * i / n)
        s_i, p_i = make_pdt(mu, s_t, tau)
        samples.append(int(rng.choice(s_i, p=p_i)))
    return samples


# ---------------------------------------------------------------------------
# Multivariate vectors
# ---------------------------------------------------------------------------

def multivariate_perfect(rng, dim, sigma, n):
    """iid N(0, sigma^2) per coordinate."""
    return rng.normal(0, sigma, size=(n, dim)).tolist()


def multivariate_inflated_correlation(rng, dim, sigma, n, rho_pair, pair):
    """
    Perfect marginals, but coordinates pair[0] and pair[1] have
    correlation rho_pair. All others independent.
    """
    data = rng.normal(0, sigma, size=(n, dim))
    i, j = pair
    z = rng.normal(0, sigma, size=n)
    data[:, i] = rho_pair * z + sqrt(1 - rho_pair**2) * data[:, i]
    data[:, j] = rho_pair * z + sqrt(1 - rho_pair**2) * data[:, j]
    return data.tolist()


def multivariate_wrong_variance_one_coord(rng, dim, sigma, n, coord, factor):
    """One coordinate has variance scaled by factor^2."""
    data = rng.normal(0, sigma, size=(n, dim))
    data[:, coord] *= factor
    return data.tolist()


def multivariate_fft_tree_zeroed(rng, dim, sigma, n, zero_freq):
    """
    Simulate FFT-tree zeroing: one frequency bin has zero variance.
    Generate in FFT domain, zero one bin, inverse FFT.
    """
    half = dim // 2
    data = []
    for _ in range(n):
        coeffs = rng.normal(0, sigma, size=half) + 1j * rng.normal(0, sigma, size=half)
        coeffs[zero_freq] = 0  # zeroed node
        sig = np.fft.irfft(coeffs, n=half)
        sig2 = rng.normal(0, sigma, size=half)
        data.append(np.concatenate([sig, sig2]).tolist())
    return data


# ---------------------------------------------------------------------------
# Vector definitions
# ---------------------------------------------------------------------------

def build_univariate_vectors(n, seed):
    """Return list of (metadata_dict, samples) for all univariate test vectors."""
    vectors = []
    # Canonical parameters
    params_list = [
        (0.0, 1.55, "centered_typical"),
        (0.0, 1.3, "centered_narrow"),
        (0.0, 1.8, "centered_wide"),
        (0.5, 1.55, "half_center"),         # hostile fractional mu
        (0.25, 1.55, "quarter_center"),      # hostile fractional mu
        (3.7, 1.55, "nonzero_center"),
    ]

    for mu, sigma, tag in params_list:
        rng = np.random.default_rng(seed)
        support, probs = make_pdt(mu, sigma)

        # ---- GOOD ----
        vectors.append({
            "label": f"good_{tag}",
            "tier": "good",
            "description": f"Perfect discrete Gaussian, mu={mu}, sigma={sigma}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "none"},
            "expected_detection": {
                "chi2": False, "AD": False, "HC": False,
                "sign_test": False, "ljung_box": False,
                "tail_exceedance": False, "norm_test": False
            },
            "samples": sample_perfect(rng, support, probs, n)
        })

        # ---- BAD ----

        # Wrong sigma (10% too wide)
        rng = np.random.default_rng(seed + 1)
        vectors.append({
            "label": f"bad_sigma_1.1x_{tag}",
            "tier": "bad",
            "description": f"Sigma inflated by 10%, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "sigma_scale", "factor": 1.1},
            "expected_detection": {
                "chi2": True, "AD": True, "HC": False,
                "sign_test": False, "ljung_box": False,
                "tail_exceedance": True, "norm_test": True
            },
            "samples": sample_wrong_sigma(rng, mu, sigma, 1.1, n)
        })

        # Wrong mu (shifted by 0.5)
        rng = np.random.default_rng(seed + 2)
        vectors.append({
            "label": f"bad_mu_shift_0.5_{tag}",
            "tier": "bad",
            "description": f"Mu shifted by 0.5, true mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "mu_shift", "shift": 0.5},
            "expected_detection": {
                "chi2": True, "AD": True, "HC": False,
                "sign_test": True, "ljung_box": False,
                "tail_exceedance": False, "norm_test": False
            },
            "samples": sample_wrong_mu(rng, mu, sigma, 0.5, n)
        })

        # Biased sign (60/40)
        rng = np.random.default_rng(seed + 3)
        vectors.append({
            "label": f"bad_sign_bias_60_{tag}",
            "tier": "bad",
            "description": f"Sign bias 60/40, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "sign_bias", "bias": 0.6},
            "expected_detection": {
                "chi2": True, "AD": True, "HC": False,
                "sign_test": True, "ljung_box": False,
                "tail_exceedance": False, "norm_test": False
            },
            "samples": sample_biased_sign(rng, support, probs, n, 0.6, mu)
        })

        # Uniform on support
        rng = np.random.default_rng(seed + 4)
        vectors.append({
            "label": f"bad_uniform_{tag}",
            "tier": "bad",
            "description": f"Uniform distribution on support, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "uniform"},
            "expected_detection": {
                "chi2": True, "AD": True, "HC": True,
                "sign_test": True, "ljung_box": False,
                "tail_exceedance": True, "norm_test": True
            },
            "samples": sample_uniform(rng, support, n)
        })

        # Single table entry wrong (entry near mode, 3x inflated)
        rng = np.random.default_rng(seed + 5)
        mode_idx = np.argmax(probs)
        vectors.append({
            "label": f"bad_table_error_{tag}",
            "tier": "bad",
            "description": f"Single PDT entry (near mode) inflated 3x, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "table_error", "index": int(mode_idx + 2), "factor": 3.0},
            "expected_detection": {
                "chi2": False, "AD": False, "HC": True,
                "sign_test": False, "ljung_box": False,
                "tail_exceedance": False, "norm_test": False
            },
            "samples": sample_single_table_error(rng, support, probs, n,
                                                  int(mode_idx + 2), 3.0)
        })

        # Truncated at 3-sigma (detectable: removes ~30 expected samples)
        rng = np.random.default_rng(seed + 6)
        vectors.append({
            "label": f"bad_truncated_3sig_{tag}",
            "tier": "bad",
            "description": f"Tails truncated at 3*sigma, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "tail_truncation", "tail_sigma": 3},
            "expected_detection": {
                "chi2": False, "AD": True, "HC": False,
                "sign_test": False, "ljung_box": False,
                "tail_exceedance": True, "norm_test": False
            },
            "samples": sample_truncated_tails(rng, support, probs, n, 3, mu, sigma)
        })

        # Correlated (Markov, rho=0.3)
        rng = np.random.default_rng(seed + 7)
        vectors.append({
            "label": f"bad_markov_0.3_{tag}",
            "tier": "bad",
            "description": f"Lag-1 Markov (rho=0.3), perfect marginal, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "markov", "rho": 0.3},
            "expected_detection": {
                "chi2": False, "AD": False, "HC": False,
                "sign_test": False, "ljung_box": True,
                "tail_exceedance": False, "norm_test": False
            },
            "samples": sample_correlated_markov(rng, support, probs, n, 0.3)
        })

        # ---- MEDIOCRE (borderline) ----

        # Sigma off by 1%
        rng = np.random.default_rng(seed + 10)
        vectors.append({
            "label": f"med_sigma_1.01x_{tag}",
            "tier": "mediocre",
            "description": f"Sigma inflated by 1%, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "sigma_scale", "factor": 1.01},
            "expected_detection": {
                "chi2": "maybe", "AD": "maybe", "HC": False,
                "sign_test": False, "ljung_box": False,
                "tail_exceedance": "maybe", "norm_test": "maybe"
            },
            "samples": sample_wrong_sigma(rng, mu, sigma, 1.01, n)
        })

        # Slight sign bias (52/48)
        rng = np.random.default_rng(seed + 11)
        vectors.append({
            "label": f"med_sign_bias_52_{tag}",
            "tier": "mediocre",
            "description": f"Sign bias 52/48, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "sign_bias", "bias": 0.52},
            "expected_detection": {
                "chi2": "maybe", "AD": "maybe", "HC": False,
                "sign_test": "maybe", "ljung_box": False,
                "tail_exceedance": False, "norm_test": False
            },
            "samples": sample_biased_sign(rng, support, probs, n, 0.52, mu)
        })

        # 8-bit quantized PDT
        rng = np.random.default_rng(seed + 12)
        vectors.append({
            "label": f"med_quantized_8bit_{tag}",
            "tier": "mediocre",
            "description": f"PDT quantized to 8 bits, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "quantized_pdt", "bits": 8},
            "expected_detection": {
                "chi2": "maybe", "AD": "maybe", "HC": True,
                "sign_test": False, "ljung_box": False,
                "tail_exceedance": False, "norm_test": False
            },
            "samples": sample_quantized_pdt(rng, support, probs, n, 8)
        })

        # 1% contamination with uniform
        rng = np.random.default_rng(seed + 13)
        vectors.append({
            "label": f"med_contaminated_1pct_{tag}",
            "tier": "mediocre",
            "description": f"1% uniform contamination, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "contamination", "epsilon": 0.01},
            "expected_detection": {
                "chi2": "maybe", "AD": "maybe", "HC": "maybe",
                "sign_test": False, "ljung_box": False,
                "tail_exceedance": "maybe", "norm_test": False
            },
            "samples": sample_contaminated(rng, support, probs, n, 0.01)
        })

        # Drift (sigma grows 5% over the stream)
        rng = np.random.default_rng(seed + 14)
        vectors.append({
            "label": f"med_drift_5pct_{tag}",
            "tier": "mediocre",
            "description": f"Sigma drifts +5% over stream, mu={mu}",
            "params": {"mu": mu, "sigma": sigma, "n": n},
            "flaw": {"type": "drift", "drift_rate": 0.05},
            "expected_detection": {
                "chi2": "maybe", "AD": "maybe", "HC": False,
                "sign_test": False, "ljung_box": False,
                "tail_exceedance": "maybe", "norm_test": False
            },
            "samples": sample_drift(rng, support, probs, n, mu, sigma, 0.05)
        })

    return vectors


def build_multivariate_vectors(n, seed):
    """Return list of (metadata_dict, samples) for multivariate test vectors."""
    vectors = []
    dim = 128  # Falcon-64 signature dimension
    sigma = 1.55

    # Good: iid Gaussian
    rng = np.random.default_rng(seed + 100)
    vectors.append({
        "label": "mv_good_iid",
        "tier": "good",
        "description": f"iid N(0, {sigma}^2), dim={dim}",
        "params": {"sigma": sigma, "dim": dim, "n": n},
        "flaw": {"type": "none"},
        "samples": multivariate_perfect(rng, dim, sigma, n)
    })

    # Bad: one pair of coordinates correlated (rho=0.5)
    rng = np.random.default_rng(seed + 101)
    vectors.append({
        "label": "mv_bad_corr_pair",
        "tier": "bad",
        "description": f"Coords 0,1 correlated (rho=0.5), dim={dim}",
        "params": {"sigma": sigma, "dim": dim, "n": n},
        "flaw": {"type": "correlated_pair", "pair": [0, 1], "rho": 0.5},
        "samples": multivariate_inflated_correlation(rng, dim, sigma, n, 0.5, (0, 1))
    })

    # Bad: one coordinate has wrong variance
    rng = np.random.default_rng(seed + 102)
    vectors.append({
        "label": "mv_bad_variance_one_coord",
        "tier": "bad",
        "description": f"Coord 7 has 1.3x variance, dim={dim}",
        "params": {"sigma": sigma, "dim": dim, "n": n},
        "flaw": {"type": "wrong_variance", "coord": 7, "factor": 1.3},
        "samples": multivariate_wrong_variance_one_coord(rng, dim, sigma, n, 7, 1.3)
    })

    # Bad: FFT-tree zeroed frequency
    rng = np.random.default_rng(seed + 103)
    vectors.append({
        "label": "mv_bad_fft_zeroed",
        "tier": "bad",
        "description": f"FFT freq 3 zeroed (simulates tree bug), dim={dim}",
        "params": {"sigma": sigma, "dim": dim, "n": n},
        "flaw": {"type": "fft_zeroed", "freq": 3},
        "samples": multivariate_fft_tree_zeroed(rng, dim, sigma, n, 3)
    })

    # Bad: inflated norm (all coordinates positively correlated)
    rng = np.random.default_rng(seed + 104)
    data_norm = rng.normal(0, sigma, size=(n, dim))
    shared = rng.normal(0, sigma * 0.3, size=n)
    data_norm += shared[:, np.newaxis]
    vectors.append({
        "label": "mv_bad_norm_inflated",
        "tier": "bad",
        "description": f"All coords share a common factor (norm inflated), dim={dim}",
        "params": {"sigma": sigma, "dim": dim, "n": n},
        "flaw": {"type": "norm_inflation", "shared_sigma_fraction": 0.3},
        "samples": data_norm.tolist()
    })

    # Bad: two different distributions (simulates key-dependent output)
    rng = np.random.default_rng(seed + 105)
    data_key1 = rng.normal(0, sigma, size=(n // 2, dim))
    data_key2 = rng.normal(0, sigma * 1.05, size=(n // 2, dim))
    vectors.append({
        "label": "mv_bad_cross_key",
        "tier": "bad",
        "description": f"Mixed: half from sigma, half from sigma*1.05, dim={dim}",
        "params": {"sigma": sigma, "dim": dim, "n": n},
        "flaw": {"type": "cross_key", "sigma_ratio": 1.05},
        "samples": np.vstack([data_key1, data_key2]).tolist(),
        "samples_key1": data_key1.tolist(),
        "samples_key2": data_key2.tolist(),
    })

    # Mediocre: weak single-pair correlation (rho=0.1)
    rng = np.random.default_rng(seed + 106)
    vectors.append({
        "label": "mv_med_weak_corr",
        "tier": "mediocre",
        "description": f"Coords 0,1 weakly correlated (rho=0.1), dim={dim}",
        "params": {"sigma": sigma, "dim": dim, "n": n},
        "flaw": {"type": "correlated_pair", "pair": [0, 1], "rho": 0.1},
        "samples": multivariate_inflated_correlation(rng, dim, sigma, n, 0.1, (0, 1))
    })

    # Mediocre: FFT frequency with reduced (not zeroed) variance
    rng = np.random.default_rng(seed + 107)
    half = dim // 2
    data_fft_weak = []
    for _ in range(n):
        coeffs = rng.normal(0, sigma, size=half) + 1j * rng.normal(0, sigma, size=half)
        coeffs[5] *= 0.7  # 70% of normal variance at freq 5
        sig_half = np.fft.irfft(coeffs, n=half)
        sig2 = rng.normal(0, sigma, size=half)
        data_fft_weak.append(np.concatenate([sig_half, sig2]).tolist())
    vectors.append({
        "label": "mv_med_fft_weak",
        "tier": "mediocre",
        "description": f"FFT freq 5 at 70% variance, dim={dim}",
        "params": {"sigma": sigma, "dim": dim, "n": n},
        "flaw": {"type": "fft_weak", "freq": 5, "factor": 0.7},
        "samples": data_fft_weak,
    })

    return vectors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate SAGA test vectors")
    parser.add_argument("--outdir", default="test_vectors", help="Output directory")
    parser.add_argument("--n", type=int, default=10000, help="Samples per vector")
    parser.add_argument("--seed", type=int, default=2026, help="Base RNG seed")
    parser.add_argument("--no-multivariate", action="store_true",
                        help="Skip multivariate vectors (faster)")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, "univariate"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "multivariate"), exist_ok=True)

    print(f"Generating univariate vectors (n={args.n}, seed={args.seed})...")
    uni_vectors = build_univariate_vectors(args.n, args.seed)
    for v in uni_vectors:
        path = os.path.join(args.outdir, "univariate", f"{v['label']}.json")
        with open(path, 'w') as f:
            json.dump(v, f, indent=2)
    print(f"  wrote {len(uni_vectors)} univariate vectors")

    if not args.no_multivariate:
        mv_n = min(args.n, 2000)  # multivariate is expensive
        print(f"Generating multivariate vectors (n={mv_n}, seed={args.seed})...")
        mv_vectors = build_multivariate_vectors(mv_n, args.seed)
        for v in mv_vectors:
            path = os.path.join(args.outdir, "multivariate", f"{v['label']}.json")
            with open(path, 'w') as f:
                json.dump(v, f, indent=2)
        print(f"  wrote {len(mv_vectors)} multivariate vectors")

    # Summary
    print("\n--- Vector summary ---")
    all_v = uni_vectors + (mv_vectors if not args.no_multivariate else [])
    for tier in ["good", "bad", "mediocre"]:
        count = sum(1 for v in all_v if v["tier"] == tier)
        print(f"  {tier:>10}: {count}")
    print(f"  {'total':>10}: {len(all_v)}")
    print(f"\nOutput: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
