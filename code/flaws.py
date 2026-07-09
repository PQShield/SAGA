"""
Univariate flaw generators for calibration and power-matrix studies.

Each gen_* draws n integer samples from a *flawed* discrete Gaussian, used to
measure a test's detection power. Single source of truth: these were previously
copy-pasted -- and had already drifted (gen_table_error's bounds guard existed
in only one copy) -- between calibration.py and
scripts/run_extensive_calibration.py.
"""
import numpy as np

from distributions import pdt_arrays


def gen_perfect(rng, mu, sigma, n):
    s, p = pdt_arrays(mu, sigma)
    return rng.choice(s, size=n, p=p).tolist()


def gen_sigma_shift(rng, mu, sigma, n, epsilon):
    s, p = pdt_arrays(mu, sigma * (1 + epsilon))
    return rng.choice(s, size=n, p=p).tolist()


def gen_mu_shift(rng, mu, sigma, n, delta):
    s, p = pdt_arrays(mu + delta, sigma)
    return rng.choice(s, size=n, p=p).tolist()


def gen_sign_coupling(rng, mu, sigma, n, bias):
    s, p = pdt_arrays(mu, sigma)
    p_biased = p.copy()
    for i, z in enumerate(s):
        if z > mu:
            p_biased[i] *= bias / 0.5
        elif z < mu:
            p_biased[i] *= (1 - bias) / 0.5
    p_biased /= p_biased.sum()
    return rng.choice(s, size=n, p=p_biased).tolist()


def gen_tail_truncation(rng, mu, sigma, n, k_sigma):
    s, p = pdt_arrays(mu, sigma)
    mask = np.abs(s - mu) <= k_sigma * sigma
    p_trunc = p * mask
    p_trunc /= p_trunc.sum()
    return rng.choice(s, size=n, p=p_trunc).tolist()


def gen_table_error(rng, mu, sigma, n, factor):
    s, p = pdt_arrays(mu, sigma)
    mode_idx = np.argmax(p)
    # Bounds guard: mode_idx + 2 can run past the support edge for small or
    # asymmetric supports. (This guard existed in only one of the two former
    # copies -- the reason the duplication was worth removing.)
    idx = min(mode_idx + 2, len(p) - 1)
    p_err = p.copy()
    p_err[idx] *= factor
    p_err /= p_err.sum()
    return rng.choice(s, size=n, p=p_err).tolist()


def gen_contamination(rng, mu, sigma, n, epsilon):
    s, p = pdt_arrays(mu, sigma)
    p_cont = (1 - epsilon) * p + epsilon / len(p)
    p_cont /= p_cont.sum()
    return rng.choice(s, size=n, p=p_cont).tolist()


def gen_markov(rng, mu, sigma, n, rho):
    s, p = pdt_arrays(mu, sigma)
    samples = [rng.choice(s, p=p)]
    for _ in range(n - 1):
        if rng.random() < rho:
            samples.append(samples[-1])
        else:
            samples.append(rng.choice(s, p=p))
    return [int(x) for x in samples]


def gen_bimodal(rng, mu, sigma, n, separation):
    """Mix two shifted Gaussians -- creates excess kurtosis."""
    s1, p1 = pdt_arrays(mu - separation, sigma * 0.8)
    s2, p2 = pdt_arrays(mu + separation, sigma * 0.8)
    half = n // 2
    a = rng.choice(s1, size=half, p=p1).tolist()
    b = rng.choice(s2, size=n - half, p=p2).tolist()
    combined = a + b
    rng.shuffle(combined)
    return [int(x) for x in combined]


def gen_periodic(rng, mu, sigma, n, period):
    """Periodic pattern in the sample stream."""
    s, p = pdt_arrays(mu, sigma)
    samples = rng.choice(s, size=n, p=p).tolist()
    for i in range(0, n, period):
        samples[i] = int(round(mu))
    return samples
