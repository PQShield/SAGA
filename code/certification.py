"""
Analytic Renyi divergence certification for discrete Gaussian samplers.

Computes R_a(P_spec || P_ideal) exactly from the sampler specification
(RCDT tables, BerExp quantization), without sampling. This is the
mechanism that actually certifies proof-level security properties
(R_a - 1 <= 2^{-75}) — statistical testing cannot reach this resolution.

References:
    [BLLSS18] Bai et al. "Improved Security Proofs in Lattice-Based
              Cryptography: Using the Renyi Divergence." J. Cryptology 2018.
    [Pre17]   Prest. "Sharper Bounds in Lattice-Based Cryptography Using
              the Renyi Divergence." ASIACRYPT 2017.
    [HPRR20]  Howe, Prest, Ricosset, Rossi. "Isochronous Gaussian
              Sampling." PQCrypto 2020.

Usage:
    from certification import certify_base_sampler, certify_full_sampler

    result = certify_base_sampler()
    print(result)

    result = certify_full_sampler(sigma=1.55, mu=0.0)
    print(result)
"""

try:
    import mpmath
except ImportError:
    raise ImportError("mpmath is required for certification: pip install mpmath")

from sampler import halfgaussian_pdt, sigma0, sigmin


PRECISION = 400
N_ENTRIES = 19
CDT_BITS = 72
BEREXP_BITS = 64
TAU = 14

# Security level targets: order-lambda Renyi divergence
# Falcon-512: a=128, Falcon-1024: a=256
# Requirement: R_a - 1 <= 2^{-75}
SECURITY_TARGETS = {
    "falcon-512": {"order": 128, "max_ra_minus_1": mpmath.power(2, -75)},
    "falcon-1024": {"order": 256, "max_ra_minus_1": mpmath.power(2, -75)},
}


def _set_precision():
    mpmath.mp.prec = PRECISION


def compute_base_sampler_distribution():
    """
    Exact distribution of sampler0() from the RCDT/PDT table.

    Returns:
        (support, probs): lists where support[i] = i, probs[i] = PDT[i] / 2^72
    """
    _set_precision()
    scale = mpmath.power(2, CDT_BITS)
    support = list(range(N_ENTRIES))
    probs = [mpmath.mpf(pdt_entry) / scale for pdt_entry in halfgaussian_pdt]
    return support, probs


def compute_ideal_halfgaussian():
    """
    Ideal half-Gaussian distribution with sigma = sigma0.

    Returns:
        (support, probs): exact probabilities computed at high precision
    """
    _set_precision()
    s0 = mpmath.mpf(str(sigma0))
    two_s2 = 2 * s0 ** 2

    raw = [mpmath.exp(-(z ** 2) / two_s2) for z in range(N_ENTRIES)]
    total = sum(raw)
    probs = [r / total for r in raw]
    return list(range(N_ENTRIES)), probs


def renyi_divergence(P, Q, a):
    """
    Compute R_a(P || Q) for finite discrete distributions.

    Args:
        P, Q: lists of probabilities (same length, same support)
        a: order (int or mpf, > 1; use 'inf' for max-divergence)

    Returns:
        R_a as an mpmath.mpf value
    """
    _set_precision()
    assert len(P) == len(Q)

    if a == 'inf' or a == float('inf'):
        ratios = [p / q for p, q in zip(P, Q) if q > 0]
        return mpmath.log(max(ratios))

    a = mpmath.mpf(a)
    total = sum(p ** a / q ** (a - 1) for p, q in zip(P, Q) if q > 0)
    return mpmath.log(total) / (a - 1)


def total_variation(P, Q):
    """TV(P, Q) = 0.5 * sum |P(x) - Q(x)|"""
    _set_precision()
    return sum(abs(p - q) for p, q in zip(P, Q)) / 2


def certify_base_sampler(a_orders=None):
    """
    Certify the base sampler (sampler0) against the ideal half-Gaussian.

    Returns dict with R_a values and pass/fail against security targets.
    """
    if a_orders is None:
        a_orders = [2, 128, 256, 'inf']

    _, P = compute_base_sampler_distribution()
    _, Q = compute_ideal_halfgaussian()

    results = {"type": "base_sampler", "sigma0": sigma0}
    rd_results = {}

    for a in a_orders:
        label = f"R_{a}" if a != 'inf' else "R_inf"
        ra = renyi_divergence(P, Q, a)
        rd_results[label] = float(ra)
        if a != 'inf' and a != 2:
            # renyi_divergence returns log(R_a) in the [Pre17] ratio
            # convention, so R_a - 1 = exp(ra) - 1. (exp(ra*(a-1)) - 1
            # would be Sum p^a/q^(a-1) - 1 ~ (a-1)(R_a - 1), a factor
            # a-1 too large vs the requirement's convention.)
            rd_results[f"{label}_minus_1"] = float(mpmath.exp(ra) - 1)

    results["renyi_divergences"] = rd_results
    results["total_variation"] = float(total_variation(P, Q))

    results["security_check"] = {}
    results["security_check"]["note"] = (
        "Base sampler divergence is informational only. "
        "The full sampler (with BerExp rejection) is what the security proofs certify."
    )
    for target_name, target in SECURITY_TARGETS.items():
        a = target["order"]
        ra = renyi_divergence(P, Q, a)
        ra_minus_1 = mpmath.exp(ra) - 1
        results["security_check"][target_name] = {
            "order": a,
            "R_a": float(ra),
            "R_a_minus_1": float(ra_minus_1),
            "requirement": f"R_a - 1 <= 2^{{-75}} = {float(target['max_ra_minus_1']):.2e}",
        }

    return results


def compute_full_sampler_distribution(mu, sigma, tau=TAU):
    """
    Exact distribution of samplerz(mu, sigma) from the spec.

    Enumerates all (z0, b) pairs, computes BerExp acceptance probability
    at full precision, and normalizes.

    Args:
        mu: center (float or mpmath.mpf)
        sigma: standard deviation
        tau: tail cutoff

    Returns:
        (support, P_spec, Q_ideal): integer support, spec distribution,
        and ideal discrete Gaussian distribution
    """
    _set_precision()
    mu = mpmath.mpf(str(mu))
    sigma = mpmath.mpf(str(sigma))
    s0 = mpmath.mpf(str(sigma0))
    smin = mpmath.mpf(str(sigmin))
    sf = smin / sigma

    c0 = mu - mpmath.floor(mu)
    floor_mu = int(mpmath.floor(mu))

    _, P_base = compute_base_sampler_distribution()

    zmax = int(mpmath.ceil(tau * sigma))
    support = list(range(floor_mu - zmax, floor_mu + zmax))

    berexp_scale = mpmath.power(2, BEREXP_BITS)

    P_unnorm = {}
    for z_abs in support:
        z_out = z_abs - floor_mu
        mass = mpmath.mpf(0)

        for z0 in range(N_ENTRIES):
            for b in [0, 1]:
                z_candidate = ((b << 1) - 1) * z0 + b
                if z_candidate != z_out:
                    continue

                x = (z_out - c0) ** 2 / (2 * sigma ** 2) - z0 ** 2 / (2 * s0 ** 2)

                if x < 0:
                    accept_p = mpmath.mpf(1)
                else:
                    exp_neg_x = mpmath.exp(-x)
                    p_berexp = mpmath.floor(exp_neg_x * sf * berexp_scale - 1)
                    if p_berexp < 0:
                        accept_p = mpmath.mpf(0)
                    else:
                        accept_p = p_berexp / berexp_scale

                mass += P_base[z0] * mpmath.mpf(1) / 2 * accept_p

        P_unnorm[z_abs] = mass

    Z_total = sum(P_unnorm.values())
    P_spec = [P_unnorm.get(z, mpmath.mpf(0)) / Z_total for z in support]

    two_sigma2 = 2 * sigma ** 2
    raw_ideal = [mpmath.exp(-(z - mu) ** 2 / two_sigma2) for z in support]
    Z_ideal = sum(raw_ideal)
    Q_ideal = [r / Z_ideal for r in raw_ideal]

    return support, P_spec, Q_ideal


def certify_full_sampler(sigma, mu, a_orders=None, tau=TAU):
    """
    Certify the full sampler at a specific (mu, sigma).

    Returns dict with R_a values and pass/fail.
    """
    if a_orders is None:
        a_orders = [2, 128, 256, 'inf']

    support, P_spec, Q_ideal = compute_full_sampler_distribution(mu, sigma, tau)

    results = {
        "type": "full_sampler",
        "mu": float(mu),
        "sigma": float(sigma),
        "support_size": len(support),
    }

    rd_results = {}
    for a in a_orders:
        label = f"R_{a}" if a != 'inf' else "R_inf"
        ra = renyi_divergence(P_spec, Q_ideal, a)
        rd_results[label] = float(ra)

    results["renyi_divergences"] = rd_results
    results["total_variation"] = float(total_variation(P_spec, Q_ideal))

    results["security_check"] = {}
    for target_name, target in SECURITY_TARGETS.items():
        a = target["order"]
        ra = renyi_divergence(P_spec, Q_ideal, a)
        ra_minus_1 = mpmath.exp(ra) - 1
        passes = ra_minus_1 <= target["max_ra_minus_1"]
        results["security_check"][target_name] = {
            "order": a,
            "R_a_minus_1": float(ra_minus_1),
            "passes": bool(passes),
        }

    return results


if __name__ == "__main__":
    import json
    print("=== Base sampler certification ===")
    base = certify_base_sampler()
    print(json.dumps(base, indent=2))

    print("\n=== Full sampler certification (mu=0.0, sigma=1.55) ===")
    full = certify_full_sampler(sigma=1.55, mu=0.0)
    print(json.dumps(full, indent=2))

    print("\n=== Full sampler certification (mu=0.5, sigma=1.55) ===")
    full2 = certify_full_sampler(sigma=1.55, mu=0.5)
    print(json.dumps(full2, indent=2))
