"""
SAGA Report Card — terminal output with ANSI colors.

Usage:
    python report_card.py [--samples FILE] [--mu 0] [--sigma 1.55] [--n 10000]

Generates samples if no file given, runs all tests, prints the card.
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from saga import UnivariateSamples, make_gaussian_pdt
from univariate_tests import (
    tail_exceedance, sign_halfgaussian, discrete_anderson_darling,
    higher_criticism, ljung_box, runs_test, block_homogeneity,
)


# ---------------------------------------------------------------------------
# ANSI escape codes
# ---------------------------------------------------------------------------

class C:
    RST       = '\033[0m'
    BOLD      = '\033[1m'
    DIM       = '\033[2m'
    ITALIC    = '\033[3m'
    UNDER     = '\033[4m'
    # Foreground
    RED       = '\033[38;5;196m'
    GREEN     = '\033[38;5;78m'
    YELLOW    = '\033[38;5;220m'
    BLUE      = '\033[38;5;75m'
    CYAN      = '\033[38;5;116m'
    GREY      = '\033[38;5;245m'
    WHITE     = '\033[38;5;255m'
    ORANGE    = '\033[38;5;214m'
    PINK      = '\033[38;5;205m'
    # Background
    BG_DARK   = '\033[48;5;234m'
    BG_RED    = '\033[48;5;52m'
    BG_GREEN  = '\033[48;5;22m'
    BG_BLUE   = '\033[48;5;17m'
    BG_GREY   = '\033[48;5;236m'


W = 72  # card width


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def box_top():
    return f"{C.BLUE}╔{'═' * W}╗{C.RST}"

def box_bot():
    return f"{C.BLUE}╚{'═' * W}╝{C.RST}"

def box_sep():
    return f"{C.BLUE}╠{'═' * W}╣{C.RST}"

def box_sep_thin():
    return f"{C.BLUE}╟{'─' * W}╢{C.RST}"

def box_line(content='', align='left'):
    stripped = strip_ansi(content)
    pad = W - len(stripped)
    if pad < 0:
        pad = 0
    if align == 'center':
        left_pad = pad // 2
        right_pad = pad - left_pad
        inner = ' ' * left_pad + content + ' ' * right_pad
    elif align == 'right':
        inner = ' ' * pad + content
    else:
        inner = content + ' ' * pad
    return f"{C.BLUE}║{C.RST}{inner}{C.BLUE}║{C.RST}"

def strip_ansi(s):
    import re
    return re.sub(r'\033\[[0-9;]*m', '', s)

def box_kv(key, value, key_color=C.WHITE, val_color=C.GREY):
    k = f"{key_color}{key}{C.RST}"
    v = f"{val_color}{value}{C.RST}"
    gap = W - len(strip_ansi(k)) - len(strip_ansi(v))
    if gap < 1:
        gap = 1
    dots = f"{C.DIM}{'·' * gap}{C.RST}"
    return box_line(f"{k}{dots}{v}")


def bar(fraction, width=20, fill_color=C.GREEN, empty_color=C.DIM):
    filled = int(fraction * width)
    empty = width - filled
    return (f"{fill_color}{'█' * filled}{C.RST}"
            f"{empty_color}{'░' * empty}{C.RST}")


# ---------------------------------------------------------------------------
# Diagnostic lookup table
# ---------------------------------------------------------------------------

DIAGNOSTICS = {
    'chi2':            'RCDT table values, BerExp acceptance probs',
    'tail_exceedance': 'RCDT support bounds, tail truncation in BaseSampler',
    'sign_halfgauss':  'sign selection (s = b + (2b-1)·z), half-Gaussian fold',
    'discrete_ad':     'RCDT table rounding, BerExp quantization',
    'higher_crit':     'single entry in RCDT or BerExp lookup table',
    'ljung_box':       'RNG state management, PRNG buffer reuse between calls',
    'runs_test':       'sequential dependence, shared PRNG state',
    'block_homog':     'state corruption over time, memory buffer reuse',
}

FLAW_FINGERPRINTS = [
    # (name, key_failures, key_passes)
    # key_failures: at least one must fail. key_passes: these passing strengthens the match.
    ('sigma inflation',
     {'chi2', 'discrete_ad', 'higher_crit'},
     {'ljung_box', 'runs_test'}),
    ('mu shift',
     {'chi2', 'discrete_ad'},
     {'sign_halfgauss', 'ljung_box', 'runs_test'}),
    ('RCDT/BerExp table error',
     {'chi2', 'higher_crit'},
     {'ljung_box', 'runs_test'}),
    ('sign coupling (SHIFT SNARE surface)',
     {'chi2', 'sign_halfgauss'},
     {'ljung_box', 'runs_test'}),
    ('tail truncation',
     {'tail_exceedance', 'chi2'},
     {'ljung_box', 'runs_test'}),
    ('Markov coupling / serial dependence',
     {'ljung_box', 'runs_test'},
     {'chi2', 'tail_exceedance'}),
    ('mid-run state drift / buffer corruption',
     {'block_homog'},
     {'chi2', 'ljung_box'}),
    ('uniform contamination',
     {'chi2', 'discrete_ad'},
     {'sign_halfgauss', 'ljung_box'}),
]


def match_fingerprint(failed_tests, passed_tests):
    matches = []
    for name, key_fail, key_pass in FLAW_FINGERPRINTS:
        fail_overlap = len(key_fail & failed_tests)
        if fail_overlap == 0:
            continue
        pass_confirmed = len(key_pass & passed_tests)
        pass_violated = len(key_pass & failed_tests)
        score = fail_overlap * 3 + pass_confirmed - pass_violated * 2
        matches.append((score, name, key_fail, key_pass))
    matches.sort(reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Format p-value
# ---------------------------------------------------------------------------

def fmt_p(p, alpha=0.001):
    if p == 0 or p < 1e-300:
        s = "< 10^-300"
    elif p < 1e-10:
        exp = int(np.floor(np.log10(p)))
        mantissa = p / 10**exp
        s = f"{mantissa:.1f}×10^{exp}"
    elif p < 0.001:
        s = f"{p:.1e}"
    else:
        s = f"{p:.3f}"
    if p <= alpha:
        return f"{C.RED}{C.BOLD}{s}{C.RST}"
    else:
        return f"{C.GREEN}{s}{C.RST}"


# ---------------------------------------------------------------------------
# Run all tests and collect results
# ---------------------------------------------------------------------------

def run_all_tests(mu, sigma, samples, alpha=0.001):
    results = []

    # chi2 via UnivariateSamples
    uv = UnivariateSamples(mu, sigma, samples)
    results.append({
        'name': 'chi2',
        'label': 'χ² goodness-of-fit',
        'pvalue': float(uv.chi2_pvalue),
        'passes': uv.is_valid,
        'stat': float(uv.chi2_stat),
    })

    # tail exceedance
    r = tail_exceedance(mu, sigma, samples, alpha=alpha)
    min_p = min(v['pvalue'] for v in r['thresholds'].values())
    results.append({
        'name': 'tail_exceedance',
        'label': 'Tail exceedance (3-6σ)',
        'pvalue': min_p,
        'passes': r['passes'],
    })

    # sign/half-Gaussian
    r = sign_halfgaussian(mu, sigma, samples, alpha=alpha)
    results.append({
        'name': 'sign_halfgauss',
        'label': 'Sign / half-Gaussian',
        'pvalue': r['magnitude_chi2']['pvalue'],
        'passes': r['passes'],
    })

    # discrete AD
    r = discrete_anderson_darling(mu, sigma, samples, alpha=alpha, mc_B=200)
    results.append({
        'name': 'discrete_ad',
        'label': 'Discrete Anderson-Darling',
        'pvalue': r['pvalue'],
        'passes': r['passes'],
        'stat': r['statistic'],
    })

    # higher criticism
    r = higher_criticism(mu, sigma, samples, alpha=alpha, mc_B=200)
    results.append({
        'name': 'higher_crit',
        'label': 'Higher criticism',
        'pvalue': r['pvalue'],
        'passes': r['passes'],
        'stat': r['statistic'],
    })

    # ljung-box
    r = ljung_box(samples, alpha=alpha)
    results.append({
        'name': 'ljung_box',
        'label': 'Ljung-Box Q(20)',
        'pvalue': r['pvalue'],
        'passes': r['passes'],
        'stat': r['statistic'],
    })

    # runs test
    r = runs_test(samples, alpha=alpha)
    results.append({
        'name': 'runs_test',
        'label': 'Wald-Wolfowitz runs',
        'pvalue': r['pvalue'],
        'passes': r['passes'],
    })

    # block homogeneity
    r = block_homogeneity(mu, sigma, samples, alpha=alpha)
    results.append({
        'name': 'block_homog',
        'label': 'Block homogeneity',
        'pvalue': r['pvalue'],
        'passes': r['passes'],
    })

    return results


# ---------------------------------------------------------------------------
# Print the report card
# ---------------------------------------------------------------------------

def print_report_card(mu, sigma, n, results, alpha=0.001):
    all_pass = all(r['passes'] for r in results)
    n_fail = sum(1 for r in results if not r['passes'])
    n_pass = sum(1 for r in results if r['passes'])

    lines = []
    lines.append('')
    lines.append(box_top())

    # Title
    lines.append(box_line())
    title = f"{C.BOLD}{C.CYAN}   ███████╗ █████╗  ██████╗  █████╗{C.RST}"
    lines.append(box_line(title, 'center'))
    title2 = f"{C.BOLD}{C.CYAN}   ██╔════╝██╔══██╗██╔════╝ ██╔══██╗{C.RST}"
    lines.append(box_line(title2, 'center'))
    title3 = f"{C.BOLD}{C.CYAN}   ███████╗███████║██║  ███╗███████║{C.RST}"
    lines.append(box_line(title3, 'center'))
    title4 = f"{C.BOLD}{C.CYAN}   ╚════██║██╔══██║██║   ██║██╔══██║{C.RST}"
    lines.append(box_line(title4, 'center'))
    title5 = f"{C.BOLD}{C.CYAN}   ███████║██║  ██║╚██████╔╝██║  ██║{C.RST}"
    lines.append(box_line(title5, 'center'))
    title6 = f"{C.BOLD}{C.CYAN}   ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝{C.RST}"
    lines.append(box_line(title6, 'center'))
    lines.append(box_line(
        f"{C.DIM}Statistical Analysis of Gaussian Adequacy{C.RST}", 'center'))
    lines.append(box_line())

    lines.append(box_sep())

    # Verdict
    lines.append(box_line())
    if all_pass:
        verdict = (f"  {C.BG_GREEN}{C.WHITE}{C.BOLD}"
                   f"   ✓  ALL TESTS PASSED   "
                   f"{C.RST}")
    else:
        verdict = (f"  {C.BG_RED}{C.WHITE}{C.BOLD}"
                   f"   ✗  {n_fail} TEST{'S' if n_fail > 1 else ''} FAILED   "
                   f"{C.RST}")
    lines.append(box_line(verdict, 'center'))
    lines.append(box_line())

    # Parameters
    params = (f"  {C.DIM}μ={C.RST}{C.WHITE}{mu}{C.RST}"
              f"  {C.DIM}σ={C.RST}{C.WHITE}{sigma}{C.RST}"
              f"  {C.DIM}n={C.RST}{C.WHITE}{n:,}{C.RST}"
              f"  {C.DIM}α={C.RST}{C.WHITE}{alpha}{C.RST}")
    lines.append(box_line(params, 'center'))
    lines.append(box_line())

    # Progress bar
    frac = n_pass / len(results)
    pbar = bar(frac, width=40,
               fill_color=C.GREEN if all_pass else C.YELLOW)
    pbar_label = f"  {pbar} {C.WHITE}{n_pass}/{len(results)}{C.RST}"
    lines.append(box_line(pbar_label, 'center'))
    lines.append(box_line())

    lines.append(box_sep())

    # Section: distributional
    lines.append(box_line())
    lines.append(box_line(
        f"  {C.BLUE}{C.BOLD}▸ DISTRIBUTIONAL{C.RST}"))
    lines.append(box_line())

    dist_tests = ['chi2', 'tail_exceedance', 'sign_halfgauss',
                  'discrete_ad', 'higher_crit']
    for r in results:
        if r['name'] not in dist_tests:
            continue
        icon = f"{C.GREEN}●{C.RST}" if r['passes'] else f"{C.RED}○{C.RST}"
        p_str = fmt_p(r['pvalue'], alpha)
        label = f"{C.WHITE}{r['label']}{C.RST}"
        line_content = f"  {icon}  {label}"
        p_part = f"p = {p_str}"
        gap = W - len(strip_ansi(line_content)) - len(strip_ansi(p_part)) - 1
        if gap < 1:
            gap = 1
        lines.append(box_line(
            f"{line_content}{' ' * gap}{p_part}"))
        if not r['passes'] and r['name'] in DIAGNOSTICS:
            hint = DIAGNOSTICS[r['name']]
            lines.append(box_line(
                f"       {C.ORANGE}↳ check: {hint}{C.RST}"))

    lines.append(box_line())
    lines.append(box_sep_thin())

    # Section: sequence
    lines.append(box_line())
    lines.append(box_line(
        f"  {C.BLUE}{C.BOLD}▸ SEQUENCE{C.RST}"))
    lines.append(box_line())

    seq_tests = ['ljung_box', 'runs_test', 'block_homog']
    for r in results:
        if r['name'] not in seq_tests:
            continue
        icon = f"{C.GREEN}●{C.RST}" if r['passes'] else f"{C.RED}○{C.RST}"
        p_str = fmt_p(r['pvalue'], alpha)
        label = f"{C.WHITE}{r['label']}{C.RST}"
        line_content = f"  {icon}  {label}"
        p_part = f"p = {p_str}"
        gap = W - len(strip_ansi(line_content)) - len(strip_ansi(p_part)) - 1
        if gap < 1:
            gap = 1
        lines.append(box_line(
            f"{line_content}{' ' * gap}{p_part}"))
        if not r['passes'] and r['name'] in DIAGNOSTICS:
            hint = DIAGNOSTICS[r['name']]
            lines.append(box_line(
                f"       {C.ORANGE}↳ check: {hint}{C.RST}"))

    lines.append(box_line())

    # Fingerprint diagnosis (only if failures)
    if not all_pass:
        lines.append(box_sep())
        lines.append(box_line())
        lines.append(box_line(
            f"  {C.RED}{C.BOLD}▸ FAILURE FINGERPRINT{C.RST}"))
        lines.append(box_line())

        failed = {r['name'] for r in results if not r['passes']}
        passed = {r['name'] for r in results if r['passes']}

        # Show pattern
        pattern_parts = []
        for r in results:
            if r['passes']:
                pattern_parts.append(f"{C.GREEN}•{C.RST}")
            else:
                pattern_parts.append(f"{C.RED}✗{C.RST}")
        pattern_str = ' '.join(pattern_parts)
        lines.append(box_line(f"  Pattern: {pattern_str}"))
        lines.append(box_line())

        matches = match_fingerprint(failed, passed)
        if matches:
            best = matches[0]
            lines.append(box_line(
                f"  {C.YELLOW}{C.BOLD}→ Best match: "
                f"{best[1]}{C.RST}"))
            if len(matches) > 1:
                also = matches[1]
                lines.append(box_line(
                    f"  {C.YELLOW}  Also consistent: "
                    f"{also[1]}{C.RST}"))

            ruled_out = []
            for name, ef, ep in FLAW_FINGERPRINTS:
                if all(m[1] != name for m in matches):
                    if len(ep & passed) > 0 or len(ef & failed) == 0:
                        ruled_out.append(name)
            if ruled_out:
                lines.append(box_line(
                    f"  {C.DIM}Ruled out: "
                    f"{', '.join(ruled_out[:3])}{C.RST}"))
        else:
            lines.append(box_line(
                f"  {C.DIM}No known fingerprint match — "
                f"novel flaw pattern{C.RST}"))

        lines.append(box_line())

    # Footer
    lines.append(box_sep())
    lines.append(box_line())
    status_color = C.GREEN if all_pass else C.RED
    footer = (f"  {status_color}{n_pass} passed{C.RST}"
              f"{C.DIM} · {C.RST}"
              f"{status_color}{n_fail} failed{C.RST}"
              f"{C.DIM} · {C.RST}"
              f"{C.GREY}8 tests{C.RST}"
              f"{C.DIM} · {C.RST}"
              f"{C.GREY}SAGA v2{C.RST}")
    lines.append(box_line(footer, 'center'))
    lines.append(box_line())
    lines.append(box_bot())
    lines.append('')

    print('\n'.join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SAGA Report Card')
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--sigma', type=float, default=1.55)
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--flaw', choices=[
        'none', 'sigma5', 'sigma10', 'sign55', 'markov',
        'trunc3', 'table2x', 'contam5'
    ], default='none', help='Inject a flaw for demo')
    args = parser.parse_args()

    pdt = make_gaussian_pdt(args.mu, args.sigma)
    support = np.array(sorted(pdt.keys()))
    probs = np.array([pdt[z] for z in support])
    rng = np.random.default_rng(args.seed)

    if args.flaw == 'none':
        samples = rng.choice(support, size=args.n, p=probs).tolist()
    elif args.flaw == 'sigma5':
        pdt2 = make_gaussian_pdt(args.mu, args.sigma * 1.05)
        s2 = np.array(sorted(pdt2.keys()))
        p2 = np.array([pdt2[z] for z in s2])
        samples = rng.choice(s2, size=args.n, p=p2).tolist()
    elif args.flaw == 'sigma10':
        pdt2 = make_gaussian_pdt(args.mu, args.sigma * 1.10)
        s2 = np.array(sorted(pdt2.keys()))
        p2 = np.array([pdt2[z] for z in s2])
        samples = rng.choice(s2, size=args.n, p=p2).tolist()
    elif args.flaw == 'sign55':
        p_biased = probs.copy()
        for i, z in enumerate(support):
            if z > args.mu:
                p_biased[i] *= 0.55 / 0.5
            elif z < args.mu:
                p_biased[i] *= 0.45 / 0.5
        p_biased /= p_biased.sum()
        samples = rng.choice(support, size=args.n, p=p_biased).tolist()
    elif args.flaw == 'markov':
        samples = [rng.choice(support, p=probs)]
        for _ in range(args.n - 1):
            if rng.random() < 0.15:
                samples.append(samples[-1])
            else:
                samples.append(rng.choice(support, p=probs))
        samples = [int(x) for x in samples]
    elif args.flaw == 'trunc3':
        mask = np.abs(support - args.mu) <= 3 * args.sigma
        p_trunc = probs * mask
        p_trunc /= p_trunc.sum()
        samples = rng.choice(support, size=args.n, p=p_trunc).tolist()
    elif args.flaw == 'table2x':
        p_err = probs.copy()
        mode = np.argmax(p_err)
        p_err[mode + 2] *= 2.0
        p_err /= p_err.sum()
        samples = rng.choice(support, size=args.n, p=p_err).tolist()
    elif args.flaw == 'contam5':
        p_cont = 0.95 * probs + 0.05 / len(probs)
        p_cont /= p_cont.sum()
        samples = rng.choice(support, size=args.n, p=p_cont).tolist()

    results = run_all_tests(args.mu, args.sigma, samples, alpha=0.001)
    print_report_card(args.mu, args.sigma, args.n, results)


if __name__ == '__main__':
    main()
