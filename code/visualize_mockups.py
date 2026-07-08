"""
Generate mockup visualizations for SAGA v2.
Run: python visualize_mockups.py
Outputs PNGs to ../figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects

FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.family': 'monospace',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.facecolor': 'white',
})

SAGA_BLUE = '#1e3a5f'
SAGA_GREEN = '#2d8659'
SAGA_RED = '#c0392b'
SAGA_AMBER = '#d4a017'
SAGA_GREY = '#7f8c8d'


# =====================================================================
# 1. Power Matrix Heatmap
# =====================================================================

def plot_power_matrix():
    tests = ['chi2', 'tail_exc', 'sign_hg', 'disc_AD', 'higher_c',
             'ljung_box', 'runs', 'block_h']
    flaws = [
        'sigma+0.1%', 'sigma+0.5%', 'sigma+1%', 'sigma+2%',
        'sigma+5%', 'sigma+10%',
        'mu+0.05', 'mu+0.1', 'mu+0.2', 'mu+0.5',
        'sign 52/48', 'sign 55/45', 'sign 60/40',
        'trunc 3sig', 'trunc 4sig', 'trunc 5sig',
        'table 1.5x', 'table 2x', 'table 3x',
        'contam 1%', 'contam 5%',
        'markov 0.05', 'markov 0.1', 'markov 0.3',
    ]

    rng = np.random.default_rng(42)
    data = np.zeros((len(flaws), len(tests)))

    profiles = {
        'chi2':     [0, .02, .15, .55, .98, 1,   .05, .30, .85, 1,   0, 0, 0,     .60, .10, 0,   .40, .90, 1,   .08, .75,  0, 0, 0],
        'tail_exc': [0, 0,   .02, .08, .35, .80, 0,   .05, .20, .70, 0, 0, 0,     1,   .95, .40, .10, .30, .60, .02, .25,  0, 0, 0],
        'sign_hg':  [0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   .55,.92,1,   0,   0,   0,   0,   0,   0,   0,   0,    0, 0, 0],
        'disc_AD':  [0, .05, .25, .70, 1,   1,   .08, .40, .92, 1,   0, 0, 0,     .85, .25, .05, .55, .95, 1,   .12, .82,  0, 0, 0],
        'higher_c': [0, .03, .18, .60, .99, 1,   .06, .35, .88, 1,   0, 0, 0,     .70, .15, .02, .65, .98, 1,   .10, .78,  0, 0, 0],
        'ljung_box':[0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0,     0,   0,   0,   0,   0,   0,   0,   0,    .30,.88,1],
        'runs':     [0, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0,     0,   0,   0,   0,   0,   0,   0,   0,    .15,.65,.98],
        'block_h':  [0, 0,   .02, .08, .30, .65, .02, .08, .25, .55, 0, 0, 0,     .15, .03, 0,   .10, .25, .50, .03, .20,  .05,.20,.55],
    }

    for j, t in enumerate(tests):
        data[:, j] = profiles[t]

    noise = rng.normal(0, 0.02, data.shape)
    data = np.clip(data + noise, 0, 1)

    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = LinearSegmentedColormap.from_list('saga',
        [(0, '#2c3e50'), (0.2, '#e74c3c'), (0.5, '#f39c12'),
         (0.8, '#27ae60'), (1.0, '#1abc9c')])

    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(flaws)))
    ax.set_yticklabels(flaws, fontsize=8)

    for i in range(len(flaws)):
        for j in range(len(tests)):
            val = data[i, j]
            if val >= 0.01:
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    # Flaw family separators
    for y in [6, 10, 13, 16, 19, 21]:
        ax.axhline(y - 0.5, color='white', linewidth=1.5, alpha=0.8)

    cbar = plt.colorbar(im, ax=ax, label='Detection Power', shrink=0.8)
    cbar.set_ticks([0, 0.2, 0.5, 0.8, 1.0])
    cbar.set_ticklabels(['0%', '20%', '50%', '80%', '100%'])

    ax.set_title('SAGA v2 Power Matrix — Detection Rate by Test × Flaw\n'
                 '(n=10,000, α=10⁻³, 100 replicates)', fontsize=13, pad=15)
    ax.set_xlabel('Test')
    ax.set_ylabel('Flaw Type')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '1_power_matrix.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("  1. Power matrix heatmap")


# =====================================================================
# 2. Detection Frontier Curves
# =====================================================================

def plot_detection_frontiers():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    families = [
        ('sigma shift', [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
         'ε (fractional shift)'),
        ('mu shift', [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
         'δ (additive shift)'),
        ('sign coupling', [0.505, 0.51, 0.52, 0.55, 0.6, 0.65],
         'bias (prob of positive sign)'),
        ('tail truncation', [2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
         'k (truncation at k·σ)'),
        ('table error', [1.1, 1.2, 1.5, 2.0, 3.0, 5.0],
         'factor (probability multiplier)'),
        ('Markov coupling', [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
         'ρ (repeat probability)'),
    ]

    n_values = [1000, 2000, 5000, 10000, 20000, 50000]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c']

    for idx, (name, strengths, xlabel) in enumerate(families):
        ax = axes[idx // 3, idx % 3]

        for ni, n in enumerate(n_values):
            powers = []
            for s in strengths:
                if name == 'sigma shift':
                    power = 1 - np.exp(-n * s**2 / 2)
                elif name == 'mu shift':
                    power = 1 - np.exp(-n * s**2 / 8)
                elif name == 'sign coupling':
                    power = 1 - np.exp(-n * (s - 0.5)**2 * 20)
                elif name == 'tail truncation':
                    mass = np.exp(-s**2 / 2) * 0.5
                    power = 1 - np.exp(-n * mass * 2)
                elif name == 'table error':
                    power = 1 - np.exp(-n * (s - 1)**2 / 50)
                elif name == 'Markov coupling':
                    power = 1 - np.exp(-n * s**2 * 5)
                powers.append(min(power, 1.0))

            ax.plot(strengths, powers, '-o', color=colors[ni],
                    label=f'n={n:,}', markersize=3, linewidth=1.5)

        ax.axhline(0.8, color='grey', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(strengths[0], 0.82, '80% power', fontsize=7, color='grey')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Detection power', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale('log' if name != 'sign coupling' else 'linear')
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc='lower right', ncol=2)
    fig.suptitle('Detection Frontiers — Minimum Flaw Strength at 80% Power\n'
                 '(α=10⁻³, best test per flaw family)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '2_detection_frontiers.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("  2. Detection frontier curves")


# =====================================================================
# 3. p-value Histogram (calibration check)
# =====================================================================

def plot_pvalue_histogram():
    rng = np.random.default_rng(3000)
    pvalues = rng.uniform(0, 1, 200)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(pvalues, bins=20, color=SAGA_BLUE, edgecolor='white',
            alpha=0.85, density=True)
    ax.axhline(1.0, color=SAGA_RED, linestyle='--', linewidth=2,
               label='U[0,1] expected')
    ax.set_xlabel('p-value')
    ax.set_ylabel('Density')
    ax.set_title('χ² p-values under H₀\n(should be uniform)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    sorted_p = np.sort(pvalues)
    n = len(sorted_p)
    theoretical = np.arange(1, n + 1) / (n + 1)
    ax.scatter(theoretical, sorted_p, s=8, color=SAGA_BLUE, alpha=0.6)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y = x (perfect)')
    ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.05, color='red')
    ci = 1.36 / np.sqrt(n)
    ax.plot([0, 1], [-ci, 1 - ci], 'grey', linewidth=0.8, linestyle=':')
    ax.plot([0, 1], [ci, 1 + ci], 'grey', linewidth=0.8, linestyle=':',
            label='KS 95% band')
    ax.set_xlabel('Theoretical quantile')
    ax.set_ylabel('Observed p-value')
    ax.set_title('p-value QQ-plot\n(KS test of uniformity)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '3_pvalue_calibration.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("  3. p-value calibration check")


# =====================================================================
# 4. FFT-domain Variance Spectrum
# =====================================================================

def plot_fft_spectrum():
    rng = np.random.default_rng(99)
    dim = 64

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Good sampler
    variance_good = np.ones(dim) + rng.normal(0, 0.05, dim)
    ax = axes[0]
    bars = ax.bar(range(dim), variance_good, color=SAGA_GREEN, alpha=0.7,
                  edgecolor='none')
    ax.axhline(1.0, color='black', linestyle='-', linewidth=1.5)
    ax.fill_between(range(dim), 0.85, 1.15, alpha=0.15, color='grey',
                    label='±15% tolerance')
    ax.set_xlabel('Frequency index')
    ax.set_ylabel('Normalized variance')
    ax.set_title('Good sampler — FFT variance spectrum', fontsize=11,
                 color=SAGA_GREEN)
    ax.set_ylim(0.4, 1.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Bad sampler (ffSampling flaw: zeroed subtree)
    variance_bad = np.ones(dim) + rng.normal(0, 0.05, dim)
    variance_bad[16:24] *= 0.3  # subtree zeroed
    variance_bad[48:52] *= 1.8  # compensating excess

    ax = axes[1]
    colors_bad = [SAGA_RED if (16 <= i < 24 or 48 <= i < 52)
                  else SAGA_BLUE for i in range(dim)]
    ax.bar(range(dim), variance_bad, color=colors_bad, alpha=0.7,
           edgecolor='none')
    ax.axhline(1.0, color='black', linestyle='-', linewidth=1.5)
    ax.fill_between(range(dim), 0.85, 1.15, alpha=0.15, color='grey')

    ax.annotate('subtree zeroed', xy=(20, 0.35), fontsize=9,
                color=SAGA_RED, ha='center', fontweight='bold')
    ax.annotate('excess variance', xy=(50, 1.85), fontsize=9,
                color=SAGA_RED, ha='center', fontweight='bold')
    ax.set_xlabel('Frequency index')
    ax.set_ylabel('Normalized variance')
    ax.set_title('Flawed sampler — FFT tree node zeroed', fontsize=11,
                 color=SAGA_RED)
    ax.set_ylim(0.0, 2.2)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '4_fft_spectrum.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("  4. FFT-domain variance spectrum")


# =====================================================================
# 5. Manhattan Plot (per-coordinate p-values)
# =====================================================================

def plot_manhattan():
    rng = np.random.default_rng(77)
    dim = 512

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Good
    pvals_good = rng.uniform(0, 1, dim)
    neglog_good = -np.log10(pvals_good)
    bh_threshold = -np.log10(0.001)

    ax = axes[0]
    colors_g = [SAGA_BLUE if nl < bh_threshold else SAGA_RED
                for nl in neglog_good]
    ax.scatter(range(dim), neglog_good, c=colors_g, s=6, alpha=0.6)
    ax.axhline(bh_threshold, color=SAGA_RED, linestyle='--', linewidth=1.5,
               label=f'BH threshold (α=10⁻³)')
    ax.set_ylabel('-log₁₀(p)')
    ax.set_title('Good sampler — per-coordinate χ² p-values', fontsize=11,
                 color=SAGA_GREEN)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Bad (coordinates 200-210 have inflated variance)
    pvals_bad = rng.uniform(0, 1, dim)
    pvals_bad[200:210] = rng.uniform(1e-12, 1e-6, 10)
    pvals_bad[350:353] = rng.uniform(1e-8, 1e-4, 3)
    neglog_bad = -np.log10(np.clip(pvals_bad, 1e-15, 1))

    ax = axes[1]
    colors_b = [SAGA_RED if nl > bh_threshold else SAGA_BLUE
                for nl in neglog_bad]
    ax.scatter(range(dim), neglog_bad, c=colors_b, s=6, alpha=0.6)
    ax.axhline(bh_threshold, color=SAGA_RED, linestyle='--', linewidth=1.5,
               label=f'BH threshold (α=10⁻³)')

    ax.annotate('coords 200-210\n(wrong variance)', xy=(205, 9),
                fontsize=9, color=SAGA_RED, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=SAGA_RED),
                xytext=(250, 11))
    ax.set_xlabel('Coordinate index')
    ax.set_ylabel('-log₁₀(p)')
    ax.set_title('Flawed sampler — AVX2 lane reuse on coords 200-210',
                 fontsize=11, color=SAGA_RED)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '5_manhattan.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("  5. Per-coordinate Manhattan plot")


# =====================================================================
# 6. Autocorrelation (ACF) Plot
# =====================================================================

def plot_acf():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n = 10000
    max_lag = 20

    # Good: near-zero autocorrelation
    rng = np.random.default_rng(55)
    acf_good = rng.normal(0, 1/np.sqrt(n), max_lag)

    ax = axes[0]
    ax.bar(range(1, max_lag + 1), acf_good, color=SAGA_BLUE, alpha=0.7)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color=SAGA_RED, linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(-ci, color=SAGA_RED, linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(range(0, max_lag + 2), -ci, ci, alpha=0.1,
                    color=SAGA_RED, label='95% CI under iid')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Good sampler — ACF', fontsize=11, color=SAGA_GREEN)
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, max_lag + 0.5)
    ax.grid(True, alpha=0.2)

    # Bad: Markov coupling (rho=0.1)
    acf_bad = 0.1 ** np.arange(1, max_lag + 1) + rng.normal(0, 0.005, max_lag)

    ax = axes[1]
    colors = [SAGA_RED if abs(a) > ci else SAGA_BLUE for a in acf_bad]
    ax.bar(range(1, max_lag + 1), acf_bad, color=colors, alpha=0.7)
    ax.axhline(ci, color=SAGA_RED, linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(-ci, color=SAGA_RED, linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(range(0, max_lag + 2), -ci, ci, alpha=0.1, color=SAGA_RED)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Flawed sampler — Markov ρ=0.1', fontsize=11, color=SAGA_RED)
    ax.set_xlim(0.5, max_lag + 0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '6_acf.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("  6. Autocorrelation plot")


# =====================================================================
# 7. Renyi Divergence Profile
# =====================================================================

def plot_renyi_profile():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    sigmas = np.linspace(1.0, 3.0, 200)
    sigmin = 1.2778

    rd_128 = np.exp(-128 * (sigmas - sigmin)**2 / (2 * sigmas**2)) * 1e-10
    rd_128 = np.where(sigmas < sigmin, np.exp((sigmin/sigmas - 1) * 80), rd_128)

    target = 2.65e-23
    ax.semilogy(sigmas, rd_128, color=SAGA_BLUE, linewidth=2.5,
                label='R₁₂₈ - 1 (base sampler)')

    ax.axhline(target, color=SAGA_RED, linestyle='--', linewidth=2,
               label=f'Certification target (2⁻⁷⁵ ≈ {target:.1e})')

    # Mark key points
    ax.plot(1.55, 1.3e-29, 'o', color=SAGA_GREEN, markersize=12, zorder=5)
    ax.annotate('σ=1.55 (default)\nR₁₂₈-1 ≈ 1.3×10⁻²⁹  ✓',
                xy=(1.55, 1.3e-29), xytext=(1.8, 1e-32),
                fontsize=9, color=SAGA_GREEN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=SAGA_GREEN))

    ax.plot(1.80, 3.94e-23, 's', color=SAGA_RED, markersize=12, zorder=5)
    ax.annotate('σ=1.80 (boundary)\nR₁₂₈-1 ≈ 3.9×10⁻²³  ✗',
                xy=(1.80, 3.94e-23), xytext=(2.1, 1e-20),
                fontsize=9, color=SAGA_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=SAGA_RED))

    ax.plot(1.2778, 5e-11, 'D', color=SAGA_AMBER, markersize=10, zorder=5)
    ax.annotate('σ_min = 1.2778',
                xy=(1.2778, 5e-11), xytext=(1.05, 1e-14),
                fontsize=9, color=SAGA_AMBER, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=SAGA_AMBER))

    ax.fill_between(sigmas, target, 1, alpha=0.08, color=SAGA_RED,
                    label='FAIL region')
    ax.fill_between(sigmas, 1e-40, target, alpha=0.05, color=SAGA_GREEN,
                    label='PASS region')

    ax.set_xlabel('σ (sampler standard deviation)', fontsize=11)
    ax.set_ylabel('R₁₂₈ - 1  (Rényi divergence excess)', fontsize=11)
    ax.set_title('Rényi Divergence Certification Profile\n'
                 'Base sampler BerExp chain vs ideal Gaussian',
                 fontsize=13, pad=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(1e-35, 1e-5)
    ax.set_xlim(1.0, 3.0)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '7_renyi_profile.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("  7. Renyi divergence profile")


# =====================================================================
# 8. QQ-plot of Squared Norms
# =====================================================================

def plot_qq_norms():
    from scipy.stats import chi2
    rng = np.random.default_rng(123)
    dim = 128
    n = 2000

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Good
    data_good = rng.normal(0, 1.55, (n, dim))
    norms_good = np.sum(data_good**2, axis=1) / 1.55**2
    theoretical_q = chi2.ppf(np.linspace(0.005, 0.995, n), dim)
    empirical_q = np.sort(norms_good)

    ax = axes[0]
    ax.scatter(theoretical_q, empirical_q, s=4, color=SAGA_BLUE, alpha=0.4)
    lims = [min(theoretical_q), max(theoretical_q)]
    ax.plot(lims, lims, 'r-', linewidth=2, label='y = x')
    ax.set_xlabel('χ²(128) theoretical quantiles')
    ax.set_ylabel('||x||²/σ² empirical quantiles')
    ax.set_title('Good sampler', fontsize=11, color=SAGA_GREEN)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Bad: inflated norms
    data_bad = rng.normal(0, 1.55, (n, dim))
    data_bad[:, :20] *= 1.3
    norms_bad = np.sum(data_bad**2, axis=1) / 1.55**2
    empirical_bad = np.sort(norms_bad)

    ax = axes[1]
    ax.scatter(theoretical_q, empirical_bad, s=4, color=SAGA_RED, alpha=0.4)
    ax.plot(lims, lims, 'r-', linewidth=2, label='y = x')
    ax.set_xlabel('χ²(128) theoretical quantiles')
    ax.set_ylabel('||x||²/σ² empirical quantiles')
    ax.set_title('Flawed sampler (20 coords inflated)', fontsize=11,
                 color=SAGA_RED)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '8_qq_norms.png'), dpi=200,
                bbox_inches='tight')
    plt.close(fig)
    print("  8. Squared-norm QQ-plot")


# =====================================================================
# 9. SAGA Report Card (the "pokemon card")
# =====================================================================

def plot_report_card():
    fig = plt.figure(figsize=(8.5, 12))

    # Background
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.set_aspect('equal')
    ax.axis('off')

    # Card background with rounded corners
    card = FancyBboxPatch((3, 3), 94, 134, boxstyle="round,pad=2",
                          facecolor='#161b22', edgecolor='#30363d',
                          linewidth=2)
    ax.add_patch(card)

    # Header bar
    header = FancyBboxPatch((5, 122), 90, 13, boxstyle="round,pad=1",
                            facecolor=SAGA_BLUE, edgecolor='none')
    ax.add_patch(header)

    ax.text(50, 130, 'SAGA', fontsize=28, color='white', ha='center',
            va='center', fontweight='bold', fontfamily='monospace',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='#0a1929')])
    ax.text(50, 125.5, 'Statistical Analysis of Gaussian Adequacy',
            fontsize=8, color='#8b949e', ha='center', va='center',
            fontfamily='monospace')

    # Overall verdict
    verdict_color = '#2ea043'
    verdict_text = 'PASS'
    verdict_box = FancyBboxPatch((30, 112), 40, 8, boxstyle="round,pad=1",
                                 facecolor=verdict_color, edgecolor='none',
                                 alpha=0.25)
    ax.add_patch(verdict_box)
    ax.text(50, 116, verdict_text, fontsize=22, color=verdict_color,
            ha='center', va='center', fontweight='bold', fontfamily='monospace')

    # Parameters line
    ax.text(50, 109, 'Falcon-512  •  σ = 1.55  •  n = 10,000  •  α = 10⁻³',
            fontsize=8, color='#8b949e', ha='center', va='center',
            fontfamily='monospace')

    # ── Distributional tests section ──
    y = 103
    ax.text(8, y, '▸ DISTRIBUTIONAL', fontsize=9, color='#58a6ff',
            fontweight='bold', fontfamily='monospace')
    ax.plot([8, 92], [y - 1.5, y - 1.5], color='#21262d', linewidth=1)

    tests_dist = [
        ('χ² goodness-of-fit',     'p = 0.437', True),
        ('Discrete Anderson-Darling', 'p = 0.612', True),
        ('Higher criticism',       'p = 0.285', True),
        ('Tail exceedance (3-6σ)', 'all p > 0.05', True),
        ('Sign/half-Gaussian',    'p = 0.731', True),
    ]

    for i, (name, result, passes) in enumerate(tests_dist):
        yy = y - 4 - i * 4
        icon = '●' if passes else '○'
        icon_color = '#2ea043' if passes else SAGA_RED
        ax.text(10, yy, icon, fontsize=10, color=icon_color,
                va='center', fontfamily='monospace')
        ax.text(14, yy, name, fontsize=8.5, color='#c9d1d9',
                va='center', fontfamily='monospace')
        ax.text(88, yy, result, fontsize=8, color='#8b949e',
                va='center', ha='right', fontfamily='monospace')

    # ── Sequence tests section ──
    y = 78
    ax.text(8, y, '▸ SEQUENCE', fontsize=9, color='#58a6ff',
            fontweight='bold', fontfamily='monospace')
    ax.plot([8, 92], [y - 1.5, y - 1.5], color='#21262d', linewidth=1)

    tests_seq = [
        ('Ljung-Box Q(20)',   'p = 0.893', True),
        ('Wald-Wolfowitz runs', 'p = 0.641', True),
        ('Block homogeneity', 'p = 0.512', True),
    ]

    for i, (name, result, passes) in enumerate(tests_seq):
        yy = y - 4 - i * 4
        icon = '●' if passes else '○'
        icon_color = '#2ea043' if passes else SAGA_RED
        ax.text(10, yy, icon, fontsize=10, color=icon_color,
                va='center', fontfamily='monospace')
        ax.text(14, yy, name, fontsize=8.5, color='#c9d1d9',
                va='center', fontfamily='monospace')
        ax.text(88, yy, result, fontsize=8, color='#8b949e',
                va='center', ha='right', fontfamily='monospace')

    # ── Multivariate section ──
    y = 62
    ax.text(8, y, '▸ MULTIVARIATE', fontsize=9, color='#58a6ff',
            fontweight='bold', fontfamily='monospace')
    ax.plot([8, 92], [y - 1.5, y - 1.5], color='#21262d', linewidth=1)

    tests_mv = [
        ('Doornik-Hansen',      'p = 0.329', True),
        ('Squared-norm χ²(p)',  'p = 0.558', True),
        ('Fisher+BH meta',     '0/512 reject', True),
        ('Max |ρ_ij|',         '0.041 < 0.08', True),
        ('FFT-domain battery', 'all p > 0.01', True),
        ('Henze-Zirkler (MC)', 'p = 0.420', True),
    ]

    for i, (name, result, passes) in enumerate(tests_mv):
        yy = y - 4 - i * 4
        icon = '●' if passes else '○'
        icon_color = '#2ea043' if passes else SAGA_RED
        ax.text(10, yy, icon, fontsize=10, color=icon_color,
                va='center', fontfamily='monospace')
        ax.text(14, yy, name, fontsize=8.5, color='#c9d1d9',
                va='center', fontfamily='monospace')
        ax.text(88, yy, result, fontsize=8, color='#8b949e',
                va='center', ha='right', fontfamily='monospace')

    # ── Certification section ──
    y = 33
    ax.text(8, y, '▸ CERTIFICATION', fontsize=9, color='#58a6ff',
            fontweight='bold', fontfamily='monospace')
    ax.plot([8, 92], [y - 1.5, y - 1.5], color='#21262d', linewidth=1)

    cert = [
        ('Rényi R₁₂₈ - 1',    '1.3×10⁻²⁹', True, '< 2.65×10⁻²³'),
        ('Rényi R₂₅₆ - 1',    '8.7×10⁻⁵⁸', True, '< 2.65×10⁻²³'),
        ('Rényi R_∞  - 1',    '2.1×10⁻¹²', True, '< 2.65×10⁻²³'),
        ('Total variation',   '1.8×10⁻¹⁵', True, '< 2⁻⁴⁰'),
    ]

    for i, (name, result, passes, bound) in enumerate(cert):
        yy = y - 4 - i * 4
        icon = '●' if passes else '○'
        icon_color = '#2ea043' if passes else SAGA_RED
        ax.text(10, yy, icon, fontsize=10, color=icon_color,
                va='center', fontfamily='monospace')
        ax.text(14, yy, name, fontsize=8.5, color='#c9d1d9',
                va='center', fontfamily='monospace')
        ax.text(68, yy, result, fontsize=8, color=icon_color,
                va='center', ha='right', fontfamily='monospace',
                fontweight='bold')
        ax.text(88, yy, bound, fontsize=7.5, color='#484f58',
                va='center', ha='right', fontfamily='monospace')

    # Footer
    ax.plot([8, 92], [13, 13], color='#21262d', linewidth=1)
    ax.text(50, 10, '16 tests  •  127 assertions  •  0 false alarms',
            fontsize=8, color='#484f58', ha='center', fontfamily='monospace')
    ax.text(50, 7, 'SAGA v2 — Howe, Prest, et al. (2026)',
            fontsize=7.5, color='#30363d', ha='center', fontfamily='monospace')

    fig.savefig(os.path.join(FIGDIR, '9_report_card.png'), dpi=200,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  9. SAGA report card")


# =====================================================================
# 10. Report card — FAIL variant
# =====================================================================

def plot_report_card_fail():
    fig = plt.figure(figsize=(8.5, 13))
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 152)
    ax.set_aspect('equal')
    ax.axis('off')

    card = FancyBboxPatch((3, 3), 94, 146, boxstyle="round,pad=2",
                          facecolor='#161b22', edgecolor='#f85149',
                          linewidth=2)
    ax.add_patch(card)

    # Header
    header = FancyBboxPatch((5, 135), 90, 13, boxstyle="round,pad=1",
                            facecolor='#3d1114', edgecolor='none')
    ax.add_patch(header)
    ax.text(50, 143, 'SAGA', fontsize=28, color='#f85149', ha='center',
            va='center', fontweight='bold', fontfamily='monospace',
            path_effects=[patheffects.withStroke(linewidth=3, foreground='#1a0507')])
    ax.text(50, 138.5, 'Statistical Analysis of Gaussian Adequacy',
            fontsize=8, color='#8b949e', ha='center', va='center',
            fontfamily='monospace')

    # Verdict
    verdict_box = FancyBboxPatch((30, 126), 40, 7, boxstyle="round,pad=1",
                                 facecolor='#f85149', edgecolor='none',
                                 alpha=0.25)
    ax.add_patch(verdict_box)
    ax.text(50, 129.5, 'FAIL', fontsize=22, color='#f85149',
            ha='center', va='center', fontweight='bold', fontfamily='monospace')

    ax.text(50, 123, 'Falcon-512  •  σ = 1.55  •  n = 10,000  •  α = 10⁻³',
            fontsize=8, color='#8b949e', ha='center', va='center',
            fontfamily='monospace')

    # --- Helper to draw a test row with optional "check:" hint ---
    def draw_test(yy, name, result, passes, check_hint=None):
        icon = '●' if passes else '○'
        icon_color = '#2ea043' if passes else '#f85149'
        ax.text(10, yy, icon, fontsize=10, color=icon_color,
                va='center', fontfamily='monospace')
        ax.text(14, yy, name, fontsize=8.5, color='#c9d1d9',
                va='center', fontfamily='monospace')
        result_color = '#8b949e' if passes else '#f85149'
        ax.text(88, yy, result, fontsize=8, color=result_color,
                va='center', ha='right', fontfamily='monospace',
                fontweight='normal' if passes else 'bold')
        if check_hint and not passes:
            ax.text(14, yy - 2.5, f'↳ check: {check_hint}',
                    fontsize=6.5, color='#d29922', va='center',
                    fontfamily='monospace', fontstyle='italic')

    # ── Distributional ──
    y = 118
    ax.text(8, y, '▸ DISTRIBUTIONAL', fontsize=9, color='#58a6ff',
            fontweight='bold', fontfamily='monospace')
    ax.plot([8, 92], [y - 1.5, y - 1.5], color='#21262d', linewidth=1)

    tests_dist = [
        ('χ² goodness-of-fit',       'p = 2.1×10⁻⁸', False,
         'RCDT table values, BerExp acceptance probs'),
        ('Discrete Anderson-Darling', 'p = 0.000',     False,
         'RCDT table values, BerExp acceptance probs'),
        ('Higher criticism',          'p = 0.000',     False,
         'single entry in RCDT/BerExp table'),
        ('Tail exceedance (3σ)',      'p = 3.2×10⁻⁵', False,
         'RCDT support bounds, tail truncation'),
        ('Sign/half-Gaussian',        'p = 0.831',     True, None),
    ]

    for i, (name, result, passes, hint) in enumerate(tests_dist):
        yy = y - 4 - i * 5.5
        draw_test(yy, name, result, passes, hint)

    # ── Sequence ──
    y = 85
    ax.text(8, y, '▸ SEQUENCE', fontsize=9, color='#58a6ff',
            fontweight='bold', fontfamily='monospace')
    ax.plot([8, 92], [y - 1.5, y - 1.5], color='#21262d', linewidth=1)

    tests_seq = [
        ('Ljung-Box Q(20)',     'p = 0.712', True, None),
        ('Wald-Wolfowitz runs', 'p = 0.445', True, None),
        ('Block homogeneity',   'p = 0.003', False,
         'state corruption over time, buffer reuse'),
    ]

    for i, (name, result, passes, hint) in enumerate(tests_seq):
        yy = y - 4 - i * 5.5
        draw_test(yy, name, result, passes, hint)

    # ── Multivariate ──
    y = 65
    ax.text(8, y, '▸ MULTIVARIATE', fontsize=9, color='#58a6ff',
            fontweight='bold', fontfamily='monospace')
    ax.plot([8, 92], [y - 1.5, y - 1.5], color='#21262d', linewidth=1)

    tests_mv = [
        ('Doornik-Hansen',      'p = 0.028',  False,
         'ffSampling Gram-Schmidt, covariance structure'),
        ('Squared-norm χ²(p)',  'p = 0.004',  False,
         'ffSampling variance scaling, norm computation'),
        ('Fisher+BH meta',     '14/512 reject', False,
         'per-coordinate sampler — coords 200-213'),
        ('Max |ρ_ij|',         '0.041 < 0.08', True, None),
        ('FFT-domain battery', '2/64 freq fail', False,
         'FFT tree nodes 16-17, ffSampling subtree'),
        ('Henze-Zirkler (MC)', 'p = 0.420',   True, None),
    ]

    for i, (name, result, passes, hint) in enumerate(tests_mv):
        yy = y - 4 - i * 5.5
        draw_test(yy, name, result, passes, hint)

    # ── Fingerprint match box ──
    y = 26
    diag_box = FancyBboxPatch((7, y - 18), 86, 20, boxstyle="round,pad=1",
                               facecolor='#f8514920', edgecolor='#f85149',
                               linewidth=1)
    ax.add_patch(diag_box)

    ax.text(50, y - 1, '▸ FAILURE FINGERPRINT', fontsize=9, color='#f85149',
            ha='center', va='center', fontweight='bold',
            fontfamily='monospace')

    ax.text(12, y - 5.5,
            'Pattern:  chi2 + AD + HC + tail_exc  FAIL,  sign  PASS',
            fontsize=7.5, color='#f85149', fontfamily='monospace')
    ax.text(12, y - 9,
            'Nearest calibration match:  sigma inflation (2-5%)',
            fontsize=7.5, color='#d29922', fontfamily='monospace',
            fontweight='bold')
    ax.text(12, y - 12.5,
            'Also consistent with:  RCDT table rounding error',
            fontsize=7.5, color='#d29922', fontfamily='monospace')
    ax.text(12, y - 16,
            'Not consistent with:  sign coupling, Markov, truncation',
            fontsize=7.5, color='#484f58', fontfamily='monospace')
    ax.text(12, y - 19.5,
            'Ruled out because:  sign and sequence tests all pass',
            fontsize=7, color='#484f58', fontfamily='monospace',
            fontstyle='italic')

    # Footer
    ax.plot([8, 92], [10, 10], color='#21262d', linewidth=1)
    ax.text(50, 7.5, '7/16 tests failed  •  3 sampler components implicated',
            fontsize=8, color='#f85149', ha='center', fontfamily='monospace')
    ax.text(50, 5, 'SAGA v2 — Howe, Prest, et al. (2026)',
            fontsize=7.5, color='#30363d', ha='center', fontfamily='monospace')

    fig.savefig(os.path.join(FIGDIR, '10_report_card_fail.png'), dpi=100,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  10. SAGA report card (FAIL variant)")


# =====================================================================

if __name__ == "__main__":
    print("Generating SAGA v2 visualization mockups...\n")

    plot_power_matrix()
    plot_detection_frontiers()
    plot_pvalue_histogram()
    plot_fft_spectrum()
    plot_manhattan()
    plot_acf()
    plot_renyi_profile()
    plot_qq_norms()
    plot_report_card()
    plot_report_card_fail()

    print(f"\nAll figures saved to {os.path.abspath(FIGDIR)}/")
