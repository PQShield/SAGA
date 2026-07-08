# SAGA Test Suite: Improvement Analysis

## TL;DR

SAGA tests normality, but Falcon's security proofs use **Renyi divergence** -- so SAGA validates a proxy, not the actual security property. Meanwhile, 2023-2025 side-channel attacks (SHIFT SNARE, Zhang et al., Lin et al.) exploit **sign-magnitude correlations** and **rejection-count dependence** that SAGA doesn't test at all. The per-coordinate chi-square battery has an **uncorrected 87% false-alarm rate** for Falcon-1024, and the chi-square test's bucket aggregation creates a **blind spot in the tails** -- exactly the security-critical region. This document proposes ~20 improvements in priority order. The top four: (1) Renyi divergence estimation, (2) total variation distance, (3) formalized rejection-count independence, (4) sign/half-Gaussian decomposition tests. Each is described with full mathematical formulation, security-proof connection, and implementation sketch.

---

## Table of Contents

- [Context](#context)
- [Tier 1: Security-Critical Additions](#tier-1-security-critical-additions)
  - [1.1 Renyi Divergence Estimation](#11-renyi-divergence-estimation)
  - [1.2 Total Variation Distance](#12-total-variation-distance-with-dkw-confidence-interval)
  - [1.3 Rejection Count Independence](#13-formalized-rejection-count-independence)
  - [1.4 Sign Bit and Half-Gaussian Tests](#14-sign-bit-and-half-gaussian-distribution-tests)
- [Tier 2: Statistical Methodology Fixes](#tier-2-statistical-methodology-fixes)
  - [2.1 Multiple Testing Correction](#21-multiple-testing-correction)
  - [2.2 Power Analysis](#22-power-analysis-and-minimum-detectable-effect-size)
  - [2.3 Confidence Intervals on Moments](#23-confidence-intervals-on-sample-moments)
- [Tier 3: Additional Statistical Tests](#tier-3-additional-statistical-tests)
  - [3.1 Anderson-Darling (Discrete)](#31-discrete-anderson-darling-test)
  - [3.2 Henze-Zirkler (Multivariate)](#32-henze-zirkler-multivariate-normality-test)
  - [3.3 Squared Norm Distribution](#33-squared-norm-distribution-test)
  - [3.4 Two-Sample Cross-Implementation](#34-two-sample-cross-implementation-tests)
  - [3.5 Energy Test (Multivariate)](#35-energy-test-for-multivariate-normality)
  - [3.6 Discrete KS Test](#36-discrete-ks-test)
- [Tier 4: Independence and Sequence Analysis](#tier-4-independence-and-sequence-analysis)
  - [4.1 Autocorrelation / Ljung-Box](#41-serial-autocorrelation-ljung-box-test)
  - [4.2 Runs Test](#42-wald-wolfowitz-runs-test)
  - [4.3 Min-Entropy / Collision Entropy](#43-min-entropy-and-collision-entropy)
- [Tier 5: Lattice-Specific Structural Tests](#tier-5-lattice-specific-structural-tests)
  - [5.1 Circulant Covariance Structure](#51-circulantanti-circulant-covariance-structure)
  - [5.2 FFT-Domain Gaussianity](#52-fft-domain-gaussianity-testing)
  - [5.3 Message-Conditional Distribution](#53-message-conditional-distribution-test)
- [Tier 6: Engineering](#tier-6-engineering)
- [What NOT to Add](#what-not-to-add-and-why)
- [Recommended Implementation Order](#recommended-implementation-order)
- [Pitfalls](#pitfalls)
- [References](#references)

---

## Context

SAGA (Statistically Acceptable GAussians) is the reference statistical test suite for validating discrete Gaussian samplers in lattice-based signature schemes, particularly Falcon (now FN-DSA, FIPS 206). Published at PQCrypto 2020 [[1]](#references), it validates both univariate discrete Gaussian samplers and multivariate signature distributions from Falcon's FFT-based hash-and-sign construction.

### Current test inventory

| Domain | Test | Statistic | Null Distribution |
|---|---|---|---|
| Univariate | Pearson chi-square | $\sum (O_i - E_i)^2 / E_i$ | $\chi^2(k-1)$ |
| Univariate | Moment comparison | mean, std, skew, kurt | Point estimates only |
| Univariate | Outlier detection | count beyond $\tau\sigma$ | Hard threshold ($\tau=14$) |
| Multivariate | Doornik-Hansen [[2]](#references) | $\mathbf{z}_1^T\mathbf{z}_1 + \mathbf{z}_2^T\mathbf{z}_2$ | $\chi^2(2p)$ |
| Multivariate | diagcov (custom) | sum of normalized diagonal sums | $\chi^2(4(n/2-1))$ |
| Multivariate | Per-coordinate chi-sq | $p$ independent chi-square tests | No correction applied |
| Multivariate | Mardia (optional) [[3]](#references) | Generalized skew/kurt | $\chi^2$ / normal |

### Why revisit now

1. **Security proof mismatch.** Falcon's security proofs [[4, 5]](#references) use Renyi divergence, not normality -- SAGA tests a proxy, not the actual security property.
2. **Side-channel attacks.** Attacks in 2023-2025 [[6, 7, 8]](#references) exploit structural properties (sign leakage, rejection correlation) that SAGA does not test.
3. **Multiple testing problem.** The per-coordinate testing has an uncorrected family-wise error rate of up to 87% for Falcon-1024.
4. **Tail blind spot.** The chi-square test deliberately sacrifices tail sensitivity via bucket aggregation (`chi2_bucket = 10`), yet tails are the most security-critical region.

This analysis is ordered: **security impact > statistical rigor > usability > engineering**.

---

## Tier 1: Security-Critical Additions

### 1.1 Renyi Divergence Estimation

#### Background

The Renyi divergence of order $a > 1$ between distributions $P$ and $Q$ over a countable set $\mathcal{S}$ is:

$$R_a(P \| Q) = \frac{1}{a-1} \log \left( \sum_{x \in \mathcal{S}} \frac{P(x)^a}{Q(x)^{a-1}} \right)$$

with the limiting cases:

$$R_1(P \| Q) = D_{\mathrm{KL}}(P \| Q) \quad (a \to 1), \qquad R_\infty(P \| Q) = \log \max_x \frac{P(x)}{Q(x)} \quad (a \to \infty)$$

Bai, Langlois, Lepoint, Stehle, and Steinfeld [[4]](#references) established the **probability preservation property** for lattice-based cryptography: if an adversary has success probability $\varepsilon$ against a scheme instantiated with distribution $Q$, then the same adversary has success probability at most

$$\varepsilon' \geq \frac{\varepsilon}{R_a(P \| Q)^{1/(a-1)}}$$

against the scheme instantiated with distribution $P$. This is particularly suited for search problems (e.g., signature forgery).

Prest [[5]](#references) showed that using RD instead of statistical distance for Falcon's sampler analysis yields **~30 bits of additional security** in practice, and reduces the required CDT precision from ~266 bits to ~53 bits per entry (CDF+SD approach: ~5,300 bits total vs CoDF+RD approach: ~600 bits total).

Takashima and Takayasu [[9]](#references) further showed that **optimizing the order $a$** based on the adversary's advantage yields even tighter reductions.

#### What SAGA should test

For each univariate sample set with parameters $(\mu, \sigma)$, compute:

**$R_2$ (collision divergence):**

$$R_2(\hat{P} \| P_{\mathrm{ideal}}) = \log \left( \sum_x \frac{\hat{P}(x)^2}{P_{\mathrm{ideal}}(x)} \right)$$

**$R_\infty$ (max-divergence):**

$$R_\infty(\hat{P} \| P_{\mathrm{ideal}}) = \log \left( \max_x \frac{\hat{P}(x)}{P_{\mathrm{ideal}}(x)} \right)$$

where $\hat{P}(x) = \mathrm{count}(x) / n$ is the empirical PMF and $P_{\mathrm{ideal}}(x) = \rho_{\sigma,\mu}(x) / \sum_z \rho_{\sigma,\mu}(z)$ is the theoretical discrete Gaussian PMF already computed by `make_gaussian_pdt()`.

Compare these against the bounds required by Falcon's security proofs. For a sampler with standard deviation $\sigma$ and base standard deviation $\sigma_0 = 1.8205$, the security proof requires that $R_\infty(P_{\mathrm{actual}} \| P_{\mathrm{ideal}})$ is bounded by a value depending on $\sigma/\sigma_0$ and the CDT precision. The proof tolerates $R_\infty$ up to $\sim 2^{-64}$ for 128-bit security.

#### Finite-sample considerations

The plug-in estimator $\hat{R}_a$ is biased for finite $n$. For $R_2$, the bias is $O(1/n)$ and can be corrected:

$$\hat{R}_{2,\mathrm{corrected}} = \hat{R}_2 - \frac{k}{2n} + O(1/n^2)$$

where $k$ is the support size. For $R_\infty$, the plug-in estimator converges at rate $O(\sqrt{\log(k)/n})$ by the DKW inequality applied to the maximum.

For confidence intervals, use the bootstrap: resample from $\hat{P}$ with replacement $B=10{,}000$ times, compute $R_a$ on each bootstrap sample, and report the $(\alpha/2, 1-\alpha/2)$ quantiles.

#### Why this is the highest-priority addition

A sampler can pass every normality test in SAGA while having $R_\infty$ that violates the security proof's requirements. Consider a sampler that truncates the Gaussian at $\tau=10$ instead of $\tau=14$: the chi-square test with bucket aggregation would likely still pass (the aggregated tail bucket has few expected samples), but $R_\infty$ would blow up because $P_{\mathrm{ideal}}$ assigns nonzero probability to values that $\hat{P}$ assigns zero probability. Conversely, a sampler with a slight global bias might fail chi-square but have excellent RD bounds because the RD is forgiving of small additive perturbations when $P$ and $Q$ have similar support.

**Difficulty:** Medium. **Impact:** Very high.

---

### 1.2 Total Variation Distance with DKW Confidence Interval

#### Definition

The total variation distance between distributions $P$ and $Q$ is:

$$\mathrm{TV}(P, Q) = \frac{1}{2} \sum_x |P(x) - Q(x)| = \sup_A |P(A) - Q(A)|$$

#### Connection to security

In the statistical distance framework (pre-RD), the security loss from using an approximate sampler is bounded by:

$$|\Pr[\mathcal{A} \text{ wins with } P] - \Pr[\mathcal{A} \text{ wins with } Q]| \leq \mathrm{TV}(P, Q)$$

This is a weaker bound than RD but still widely used and more interpretable: if $\mathrm{TV}(\hat{P}, P_{\mathrm{ideal}}) = 2^{-128}$, the adversary gains at most $2^{-128}$ advantage from the sampler imperfection.

#### Confidence interval via DKW inequality

The Dvoretzky-Kiefer-Wolfowitz (1956) inequality [[10]](#references) bounds the convergence of the empirical CDF:

$$\Pr\!\left[ \sup_x |F_n(x) - F(x)| > \varepsilon \right] \leq 2 \exp(-2n\varepsilon^2)$$

For TV distance between a discrete empirical distribution and a known PMF, the analogous bound gives:

$$\Pr\!\left[ |\widehat{\mathrm{TV}} - \mathrm{TV}_{\mathrm{true}}| > \varepsilon \right] \leq 2|\mathcal{S}| \cdot \exp(-2n\varepsilon^2)$$

where $|\mathcal{S}|$ is the support size. Inverting: a 95% CI has half-width $\sqrt{\log(2|\mathcal{S}|/0.05) / (2n)}$.

For Falcon's sampler with $\sigma \approx 1.55$ and $\tau = 14$, $|\mathcal{S}| \approx 2 \cdot \lceil 14 \times 1.55 \rceil \approx 44$, so with $n = 10{,}000$ samples the 95% CI half-width is $\approx \sqrt{\log(1760) / 20{,}000} \approx 0.019$.

**Difficulty:** Low. **Impact:** High. Complementary to RD (additive vs multiplicative security metric).

---

### 1.3 Formalized Rejection Count Independence

#### Current state

`test_rejind()` calls `samplerz_rep()` which returns `(sample, rejection_count)` pairs, then visualizes the joint distribution as a heatmap. No statistical test is applied.

#### What to add

**(a) Chi-square test of independence:**

Construct the contingency table $C$ where $C[z][k]$ = count of outputs equal to $z$ with rejection count $k$. Under the null hypothesis $H_0$: $Z \perp K$:

$$E[C[z][k]] = \frac{\left(\sum_{k'} C[z][k']\right) \cdot \left(\sum_{z'} C[z'][k]\right)}{n}$$

Test statistic: $\sum_{z,k} (C[z][k] - E[z][k])^2 / E[z][k] \sim \chi^2((|Z|-1)(|K|-1))$

Available via `scipy.stats.chi2_contingency()`.

**(b) Mutual information estimation:**

$$I(Z; K) = \sum_{z,k} P(z,k) \log \frac{P(z,k)}{P(z) \cdot P(k)}$$

Under $H_0$: $I(Z;K) = 0$. The plug-in estimator has bias $O(|Z||K| / (2n))$. Use the Miller-Madow corrected estimator:

$$\hat{I}_{\mathrm{corrected}} = \hat{I}_{\mathrm{plugin}} + \frac{|Z||K| - |Z| - |K| + 1}{2n}$$

For a significance test, use the asymptotic result: $2n \cdot \hat{I} \sim \chi^2((|Z|-1)(|K|-1))$ under $H_0$.

**(c) Conditional homogeneity test:**

For each observed rejection count $k$, extract the conditional sample $\{z : \mathrm{rej\_count} = k\}$ and apply a chi-square goodness-of-fit test against the unconditional distribution $P(z)$. This is more sensitive than the omnibus independence test because it can identify which specific rejection count values are problematic.

#### Connection to side-channel attacks

The **SHIFT SNARE** attack (Qiu and Aysu, 2025) [[6]](#references) achieves 99.9999999478% per-coefficient success rate for Falcon-512 by exploiting correlations in the discrete Gaussian sampling implementation. Specifically, it targets the negation of the right-shift 63-bit operation, demonstrating that implementation choices in the rejection loop can expose intermediate value assignments.

Zhang et al. (EUROCRYPT 2023) [[7]](#references) reduced trace requirements from $\sim 10^6$ to $\sim 220{,}000$ by using **second-order statistics (covariance and spectral decomposition)** of the leakage, rather than fourth-moment parallelepiped learning. Their attack exploits the sign flip leakage within integer Gaussian sampling -- a leakage identified by Kim and Hong in 2018 but previously unexploited.

Lin et al. (2025) [[8]](#references) further reduced trace requirements by **85%** to $\sim 6{,}500$ traces by simultaneously exploiting half-Gaussian leakage and sign leakage. They validated countermeasures on ChipWhisperer.

All three attacks exploit **dependence between internal sampler state and output** -- exactly the property that formalized rejection independence tests would detect.

**Difficulty:** Medium. **Impact:** High.

---

### 1.4 Sign Bit and Half-Gaussian Distribution Tests

#### Sampler structure (from `sampler.py`)

Falcon's base sampler works in three steps:
1. Sample $z_0$ from a half-Gaussian on $\{0, 1, \ldots, 18\}$ via CDT lookup
2. Sample sign bit $b$ uniformly from $\{0, 1\}$
3. Compute $z = (2b - 1) \cdot z_0 + b$, giving $z \in \{-18, \ldots, -1, 0, 1, \ldots, 18\}$

Step 3 then undergoes rejection sampling via `berexp()` to produce the final sample.

#### Tests to add

**(a) Half-Gaussian distribution test:**

Extract $|z|$ from the output samples (adjusting for the center $\mu$). The distribution of $|z|$ should match the folded discrete Gaussian:

$$P(|z| = k) = P(z = k) + P(z = -k) \quad \text{for } k > 0, \qquad P(|z| = 0) = P(z = 0)$$

Apply chi-square goodness-of-fit to the empirical $|z|$ distribution against this PMF.

**(b) Sign independence test:**

Under correct sampling, for each $|z| > 0$:

$$P(\mathrm{sign}(z) = +1 \mid |z| = k) = \frac{P(z = k)}{P(|z| = k)}$$

For a centered Gaussian ($\mu = 0$), this should be exactly $1/2$. For non-centered Gaussians, the ratio depends on the center. Test via:
- For $\mu = 0$: binomial test on sign counts for each $|z|$
- For general $\mu$: chi-square on the joint (sign, $|z|$) table vs expected

**(c) Joint (sign, |z|) distribution test:**

Construct the $2 \times K$ contingency table (sign $\in \{+,-\}$, $|z| \in \{0,\ldots,K\}$) and test against the expected product distribution. More powerful than testing marginals separately because it directly checks the factorization $P(\mathrm{sign}, |z|) = P(\mathrm{sign} \mid |z|) \cdot P(|z|)$.

#### Why this matters structurally

The Lin et al. (2025) paper [[8]](#references) specifically measures two leakage types:
- **Half-Gaussian leakage:** The power trace reveals which CDT entry was selected (i.e., the value of $z_0$). With 27,500 traces using only half-Gaussian leakage, full key recovery succeeds on 14/40 instances.
- **Sign leakage:** The power trace reveals the sign bit $b$. With 25,000 traces using only sign leakage, full key recovery succeeds on 24/40 instances.

If a software implementation has a bug that creates correlation between sign and magnitude (e.g., due to branch prediction or cache timing), tests (b) and (c) would detect it even without power analysis equipment.

**Difficulty:** Low-Medium. **Impact:** High.

---

## Tier 2: Statistical Methodology Fixes

### 2.1 Multiple Testing Correction

#### The problem

SAGA's `MultivariateSamples` runs $p$ independent chi-square tests (one per coordinate) at significance level $\alpha = p_{\min} = 0.001$. The family-wise error rate (FWER) under the global null is:

$$\mathrm{FWER} = 1 - (1 - \alpha)^p$$

| Falcon variant | $p$ | FWER |
|---|---|---|
| Falcon-64 | 128 | 12.0% |
| Falcon-128 | 256 | 22.6% |
| Falcon-256 | 512 | 40.0% |
| Falcon-512 | 1024 | 64.2% |
| Falcon-1024 | 2048 | 87.1% |

For Falcon-1024, a **perfectly correct sampler** will be flagged as having "some failing coordinates" 87% of the time. The current code handles this informally by counting how many coordinates pass and expecting "most" to pass, but there is no formal threshold.

#### Solution: Benjamini-Hochberg (BH) procedure

The BH procedure [[11]](#references) controls the **false discovery rate** (FDR) rather than FWER. Given p-values $p_1 \leq p_2 \leq \cdots \leq p_m$ from $m$ independent tests:

1. Find the largest $k$ such that $p_k \leq (k/m) \cdot \alpha$
2. Reject $H_0$ for all tests $i = 1, \ldots, k$

This is preferred over Bonferroni (which controls FWER but is very conservative for large $m$) because:
- Bonferroni threshold: $\alpha/m = 0.001/2048 = 4.9 \times 10^{-7}$ -- this would require enormous sample sizes to achieve adequate power
- BH maintains FDR at $\alpha$ while preserving reasonable power

Implementation: collect all p-values from per-coordinate tests, apply `scipy.stats.false_discovery_control(pvalues, method='bh')` (scipy >= 1.11) or `statsmodels.stats.multitest.multipletests(pvalues, method='fdr_bh')`.

**Difficulty:** Low. **Impact:** Medium-High (essential for Falcon-512 and Falcon-1024).

---

### 2.2 Power Analysis and Minimum Detectable Effect Size

#### Chi-square power

For a chi-square goodness-of-fit test with $k$ cells, significance level $\alpha$, and $n$ samples, the power against an alternative with effect size $w$ (Cohen's $w$) is:

$$w = \sqrt{ \sum_i \frac{(P_i - Q_i)^2}{Q_i} }$$

The non-centrality parameter is $\lambda = n \cdot w^2$, and:

$$\mathrm{Power} = 1 - F_{\chi^2(k-1, \lambda)}\!\left(F_{\chi^2(k-1)}^{-1}(1-\alpha)\right)$$

where $F_{\chi^2(\cdot)}$ and $F_{\chi^2(\cdot)}^{-1}$ are the CDF and quantile function of the (non-central) chi-square distribution.

**Example for Falcon's sampler:** With $\sigma = 1.55$, the support size is $|\mathcal{S}| \approx 44$ values. With $\alpha = 0.001$ and $n = 10{,}000$ samples, the minimum detectable effect size (at power 0.8) works out to approximately $w_{\min} \approx 0.095$ (medium effect by Cohen's convention).

**Connecting power to security:** The chi-square effect size $w$ relates to TV distance approximately as:

$$\mathrm{TV}(P, Q) \leq w \cdot \sqrt{\max_i Q_i}$$

For a discrete Gaussian with $\sigma = 1.55$, $\max P_{\mathrm{ideal}}(x) \approx 0.26$ (the mode), giving $\mathrm{TV}_{\min} \approx 0.095 \times 0.51 \approx 0.048$. This means the test can detect distributional errors that give the adversary ~5% additional advantage.

#### Doornik-Hansen / diagcov power

No closed-form power analysis exists. Use Monte Carlo simulation:
1. Generate $m = 1000$ synthetic datasets under a specified alternative (e.g., Gaussian with perturbed covariance)
2. Apply the test to each
3. Report the proportion of rejections as estimated power

**Difficulty:** Medium. **Impact:** Medium-High (prevents false confidence with insufficient samples).

---

### 2.3 Confidence Intervals on Sample Moments

For a sample of size $n$ from a distribution with finite fourth moment:

| Moment | Point Estimator | Standard Error | 95% CI |
|---|---|---|---|
| Mean | $\bar{x}$ | $\sigma / \sqrt{n}$ | $\bar{x} \pm 1.96 \cdot \sigma / \sqrt{n}$ |
| Variance | $s^2$ | $\sigma^2 \sqrt{2/(n-1)}$ | $\left(\frac{(n-1)s^2}{\chi^2_{0.975,n-1}},\; \frac{(n-1)s^2}{\chi^2_{0.025,n-1}}\right)$ |
| Skewness | $g_1$ | $\sqrt{6/n}$ | $g_1 \pm 1.96 \cdot \sqrt{6/n}$ |
| Kurtosis (excess) | $g_2$ | $\sqrt{24/n}$ | $g_2 \pm 1.96 \cdot \sqrt{24/n}$ |

The discretization correction for variance is well-characterized: $\mathrm{Var}(D_{\mathbb{Z},\sigma,\mu}) = \sigma^2 - \varepsilon(\sigma)$ where $\varepsilon(\sigma)$ is exponentially small in $\sigma$ for $\sigma \geq 1$ (see Micciancio and Regev, Lemma 4.2 [[12]](#references)).

**Difficulty:** Low. **Impact:** Medium.

---

## Tier 3: Additional Statistical Tests

### 3.1 Discrete Anderson-Darling Test

#### Continuous AD statistic

The Anderson-Darling (1952) [[13]](#references) statistic is an EDF-based test:

$$A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} (2i-1) \left[ \ln F_0(x_{(i)}) + \ln(1 - F_0(x_{(n+1-i)})) \right]$$

where $x_{(i)}$ are the order statistics and $F_0$ is the hypothesized CDF. The weighting by $1/(F(x)(1-F(x)))$ gives enhanced sensitivity in the tails compared to Kolmogorov-Smirnov (uniform weighting) or Cramer-von Mises (unit weighting).

#### Discrete adaptation

For discrete distributions, Choulakian, Lockhart, and Stephens (1994) [[14]](#references) developed the discrete analogue. Let the discrete distribution have $k$ cells with probabilities $p_1, \ldots, p_k$. Define cumulative probabilities $H_j = \sum_{i=1}^{j} p_i$. Let $Z_j = S_j - T_j$ where $S_j = \sum_{i=1}^{j} O_i$ (observed cumulative) and $T_j = n \cdot H_j$ (expected cumulative). The discrete AD statistic is:

$$A^2_{\mathrm{discrete}} = \frac{1}{n} \sum_{j=1}^{k-1} \frac{Z_j^2 \cdot p_{j+1}}{H_j (1 - H_j)}$$

The asymptotic distribution under $H_0$ is **not** chi-square -- it depends on the specific null distribution. Critical values must be obtained by simulation.

#### Why this matters for Falcon

The current chi-square test uses `chi2_bucket = 10`: any cell with $E_i < 10$ expected counts is merged with adjacent cells. For $\sigma = 1.55$ and $n = 10{,}000$, cells beyond $|z| \approx 5$ have fewer than 10 expected counts and get merged. This means deviations in the range 5$\sigma$ to 14$\sigma$ are aggregated into a single "tail bucket" and tested as a single unit.

But this tail region is exactly where security-critical deviations would appear:
- A sampler that truncates at 10$\sigma$ instead of 14$\sigma$ would produce zero samples beyond 10$\sigma$. The merged tail bucket would still likely pass (expected count < 10, merged with the 5-10$\sigma$ bucket).
- The AD statistic, by weighting tails more heavily, would flag this.

Consider: for $|z| = 8$ with $\sigma = 1.55$, $P(|z| \geq 8) \sim \exp(-8^2/(2 \cdot 1.55^2)) \sim \exp(-13.3) \sim 1.7 \times 10^{-6}$. With $n = 10{,}000$, the expected count is 0.017 -- chi-square merges this into the tail bucket. But the AD weighting $1/(H(1-H)) \sim 1/(1 \times 0.000017) \sim 59{,}000$ amplifies any deviation at this point.

**Practical approach:** Since the null distribution of $A^2_{\mathrm{discrete}}$ depends on the specific $(\sigma, \mu)$, compute p-values by Monte Carlo: generate $M = 10{,}000$ samples from the theoretical discrete Gaussian, compute $A^2$, repeat $B = 1{,}000$ times, and use the empirical distribution as the null.

**Difficulty:** Medium. **Impact:** High (specifically targets the chi-square tail blind spot).

---

### 3.2 Henze-Zirkler Multivariate Normality Test

#### Test statistic

The Henze-Zirkler (1990) [[15]](#references) test is based on the BHEP (Baringhaus-Henze-Epps-Pulley) functional:

$$\mathrm{HZ}_\beta = n \int \left|\psi_n(\mathbf{t}) - \psi_0(\mathbf{t})\right|^2 w_\beta(\mathbf{t})\, d\mathbf{t}$$

where $\psi_n(\mathbf{t})$ is the empirical characteristic function, $\psi_0(\mathbf{t}) = \exp(-\|\mathbf{t}\|^2/2)$ is the CF of $N(\mathbf{0}, \mathbf{I}_p)$, and $w_\beta(\mathbf{t}) = (2\pi\beta^2)^{-p/2} \exp(-\|\mathbf{t}\|^2/(2\beta^2))$ is a Gaussian kernel weight.

After integration, this has the closed form:

$$\mathrm{HZ} = \frac{1}{n} \sum_{i,j} \exp\!\left(-\frac{\beta^2}{2} \|\mathbf{Y}_i - \mathbf{Y}_j\|^2\right) - 2(1+\beta^2)^{-p/2} \cdot \frac{1}{n} \sum_i \exp\!\left(-\frac{\beta^2}{2(1+\beta^2)} \|\mathbf{Y}_i\|^2\right) + (1+2\beta^2)^{-p/2}$$

where $\mathbf{Y}_i = \mathbf{S}^{-1/2}(\mathbf{X}_i - \bar{\mathbf{X}})$ are the standardized observations and $\mathbf{S}$ is the sample covariance matrix.

#### Smoothing parameter

Henze and Zirkler proposed:

$$\beta = \frac{1}{2p} \left( \frac{(2p+1)n}{4} \right)^{1/(p+4)}$$

As $\beta \to 0$, HZ converges to a radial test based on $\|\mathbf{Y}_i\|^2$ alone. As $\beta \to \infty$, HZ converges to a weighted combination of Mardia's skewness and kurtosis measures. The recommended $\beta$ interpolates between these extremes.

#### Null distribution

Under $H_0$, the HZ statistic is approximately log-normal. The p-value is computed as $P(\mathrm{LogNormal}(\mu_{\mathrm{LN}}, \sigma^2_{\mathrm{LN}}) > \mathrm{HZ}_{\mathrm{observed}})$.

#### Power profile comparison with Doornik-Hansen

| Alternative | DH sensitivity | HZ sensitivity |
|---|---|---|
| Skewness shift | High (direct transform) | Medium (indirect via ECF) |
| Kurtosis shift | High (direct transform) | Medium |
| Multimodality | Low | High (ECF detects modes) |
| Tail weight change | Low-Medium | High |
| Correlation structure | Low | Medium-High |
| Radial departure | Low | Depends on $\beta$ |

Key insight: DH is most powerful against moment-based departures (skew/kurtosis). HZ is most powerful against departures that affect the characteristic function globally (multimodality, tail weight, correlation changes). For Falcon, correlation structure changes (from FFT tree bugs) are a primary concern, favoring HZ as a complement.

Research consensus (Mecklin and Mundfrom, 2005 [[16]](#references); Henze, Jimenez-Gamero, and Meintanis, 2020 [[17]](#references)): "the uniformly most powerful MVN test does not exist; it is recommended to perform several tests." DH + HZ covers the widest range of alternatives with minimal implementation overlap.

**Difficulty:** Low-Medium (~80 lines from the closed form, or extract from `pingouin`). **Impact:** Medium-High.

---

### 3.3 Squared Norm Distribution Test

#### Theory

If $\mathbf{X} \sim N(\mathbf{0}, \sigma^2 \mathbf{I}_p)$, then:

$$\frac{\|\mathbf{X}\|^2}{\sigma^2} \sim \chi^2(p)$$

For Falcon signatures, the covariance is not exactly $\sigma^2 \mathbf{I}_p$ but has the NTRU lattice structure. However, under the assumption of independent coordinates (which diagcov already tests), $\|\mathrm{sig}\|^2 / \sigma^2$ should follow $\chi^2(p)$.

#### Implementation

```python
norms_sq = np.sum(data**2, axis=1) / sigma**2
stat, pvalue = scipy.stats.kstest(norms_sq, 'chi2', args=(p,))
```

For large $p$ (Falcon-512: $p=1024$), use the CLT approximation:

$$\frac{\|\mathrm{sig}\|^2/\sigma^2 - p}{\sqrt{2p}} \overset{d}{\approx} N(0, 1)$$

#### Complementarity

This is **orthogonal** to per-coordinate testing:
- Per-coordinate passes, norm fails $\to$ correct marginals but positive correlation (inflating the norm)
- Per-coordinate fails, norm passes $\to$ marginal deviations cancel in the sum (unlikely but possible)

Also complementary to diagcov, which tests off-diagonal covariance structure but not the aggregate effect on norms.

**Difficulty:** Very low. **Impact:** Medium.

---

### 3.4 Two-Sample Cross-Implementation Tests

#### Motivation

SAGA tests each Falcon implementation (AVX2, fpnative, fpemu) independently against the theoretical distribution. Two-sample tests comparing implementations directly are often **more powerful** because:

1. They do not require knowing the theoretical distribution exactly (the discrete Gaussian PMF involves a normalizing constant).
2. The detection threshold is $\sim 1/\sqrt{2} \approx 71\%$ the size of one-sample tests.
3. Implementation bugs often affect specific implementations. If AVX2 has a subtle rounding error that fpnative doesn't, the two-sample test directly detects this.

#### Tests

```python
# Two-sample KS
stat, pvalue = scipy.stats.ks_2samp(samples_avx2, samples_fpnative)

# Two-sample Cramer-von Mises (scipy >= 1.7)
result = scipy.stats.cramervonmises_2samp(samples_avx2, samples_fpnative)
```

For multivariate signature data, use the **energy distance**:

$$\mathcal{E}(P, Q) = \frac{2}{nm} \sum_{i,j} \|\mathbf{X}_i - \mathbf{Y}_j\| - \frac{1}{n^2} \sum_{i,j} \|\mathbf{X}_i - \mathbf{X}_j\| - \frac{1}{m^2} \sum_{i,j} \|\mathbf{Y}_i - \mathbf{Y}_j\|$$

For 3 implementations, apply BH correction to the $\binom{3}{2} = 3$ p-values.

**Difficulty:** Low. **Impact:** Medium. **Prerequisite:** scipy >= 1.7.

---

### 3.5 Energy Test for Multivariate Normality

The Szekely-Rizzo (2005) [[18]](#references) energy test is **consistent against all fixed alternatives** with finite first moment, affine invariant, and well-defined for $p > n$.

However, the statistic requires $O(n^2 p)$ operations. For Falcon-512 with $n = 64{,}000$ and $p = 1024$: $\sim 4 \times 10^{12}$ operations -- **prohibitively expensive** for routine testing.

**Recommendation:** Defer; HZ + DH provide adequate MVN coverage. Consider random-projection approximation if resources permit.

**Difficulty:** High. **Impact:** Medium.

---

### 3.6 Discrete KS Test

**Recommendation: Skip.** Anderson-Darling (3.1) provides strictly more tail sensitivity than KS, and tails are the security-critical region. KS's advantage (distribution-free null for continuous data) does not apply to discrete Gaussians. See Arnold and Emerson (2011) [[19]](#references) for the discrete case if needed.

---

## Tier 4: Independence and Sequence Analysis

### 4.1 Serial Autocorrelation (Ljung-Box Test)

#### Motivation

All existing SAGA tests evaluate the **marginal** or **joint (across coordinates)** distribution but none test **sequential** properties. If a sampler produces $z_1, z_2, \ldots, z_n$, SAGA verifies the empirical distribution of $\{z_i\}$ matches the target, but not that $z_i \perp z_{i-1}$.

Constant-time implementations could introduce serial dependence through shared PRNG state, cache effects, or branch predictor leakage.

#### Ljung-Box test

The Ljung-Box (1978) [[20]](#references) test statistic for serial correlation up to lag $h$ is:

$$Q(h) = n(n+2) \sum_{k=1}^{h} \frac{r_k^2}{n-k}$$

where $r_k$ is the sample autocorrelation at lag $k$. Under $H_0$ (independence), $Q(h) \sim \chi^2(h)$.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
result = acorr_ljungbox(samples, lags=20, return_df=True)
```

#### Complementarity with rejection independence (1.3)

Test 1.3 checks independence between output and internal state (rejection count). Test 4.1 checks independence between successive outputs. These are different properties.

**Difficulty:** Very low. **Impact:** Medium (catches a class of bugs no other SAGA test detects).

---

### 4.2 Wald-Wolfowitz Runs Test

The runs test checks whether the sequence of above/below-median outputs forms a random pattern. Let $n_1$ = count above median, $n_2$ = count below, $R$ = number of runs. Under $H_0$:

$$E[R] = 1 + \frac{2n_1 n_2}{n_1+n_2}, \qquad \mathrm{Var}[R] = \frac{2n_1 n_2(2n_1 n_2 - n_1 - n_2)}{(n_1+n_2)^2(n_1+n_2-1)}$$

$$Z = \frac{R - E[R]}{\sqrt{\mathrm{Var}[R]}} \sim N(0,1)$$

Too few runs: positive autocorrelation. Too many: negative autocorrelation/alternation.

**Difficulty:** Very low (~20 lines). **Impact:** Low-Medium.

---

### 4.3 Min-Entropy and Collision Entropy

$$H_\infty(P) = -\log_2(\max_x P(x)), \quad H_2(P) = -\log_2\!\left(\sum_x P(x)^2\right), \quad H_1(P) = -\sum_x P(x) \log_2 P(x)$$

A sampler with correct mean and variance but concentrated on too few values (e.g., PRNG with short period) could pass moment tests while having catastrophically low entropy. For Falcon's sampler with $\sigma \approx 1.55$, the theoretical $H_\infty \approx 1.94$ bits. The test detects **anomalous entropy reduction** relative to this value.

**Difficulty:** Very low. **Impact:** Low-Medium (cheap defense-in-depth).

---

## Tier 5: Lattice-Specific Structural Tests

### 5.1 Circulant/Anti-Circulant Covariance Structure

Falcon uses NTRU lattices in $\mathbb{Z}_q[x]/(x^n+1)$. The covariance matrix has a 2x2 block structure where each $n \times n$ block is **negacyclic**. The existing `diagcov` test checks diagonal sums across quadrants but not circulant structure within blocks.

A negacyclic matrix $M$ is diagonalized by the DFT matrix over $\mathbb{Q}(\zeta_{2n})$:

$$\mathbf{F}_n \mathbf{M} \mathbf{F}_n^{-1} = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$$

**Proposed test:** Apply DFT to each covariance block, check that off-diagonal elements in the transformed matrix are near zero:

$$\mathrm{offdiag\_norm} = \frac{\| \mathbf{\Sigma}_{\mathrm{DFT}} - \mathrm{diag}(\mathrm{diag}(\mathbf{\Sigma}_{\mathrm{DFT}})) \|_F}{\| \mathbf{\Sigma}_{\mathrm{DFT}} \|_F}$$

**Difficulty:** Medium-High. **Impact:** Medium (narrow but important class of FFT tree bugs).

---

### 5.2 FFT-Domain Gaussianity Testing

Falcon's sampler (`ffsampling.py`) works in the FFT representation. Bugs in the FFT tree would naturally manifest in specific frequency coefficients.

**Test:** Compute DFT of each signature vector. FFT coefficients should be independent complex Gaussians. Split into real/imaginary parts, test Gaussianity of each, test independence between Re and Im at each frequency, test independence across frequencies.

FFT tree bugs that create subtle time-domain correlations may produce dramatic frequency-domain failures (a butterfly bug affects specific frequencies).

**Difficulty:** Medium. **Impact:** Medium.

---

### 5.3 Message-Conditional Distribution Test

For a hash-and-sign scheme, the distribution should satisfy $P(\mathrm{sig} \mid \mathrm{msg}) = \rho_{\sigma}(\mathrm{sig} - H(\mathrm{msg})) / Z$. Generate $K = 100$ signatures for $M = 10$ messages, center by subtracting $H(\mathrm{msg})$, apply $K$-sample test across groups.

**Difficulty:** Medium. **Impact:** Low-Medium.

---

## Tier 6: Engineering

| Item | Description |
|---|---|
| **6.1 Dependency updates** | scipy 1.3.1 $\to$ >= 1.11, numpy 1.17.3 $\to$ >= 1.24. **Prerequisite for 2.1, 3.1, 3.2, 3.4.** |
| **6.2 Pytest migration** | Convert ad-hoc `test_*()` functions to pytest with assertions, `@pytest.mark.parametrize`, and fixtures. |
| **6.3 CI/CD** | GitHub Actions: install deps, run pytest on included test data. Full suite nightly (test data is ~1GB). |
| **6.4 Structured output** | Add `to_dict()` / `to_json()` to `UnivariateSamples` and `MultivariateSamples` for batch analysis. |
| **6.5 Configurable parameters** | Make `tau`, `chi2_bucket`, `pmin`, `sigma0` configurable via CLI or config file. |

---

## What NOT to Add (and Why)

| Suggestion | Why Skip | Reference |
|---|---|---|
| **Royston test** | Power unknown for $p > 20$. Falcon uses $p \geq 128$. HZ is superior for high-dimensional MVN. | [[16]](#references) |
| **Cramer-von Mises (univariate)** | AD provides strictly greater tail sensitivity via $1/(F(1-F))$ weighting. CvM is uniform-weighted. | [[14]](#references) |
| **NIST SP 800-22** | Designed for uniform binary sequences. Null hypothesis is uniformity on $\{0,1\}^n$, not Gaussianity on $\mathbb{Z}$. Autocorrelation + Runs (Tier 4) capture the useful sequential tests. | [[21]](#references) |
| **TestU01** | Same problem as NIST: uniform $[0,1)$ tests. C FFI overhead. 160+ tests almost entirely irrelevant to discrete Gaussians. | [[22]](#references) |
| **GLITCH hyperskewness/hyperkurtosis** | Deliberately dropped in SAGA. Chi-square captures all moment deviations. Higher moments have very large SEs: $\mathrm{SE}(m_5) \sim \sqrt{720/n}$. | [[23]](#references) |
| **Sequential/online testing** | SAGA's use case is batch testing. SPRT (Wald, 1945) is optimal for online detection but adds engineering complexity without matching the use case. | |
| **Mardia optimization** | $O(n^2 p)$ complexity, impractical for $p=1024$. DH + HZ cover MVN at $O(np^2)$ and $O(n^2)$. Keep as optional for small cases. | [[3]](#references) |

---

## Recommended Implementation Order

| Phase | Items | Rationale |
|---|---|---|
| **1: Foundation** | 6.1 (deps), 6.2 (pytest) | Unblocks everything else |
| **2: Security-critical** | 1.1 (Renyi), 1.2 (TV), 1.4 (sign/half-Gaussian), 2.1 (BH correction) | Security proof alignment + recent attack vectors |
| **3: Coverage** | 3.1 (Anderson-Darling), 3.2 (Henze-Zirkler), 3.3 (norm test), 1.3 (rejection independence) | Fills power gaps with complementary tests |
| **4: Depth** | 4.1 (autocorrelation), 3.4 (two-sample), 2.2 (power analysis), 2.3 (moment CIs) | Independence checks and interpretability |
| **5: Structure** | 5.1 (circulant covariance), 5.2 (FFT-domain), 6.3 (CI/CD) | Lattice-specific and infrastructure |

---

## Pitfalls

1. **Over-testing without correction.** Each test at $\alpha=0.001$ adds ~0.1% false positive rate. With 20 tests, the chance a correct sampler fails at least one is ~2%. Apply BH correction globally.
2. **Discrete vs continuous misapplication.** AD, KS, CvM are continuous-distribution tests. Applying to discrete Gaussians yields conservative (too large) p-values, reducing power. Always prefer discrete-adapted versions.
3. **Computational cost.** HZ: $O(n^2)$; energy test: $O(n^2 p)$; Mardia: $O(n^2 p)$. Benchmark before adding to default battery.
4. **Finite-sample bias.** The plug-in estimator for $R_2(P\|Q)$ is biased upward by $O(k/n)$. For $k=44$, $n=10{,}000$: bias $\sim 0.004$. Use bias-corrected estimators or bootstrap CIs.

---

## References

1. J. Howe, T. Prest, T. Ricosset, M. Rossi. "[Isochronous Gaussian Sampling: From Inception to Implementation](https://eprint.iacr.org/2019/1411)." PQCrypto 2020.
2. J. A. Doornik, H. Hansen. "[An Omnibus Test for Univariate and Multivariate Normality](https://doi.org/10.1111/j.1468-0084.2008.00537.x)." Oxford Bulletin of Economics and Statistics 70(5), 2008.
3. K. V. Mardia. "Measures of Multivariate Skewness and Kurtosis with Applications." Biometrika 57(3):519-530, 1970.
4. S. Bai, A. Langlois, T. Lepoint, D. Stehle, R. Steinfeld. "[Improved Security Proofs in Lattice-Based Cryptography: Using the Renyi Divergence Rather than the Statistical Distance](https://doi.org/10.1007/s00145-017-9265-9)." ASIACRYPT 2015 / J. Cryptology 31(2):610-640, 2018.
5. T. Prest. "[Sharper Bounds in Lattice-Based Cryptography Using the Renyi Divergence](https://eprint.iacr.org/2017/480)." ASIACRYPT 2017.
6. J. Qiu, A. Aysu. "[SHIFT SNARE: Uncovering Secret Keys in FALCON via Single-Trace Analysis](https://eprint.iacr.org/2025/146)." ePrint 2025/146.
7. S. Zhang, X. Lin, Y. Yu, W. Wang. "[Improved Power Analysis Attacks on Falcon](https://eprint.iacr.org/2023/224)." EUROCRYPT 2023.
8. X. Lin, S. Zhang, Y. Yu, W. Wang, et al. "[Thorough Power Analysis on Falcon Gaussian Samplers and Practical Countermeasure](https://eprint.iacr.org/2025/351)." ePrint 2025/351.
9. K. Takashima, A. Takayasu. "Tighter Security for Efficient Lattice Cryptography via the Renyi Divergence of Optimized Orders." ACISP 2015.
10. A. Dvoretzky, J. Kiefer, J. Wolfowitz. "Asymptotic Minimax Character of the Sample Distribution Function and of the Classical Multinomial Estimator." Annals of Mathematical Statistics 27(3):642-669, 1956.
11. Y. Benjamini, Y. Hochberg. "[Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)." J. Royal Statistical Society B 57(1):289-300, 1995.
12. D. Micciancio, O. Regev. "[Worst-Case to Average-Case Reductions Based on Gaussian Measures](https://doi.org/10.1137/S0097539705447360)." SIAM J. Computing 37(1):267-302, 2007.
13. T. W. Anderson, D. A. Darling. "Asymptotic Theory of Certain 'Goodness of Fit' Criteria Based on Stochastic Processes." Annals of Mathematical Statistics 23(2):193-212, 1952.
14. V. Choulakian, R. A. Lockhart, M. A. Stephens. "Cramer-von Mises Statistics for Discrete Distributions." Canadian J. Statistics 22(1):125-137, 1994.
15. N. Henze, B. Zirkler. "A Class of Invariant Consistent Tests for Multivariate Normality." Comm. Statist. Theory Methods 19(10):3595-3617, 1990.
16. C. J. Mecklin, D. J. Mundfrom. "A Monte Carlo Comparison of the Type I and Type II Error Rates of Tests of Multivariate Normality." J. Stat. Computation and Simulation 75(2):93-107, 2005.
17. N. Henze, M. D. Jimenez-Gamero, S. G. Meintanis. "[Tests for Multivariate Normality -- A Critical Review with Emphasis on Weighted $L^2$-Statistics](https://doi.org/10.1007/s11749-020-00740-0)." TEST 29:845-870, 2020.
18. G. J. Szekely, M. L. Rizzo. "A New Test for Multivariate Normality." J. Multivariate Analysis 93(1):58-80, 2005.
19. T. B. Arnold, J. W. Emerson. "Nonparametric Goodness-of-Fit Tests for Discrete Null Distributions." The R Journal 3(2):34-39, 2011.
20. G. M. Ljung, G. E. P. Box. "On a Measure of Lack of Fit in Time Series Models." Biometrika 65(2):297-303, 1978.
21. A. Rukhin et al. "[A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications](https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final)." NIST SP 800-22 Rev. 1a, 2010.
22. P. L'Ecuyer, R. Simard. "[TestU01: A C Library for Empirical Testing of Random Number Generators](https://doi.org/10.1145/1268776.1268777)." ACM Trans. Math. Software 33(4), 2007.
23. J. Howe, M. O'Neill. "[GLITCH: A Discrete Gaussian Testing Suite for Lattice-Based Cryptography](https://eprint.iacr.org/2017/438)." SECRYPT 2017.
