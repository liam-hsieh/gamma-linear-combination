# `toolbox.stats.GammaLinearCombination` (`gamma-linear-combination`)

A modern, `uv`-managed Python package implementing [**`toolbox.stats.GammaLinearCombination`**](https://liam-hsieh.github.io/liamhsieh-toolbox/toolbox/stats/gamma_linear_combination/) for computing the probability density (`pdf`), cumulative probability (`cdf`), quantiles (`percentile` / `ppf`), random variates (`rv` / `rvs`), and visualization (`plot_pdf`) of a **linear combination of independent Gamma distributed random variables**:

$$Y = \sum_{i=1}^n c_i X_i + L, \qquad X_i \overset{\text{ind.}}{\sim} \Gamma(\alpha_i, \beta_i), \quad c_i \in \mathbb{R}_+$$

In addition to the exact **Moschopoulos (1985)** single gamma-series expansion described in the original specification, this project incorporates recent methods from statistical literature (**Barnabani 2017**, **Ha & Provost 2013**, **Hu, Pozdnyakov, & Yan 2020**) and **10 interchangeable computational modes** so users can select the optimal balance between **effectiveness (accuracy)** and **efficiency (execution speed & scalability)**.

---

## Literature Study & Engineering Advances

### 1. New Mathematical Approaches Beyond Moschopoulos (1985)
- **Barnabani (2017) & Vellaisamy–Upadhye (2009) Generalized Negative Binomial (GNB) Mixture (`method="barnabani"`)**:
  - *References*: Vellaisamy & Upadhye (2009, *J. Appl. Probab.*); Barnabani (2017, *Commun. Stat. Simul. Comput.*); Hu, Pozdnyakov, & Yan (2020, *Comput. Stat.*, R package `coga`).
  - *Formulation*: The normalized Moschopoulos weights $w_k = C \delta_k$ define the exact probability mass function of a discrete random variable $K$ (a convolution of Negative Binomial variables). Rather than executing the $O(K^2)$ recursive relation for $\delta_k$, Barnabani matches the exact first three cumulants $(\mu, \mu_2, \mu_3)$ of $K$ using Jain & Consul's (1971) 3-parameter **Generalized Negative Binomial $\text{GNB}(k; r, p, b)$** (or 2-parameter $\text{NB}(k; r, p)$ when $A < 2$), evaluating series weights in **$O(1)$ closed form per term** and cutting initialization latency by up to $15\times$ for large $n$.
- **Ha & Provost (2013) Generalized Laguerre Orthogonal Polynomial Expansion (`method="laguerre"`)**:
  - *References*: Provost (2005); Ha & Provost (2013, *REVSTAT*).
  - *Formulation*: Expands the density around the 2-moment Welch–Satterthwaite baseline Gamma density $g(y; \alpha_0, \beta_0)$ using generalized Laguerre polynomials $L_k^{(\alpha_0 - 1)}(y/\beta_0)$ up to order $M = 6$. By deriving coefficients from the **cumulant-excess generating function** $d_r = (\tilde{\kappa}_r - (r-1)!\alpha_0)/r!$ ($r = 3, \dots, 6$), catastrophic cancellation is completely avoided, and Rodrigues' formula yields an **exact closed-form finite CDF** requiring only a single incomplete gamma evaluation:
    $$F(y) = P(\alpha_0, z) + \frac{z^{\alpha_0} e^{-z}}{\Gamma(\alpha_0)} \sum_{k=3}^6 \frac{c_k}{k} L_{k-1}^{(\alpha_0)}(z), \qquad z = \frac{y - L}{\beta_0}$$

### 2. Engineering Optimizations for Exact Moschopoulos Series (`method="moschopoulos"`)
1. **Incomplete-Gamma Forward Recurrence (Abramowitz & Stegun 6.5.21; Hu, Pozdnyakov, & Yan 2020)**:
   Instead of calling `scipy.special.gammainc(rho + k, z)` $K + 1$ times per point, `MoschopoulosEngine` evaluates `gammainc` **only once** at $k = k_{\min}$ and propagates subsequent incomplete gamma values via:
   $$P(a + 1, z) = P(a, z) - \frac{z^a e^{-z}}{\Gamma(a + 1)}, \qquad p_{a+1}(z) = p_a(z) \cdot \frac{z}{a + 1}$$
   accelerating exact CDF evaluation by **$3.5\times$ to $13\times$**.
2. **Scale Coalescing (Sufficiency Reduction)**: Automatically coalesces components with identical effective scales $\tilde{\beta}_i$ via $\Gamma(\alpha_1, \beta) + \Gamma(\alpha_2, \beta) \sim \Gamma(\alpha_1 + \alpha_2, \beta)$ prior to series expansion.
3. **Dynamic Exponent-Rescaled BLAS Recurrence & Active Window Trimming $[k_{\min}, k_{\max}]$**: Uses native float64 `np.dot` with dynamic $10^{-200}$ exponent shifting so $\delta_k$ never overflows and $C = \prod (\beta_1/\beta_i)^{\alpha_i}$ never underflows even for $n = 150+$ wafer fab segments, while restricting evaluation to the active support window $[k_{\min}, k_{\max}]$.
4. **Analytical-Derivative Safeguarded Newton–Raphson PPF**: Uses simultaneous $(F(y), f(y))$ evaluations to converge quantile queries in 3–4 iterations.

---

## Effectiveness vs. Efficiency Trade-off Guide (All 10 Modes)

| Method (`method=...`) | Init Complexity | Query Speed (`pdf`/`cdf`/`percentile`) | Accuracy (Low Scale Spread $\le 6\times$) | Accuracy (High Scale Spread $\ge 20\times$) | Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`"moschopoulos"`** | $O(n K + K^2)$ | **Fast** (1 `gammainc` + recurrence) | **Exact** ($\sim 10^{-13}$) | High (with adaptive $K$) | Default exact engine for standard combinations |
| **`"moschopoulos_quad"`** | $O(n K + K^2)$ | Moderate (`scipy.integrate.quad`) | **Exact** ($\sim 10^{-12}$) | High | Direct numerical quadrature verification |
| **`"barnabani"`** | **$O(n + K)$** | **Fast** (closed-form GNB + recurrence) | **Very High** ($\sim 10^{-5}$) | **High** ($\sim 10^{-4}$) | Fast gamma-mixture without $O(K^2)$ $\delta_k$ recurrence ($15\times$ faster init for large $n$) |
| **`"laguerre"`** | **$O(n)$** | **Ultra-Fast** ($O(1)$ 6-cumulant closed form) | **Very High** ($\sim 10^{-5}$) | Moderate ($\sim 10^{-3}$) | Sub-millisecond analytical PDF/CDF matching first 6 cumulants |
| **`"saddlepoint"`** | $O(n)$ | Fast ($O(n)$ Newton–Raphson) | **Very High** ($< 0.03\%$ err) | **Very High** ($< 0.5\%$ err) | Wafer fab cycle time ($n \ge 100$ steps) or extreme scale ratios |
| **`"fft"`** | $O(n M + G M)$ | **Ultra-Fast** ($O(1)$ spline lookup) | **High** ($\sim 10^{-6}$) | **High** ($\sim 10^{-6}$) | Repeated batch queries ($> 10{,}000$ PDF/CDF/PPF evaluations) |
| **`"moment_matching"`** | $O(n)$ | **Instant** ($O(1)$ closed-form) | **Good** ($< 0.05\%$ P95 err) | **Good** ($< 0.9\%$ P95 err) | 3-moment Pearson Type III closed-form approximation |
| **`"welch_satterthwaite"`**| $O(n)$ | **Instant** ($O(1)$ closed-form) | Moderate ($< 0.5\%$ P95 err) | Moderate | Classical 2-moment baseline |
| **`"monte_carlo"`** | $O(n N_{\text{sim}})$ | Fast ($O(\log N_{\text{sim}})$) | $O(1/\sqrt{N_{\text{sim}}})$ | $O(1/\sqrt{N_{\text{sim}}})$ | Stochastic simulation & empirical validation |
| **`"metamodel"`** | $O(n N_{\text{sim}})$ | **Instant** ($O(1)$ closed-form) | Moderate | Moderate | Compatibility with 3-parameter fitted Gamma pipelines |

---

## Installation & Usage with `uv`

```powershell
# Synchronize virtual environment and dependencies via uv
uv sync

# Run the test suite
uv run pytest -v

# Run the multi-scenario benchmark & generate outputs/method_comparison.png
uv run python -m gamma_linear_combination
```

---

## Quick Start (`method="auto"` for Non-Statistical Users)

By default (`method="auto"`), `GammaLinearCombination` automatically inspects your input data (step count, scale spread, skewness, and bottleneck dominance), selects and tunes the optimal engine across 6 data regimes, and explains its decision in plain English:

```python
from toolbox.stats import GammaLinearCombination

# Pass your Gamma components (alpha, beta) and optional linear weights:
glc = GammaLinearCombination(
    [(2.2, 1.0), (3.5, 1.8), (1.8, 2.5), (4.0, 1.2)],
    weights=[1.2, 0.8, 1.5, 1.0],
)

print(f"Mean: {glc.mean:.4f}, Std: {glc.std:.4f}, P95: {glc.percentile(0.95):.4f}")
print(glc.explain_auto_selection())
```

You can also specify an everyday goal via `auto_priority`:
- `auto_priority="balanced"` *(default)*
- `auto_priority="accuracy"` *(highest analytical precision)*
- `auto_priority="speed"` *(`< 0.1 ms` closed-form calculation for real-time loops)*
- `auto_priority="tail_risk"` *(best relative accuracy at extreme SLA quantiles like P99, P99.9)*
- `auto_priority="batch"` *(fastest vectorized evaluation for $> 5,000$ points)*

---

## Documentation & Notebooks

- **Technical Note**: [`docs/TECH_NOTE.md`](docs/TECH_NOTE.md) — Mathematical derivations, Big-O complexity, multi-regime benchmarks, and the 6-regime `method="auto"` decision flowchart.
- **Interactive Jupyter Notebook**: [`demo_gamma_linear_combination.ipynb`](demo_gamma_linear_combination.ipynb) — End-to-end walkthrough, visual comparisons, and automatic mode demonstrations.

