"""
Mathematical engines for computing PDF, CDF, and Percentiles (PPF) of a
Linear Combination of Independent Gamma Random Variables:
    Y = sum_{i=1}^n c_i X_i + loc_total,   where X_i ~ Gamma(alpha_i, beta_i)

Since c_i * X_i ~ Gamma(alpha_i, c_i * beta_i) under shape-scale parameterization,
every linear combination reduces to a sum of independent Gamma variables with
effective shapes `alphas` (alpha_i > 0) and effective scales `betas` (tilde_beta_i = c_i * beta_i > 0),
shifted by `total_loc`.

Implemented Engines (balancing Effectiveness vs. Efficiency):
1. MoschopoulosEngine ("moschopoulos"):
   Exact Moschopoulos (1985) single gamma-series representation enhanced with four
   engineering optimizations:
   (a) Scale coalescing (merging components with identical effective scales),
   (b) Dynamic exponent-rescaled BLAS recurrence (preventing float64 underflow for large n),
   (c) Active support window trimming [k_min, k_max],
   (d) Forward recurrence for regularized incomplete gamma CDF:
       P(a + 1, z) = P(a, z) - z^a * exp(-z) / Gamma(a + 1) (Hu, Pozdnyakov, & Yan 2020),
       requiring only ONE `scipy.special.gammainc` call per query point, and
   (e) Analytical-derivative safeguarded Newton-Raphson for quantile (PPF) evaluation.
2. MoschopoulosQuadEngine ("moschopoulos_quad"):
   Moschopoulos (1985) PDF paired with adaptive Gauss-Kronrod numerical quadrature
   (`scipy.integrate.quad`) for CDF.
3. BarnabaniEngine ("barnabani"):
   Barnabani (2017) & Vellaisamy-Upadhye (2009) Generalized Negative Binomial (GNB)
   mixture approximation (featured in the R `coga` package, Hu, Pozdnyakov, & Yan 2020),
   which replaces the O(K^2) recursive delta_k computation with a closed-form 3-moment
   Jain-Consul GNB / Negative Binomial PMF combined with incomplete-gamma forward recurrence.
4. LaguerreEngine ("laguerre"):
   Ha & Provost (2013) Generalized Laguerre Orthogonal Polynomial Expansion around the
   Welch-Satterthwaite baseline Gamma distribution, matching the exact first 6 cumulants
   via cumulant-excess generating functions and closed-form Rodrigues CDF integration.
5. SaddlepointEngine ("saddlepoint"):
   Daniels (1954) PDF and Lugannani-Rice (1980) CDF saddlepoint approximations
   using exact Cumulant Generating Function (CGF) inversion. Immune to large scale
   ratios (max(beta)/min(beta) >> 1) and O(n) per evaluation.
6. CharacteristicFunctionFFTEngine ("fft"):
   Exact Characteristic Function (CF) inversion via Gil-Pelaez Fourier integration
   with precomputed monotonic PCHIP cubic spline lookup tables for O(1) microsecond queries.
7. MomentMatchingEngine ("moment_matching" / "welch_satterthwaite"):
   Closed-form 2-moment (Welch-Satterthwaite) or 3-moment (Pearson Type III / skewness-matched
   shifted Gamma) approximation. O(n) setup and O(1) evaluation.
8. MonteCarloEngine ("monte_carlo" / "metamodel"):
   Vectorized Monte Carlo simulation supporting either empirical ECDF/KDE/quantiles
   or fitted 3-parameter Gamma metamodel.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import scipy.special as special
import scipy.stats as sts


def coalesce_equal_scales(
    alphas: np.ndarray,
    betas: np.ndarray,
    rtol: float = 1e-11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Engineering optimization (Hu, Pozdnyakov, & Yan 2020, Sec. 1):
    Independent Gamma variables with identical scale parameters beta_i can be
    coalesced prior to convolution via Gamma(alpha_1, beta) + Gamma(alpha_2, beta)
    ~ Gamma(alpha_1 + alpha_2, beta).
    """
    if len(alphas) <= 1:
        return alphas.copy(), betas.copy()

    order = np.argsort(betas)
    sorted_a = alphas[order]
    sorted_b = betas[order]

    merged_a = [float(sorted_a[0])]
    merged_b = [float(sorted_b[0])]

    for i in range(1, len(sorted_b)):
        if math.isclose(sorted_b[i], merged_b[-1], rel_tol=rtol, abs_tol=0.0):
            merged_a[-1] += float(sorted_a[i])
        else:
            merged_a.append(float(sorted_a[i]))
            merged_b.append(float(sorted_b[i]))

    return np.asarray(merged_a, dtype=float), np.asarray(merged_b, dtype=float)


def eval_gamma_mixture_cdf_recurrence(
    y: np.ndarray,
    beta_1: float,
    rho: float,
    k_min: int,
    weights_active: np.ndarray,
) -> np.ndarray:
    """
    Engineering Optimization: Evaluate the Gamma-series CDF
        F(y) = sum_{k = k_min}^{k_max} w_k * P(rho + k, y / beta_1)
    using only a SINGLE call to `scipy.special.gammainc` at a_0 = rho + k_min,
    followed by the exact forward recurrence (Abramowitz & Stegun 6.5.21;
    Hu, Pozdnyakov, & Yan 2020):
        P(a + 1, z) = P(a, z) - z^a * exp(-z) / Gamma(a + 1)
    where the Poisson-like term p_a(z) = z^a * exp(-z) / Gamma(a + 1) updates
    via p_{a+1}(z) = p_a(z) * (z / (a + 1)).
    """
    z = y / beta_1
    a_start = rho + float(k_min)
    P_curr = special.gammainc(a_start, z)

    # Check if log(p_start) can underflow float64 for some points
    log_z = np.log(z)
    log_p_start = a_start * log_z - z - special.gammaln(a_start + 1.0)

    # If all points have log_p_start > -600 and len(weights_active) is moderate,
    # run fast multiplicative recurrence in float64
    K_active = len(weights_active)
    if np.all(log_p_start > -550.0) and K_active <= 600:
        p_curr = np.exp(log_p_start)
        cdf_vals = weights_active[0] * P_curr
        for idx in range(1, K_active):
            a_curr = a_start + float(idx)
            P_curr = P_curr - p_curr
            cdf_vals = cdf_vals + weights_active[idx] * P_curr
            p_curr = p_curr * (z / a_curr)
        return np.clip(cdf_vals, 0.0, 1.0)

    # For wide active windows where p_a(z) starts near 0 and peaks near a ~ z,
    # compute p_a(z) in vectorized log-space and accumulate via cumsum!
    k_offsets = np.arange(K_active, dtype=float)
    a_vec = a_start + k_offsets  # shape (K_active,)
    # p_{k_min + j}(z) = z^(a_vec[j]) * exp(-z) / Gamma(a_vec[j] + 1) for j = 0 .. K_active-2
    if K_active > 1:
        a_steps = a_vec[:-1]
        log_p_mat = (
            a_steps[None, :] * log_z[:, None]
            - z[:, None]
            - special.gammaln(a_steps + 1.0)[None, :]
        )
        p_mat = np.exp(np.clip(log_p_mat, -745.0, 0.0))
        # P(a_start + j, z) = P(a_start, z) - sum_{m=0}^{j-1} p_mat[:, m]
        cum_p = np.cumsum(p_mat, axis=1)
        P_all_but_first = P_curr[:, None] - cum_p
        cdf_vals = weights_active[0] * P_curr + np.dot(P_all_but_first, weights_active[1:])
    else:
        cdf_vals = weights_active[0] * P_curr

    return np.clip(cdf_vals, 0.0, 1.0)


class BaseGammaEngine(ABC):
    """Abstract base class for Gamma linear combination computational engines."""

    name: str = "base"

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        total_loc: float = 0.0,
    ):
        self.raw_alphas = np.asarray(alphas, dtype=float)
        self.raw_betas = np.asarray(betas, dtype=float)
        # Coalesce identical scales for computational efficiency
        self.alphas, self.betas = coalesce_equal_scales(self.raw_alphas, self.raw_betas)
        self.total_loc = float(total_loc)
        self.n_terms = len(self.alphas)

        # Exact cumulants of the shifted sum Y = sum(Z_i) + total_loc
        # kappa_r = (r - 1)! * sum(alpha_i * beta_i^r)
        self.unshifted_mean = float(np.sum(self.alphas * self.betas))
        self.mean = self.unshifted_mean + self.total_loc
        self.variance = float(np.sum(self.alphas * (self.betas ** 2)))
        self.std = math.sqrt(self.variance)
        self.kappa3 = float(2.0 * np.sum(self.alphas * (self.betas ** 3)))
        self.kappa4 = float(6.0 * np.sum(self.alphas * (self.betas ** 4)))
        self.skewness = self.kappa3 / (self.variance ** 1.5) if self.variance > 0 else 0.0
        self.excess_kurtosis = self.kappa4 / (self.variance ** 2) if self.variance > 0 else 0.0

    @abstractmethod
    def pdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        """Evaluate PDF for unshifted variable y = x - total_loc > 0."""

    @abstractmethod
    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        """Evaluate CDF for unshifted variable y = x - total_loc > 0."""

    def ppf_unshifted(self, q: np.ndarray) -> np.ndarray:
        """
        Evaluate inverse CDF (quantile) for unshifted variable y > 0 using
        safeguarded Newton-Raphson with exact analytical derivative F'(y) = f(y),
        warm-started from the 3-moment Pearson Type III (or 2-moment WS) initial guess.
        """
        res = np.zeros_like(q, dtype=float)
        # 3-moment shifted Gamma initial guess (with 2-moment fallback)
        alpha_ws = (self.unshifted_mean ** 2) / self.variance
        beta_ws = self.variance / self.unshifted_mean
        beta_3m = self.kappa3 / (2.0 * self.variance) if self.variance > 0 else beta_ws
        alpha_3m = self.variance / (beta_3m ** 2) if beta_3m > 0 else alpha_ws
        loc_3m = self.unshifted_mean - alpha_3m * beta_3m

        for i, p in enumerate(q):
            if p <= 0.0:
                res[i] = 0.0
            elif p >= 1.0:
                res[i] = np.inf
            else:
                x0_3m = float(sts.gamma.ppf(p, alpha_3m, loc=loc_3m, scale=beta_3m))
                x0_ws = float(sts.gamma.ppf(p, alpha_ws, scale=beta_ws))
                x0 = x0_3m if (np.isfinite(x0_3m) and x0_3m > 0.0) else max(1e-12, x0_ws)

                low = max(1e-14, x0 * 0.25)
                high = max(x0 * 2.2, self.unshifted_mean + 6.0 * self.std)
                while float(self.cdf_unshifted(np.array([low]))[0]) > p and low > 1e-15:
                    low *= 0.2
                while float(self.cdf_unshifted(np.array([high]))[0]) < p:
                    high *= 2.0

                y_curr = min(max(x0, low), high)
                # Safeguarded Newton-Raphson using exact PDF = d(CDF)/dy
                for _ in range(18):
                    y_arr = np.array([y_curr])
                    f_val = float(self.cdf_unshifted(y_arr)[0]) - p
                    if abs(f_val) <= 1e-10:
                        break
                    if f_val < 0.0:
                        low = max(low, y_curr)
                    else:
                        high = min(high, y_curr)

                    pdf_val = float(self.pdf_unshifted(y_arr)[0])
                    if pdf_val > 1e-14:
                        y_next = y_curr - f_val / pdf_val
                        if low < y_next < high:
                            step = abs(y_next - y_curr)
                            y_curr = y_next
                            if step <= 1e-9 * max(1.0, y_curr):
                                break
                            continue
                    y_curr = 0.5 * (low + high)
                    if (high - low) <= 1e-9 * max(1.0, y_curr):
                        break

                res[i] = max(0.0, y_curr)
        return res

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        is_scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y = x_arr - self.total_loc
        out = np.zeros_like(y, dtype=float)
        pos = y > 0.0
        if np.any(pos):
            out[pos] = np.maximum(0.0, self.pdf_unshifted(y[pos]))
        return float(out[0]) if is_scalar else out

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        is_scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y = x_arr - self.total_loc
        out = np.zeros_like(y, dtype=float)
        pos = y > 0.0
        if np.any(pos):
            out[pos] = np.clip(self.cdf_unshifted(y[pos]), 0.0, 1.0)
        return float(out[0]) if is_scalar else out

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        is_scalar = np.isscalar(q)
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValueError("Quantiles q must be in [0, 1].")
        out = self.ppf_unshifted(q_arr) + self.total_loc
        return float(out[0]) if is_scalar else out

    def metadata(self) -> Dict[str, Any]:
        return {"engine": self.name, "coalesced_terms": self.n_terms}


# ==============================================================================
# Engine 1: Exact Moschopoulos (1985) Gamma-Series with Engineering Acceleration
# ==============================================================================

class MoschopoulosEngine(BaseGammaEngine):
    """
    Moschopoulos (1985) single gamma-series representation enhanced with:
    1. Scale Coalescing: Sums shape parameters of identical effective scales before expansion.
    2. Dynamic Exponent-Rescaled BLAS Recurrence: Prevents float64 overflow/underflow
       even when n >= 150 segments (where C = prod((beta_1/beta_i)^alpha_i) < 10^-300).
    3. Active Support Window Trimming [k_min, k_max]: Restricts evaluation to series indices
       carrying non-negligible weight (w_k >= 1e-15 * max(w_k)).
    4. Forward-Recurrence Incomplete Gamma CDF (Hu, Pozdnyakov, & Yan 2020):
       Evaluates `scipy.special.gammainc(rho + k_min, y / beta_1)` only ONCE per point y
       and obtains all subsequent P(rho + k, y / beta_1) via the exact recurrence
       P(a + 1, z) = P(a, z) - z^a * exp(-z) / Gamma(a + 1).
    """

    name = "moschopoulos"

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        total_loc: float = 0.0,
        max_terms: int = 150,
        tol: float = 1e-12,
        adaptive: bool = True,
    ):
        super().__init__(alphas, betas, total_loc)
        if max_terms < 1:
            raise ValueError(f"max_terms must be >= 1, got {max_terms}")

        self.max_terms = int(max_terms)
        self.tol = float(tol)
        self.adaptive = bool(adaptive)

        self.beta_1 = float(np.min(self.betas))
        self.rho = float(np.sum(self.alphas))

        # If all coalesced scales are equal (n_terms == 1), series has only 1 exact term (k=0)!
        if self.n_terms == 1:
            self.ratios = np.zeros(1, dtype=float)
            self.b_max = 0.0
            self.log_C = 0.0
            self.C = 1.0
            self.k_mean = 0.0
            self.k_std = 0.0
            self.actual_terms = 0
            self.gamma_k = np.array([1.0])
            self.delta_k = np.array([1.0])
            self.weights = np.array([1.0])
            self.log_weights = np.array([0.0])
            self.k_min = 0
            self.k_max = 0
            self.active_weights = self.weights
            self.active_log_weights = self.log_weights
            self.active_rho_plus_k = np.array([self.rho])
            self.active_lgamma_rho_k = special.gammaln(self.active_rho_plus_k)
            self.captured_mass = 1.0
            return

        # Ratio r_i = 1 - beta_1 / beta_i in [0, 1)
        self.ratios = 1.0 - (self.beta_1 / self.betas)
        self.b_max = float(np.max(self.ratios))

        # Log-constant: log(C) = sum(alpha_i * log(beta_1 / beta_i)) <= 0
        self.log_C = float(np.sum(self.alphas * np.log(self.beta_1 / self.betas)))
        self.C = math.exp(self.log_C)

        self._compute_coefficients()

    def _compute_coefficients(self) -> None:
        # The discrete mixture index K_mix ~ w_k = C * delta_k has exact moments:
        # E[K_mix]   = unshifted_mean / beta_1 - rho = sum(alpha_i * (beta_i / beta_1 - 1))
        # Var(K_mix) = variance / beta_1^2 - rho     = sum(alpha_i * ((beta_i / beta_1)^2 - 1))
        self.k_mean = max(0.0, self.unshifted_mean / self.beta_1 - self.rho)
        self.k_std = math.sqrt(max(0.0, self.variance / (self.beta_1 ** 2) - self.rho))

        if self.adaptive:
            needed_k = int(math.ceil(self.k_mean + 6.0 * self.k_std + 15))
            K = min(4000, max(self.max_terms, needed_k))
        else:
            K = self.max_terms

        k_idx = np.arange(1, K + 1, dtype=float)

        # Compute gamma_k = (1/k) * sum(alpha_i * r_i^k) for k = 1..K
        r_powers = self.ratios[:, None] ** k_idx[None, :]  # shape (n, K)
        gamma_k = np.dot(self.alphas, r_powers) / k_idx     # shape (K,)
        self.gamma_k = np.concatenate(([1.0], gamma_k))

        # Recurrence: delta_0 = 1, delta_k = (1/k) * sum_{j=1}^k (j * gamma_j) * delta_{k-j}
        # Dynamic exponent rescaling prevents float64 overflow while keeping fast BLAS np.dot!
        j_gamma = k_idx * gamma_k
        scaled_delta = np.zeros(K + 1, dtype=float)
        scaled_delta[0] = 1.0
        log_scale_offset = 0.0
        RESCALE_FACTOR = 1e-200
        LOG_RESCALE = 200.0 * math.log(10.0)
        actual_K = K

        for k in range(1, K + 1):
            dk = float(np.dot(j_gamma[:k], scaled_delta[k - 1::-1])) / k
            scaled_delta[k] = dk
            if dk > 1e200:
                scaled_delta[:k + 1] *= RESCALE_FACTOR
                log_scale_offset += LOG_RESCALE

            # Check adaptive convergence once k exceeds the mode k_mean
            if self.adaptive and k > max(10, int(self.k_mean + 3.0 * self.k_std)):
                if k % 16 == 0:
                    pos = scaled_delta[:k + 1] > 0
                    lw = self.log_C + log_scale_offset + np.log(scaled_delta[:k + 1][pos])
                    mass = float(np.sum(np.exp(np.clip(lw, -745.0, 0.0))))
                    if (1.0 - mass) <= self.tol:
                        actual_K = k
                        break

        self.actual_terms = actual_K
        sd = scaled_delta[:actual_K + 1]
        self.log_weights = np.where(
            sd > 0,
            self.log_C + log_scale_offset + np.log(np.maximum(sd, 1e-300)),
            -np.inf,
        )
        self.weights = np.exp(np.clip(self.log_weights, -745.0, 0.0))
        self.delta_k = np.exp(np.clip(log_scale_offset + np.log(np.maximum(sd, 1e-300)), -745.0, 700.0))

        self.captured_mass = float(np.sum(self.weights))
        if self.captured_mass > 0:
            self.weights = self.weights / self.captured_mass
            self.log_weights = self.log_weights - math.log(self.captured_mass)

        # Engineering optimization: Trim negligible leading/trailing weights (< 1e-15 * max_w)
        max_lw = float(np.max(self.log_weights))
        sig_indices = np.flatnonzero(self.log_weights >= (max_lw - 35.0))
        if len(sig_indices) > 0:
            self.k_min = int(sig_indices[0])
            self.k_max = int(sig_indices[-1])
        else:
            self.k_min = 0
            self.k_max = self.actual_terms

        self.active_weights = self.weights[self.k_min:self.k_max + 1]
        self.active_log_weights = self.log_weights[self.k_min:self.k_max + 1]
        active_k_range = np.arange(self.k_min, self.k_max + 1, dtype=float)
        self.active_rho_plus_k = self.rho + active_k_range
        self.active_lgamma_rho_k = special.gammaln(self.active_rho_plus_k)

    def pdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation of Moschopoulos PDF over the active support [k_min, k_max].
        """
        log_y = np.log(y)[:, None]
        log_beta1 = math.log(self.beta_1)
        y_div_beta1 = (y / self.beta_1)[:, None]

        log_gamma_pdf = (
            (self.active_rho_plus_k[None, :] - 1.0) * log_y
            - y_div_beta1
            - self.active_lgamma_rho_k[None, :]
            - self.active_rho_plus_k[None, :] * log_beta1
        )

        log_total_terms = log_gamma_pdf + self.active_log_weights[None, :]
        max_log = np.max(log_total_terms, axis=1, keepdims=True)
        pdf_vals = np.exp(np.clip(max_log[:, 0], -745.0, 700.0)) * np.sum(
            np.exp(np.clip(log_total_terms - max_log, -745.0, 0.0)), axis=1
        )
        return pdf_vals

    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation of Moschopoulos CDF using the forward recurrence
        for the regularized lower incomplete gamma function:
            P(a + 1, z) = P(a, z) - z^a * exp(-z) / Gamma(a + 1)
        requiring only a SINGLE `scipy.special.gammainc` evaluation per query point!
        """
        return eval_gamma_mixture_cdf_recurrence(
            y=y,
            beta_1=self.beta_1,
            rho=self.rho,
            k_min=self.k_min,
            weights_active=self.active_weights,
        )

    def pdf_uniform_bound(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Uniform convergence upper bound for the PDF given in Moschopoulos (1985)
        and the documentation (Eq. 245):
            f_bound(y) = (C * beta_1^(-rho) / Gamma(rho)) * y^(rho - 1) * exp(-y * (1 - b) / beta_1)
        where b = max_j (1 - beta_1 / beta_j).
        """
        is_scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y = x_arr - self.total_loc
        out = np.zeros_like(y, dtype=float)
        pos = y > 0.0
        if np.any(pos):
            yp = y[pos]
            log_bound = (
                self.log_C
                - self.rho * math.log(self.beta_1)
                - special.gammaln(self.rho)
                + (self.rho - 1.0) * np.log(yp)
                - yp * (1.0 - self.b_max) / self.beta_1
            )
            out[pos] = np.exp(log_bound)
        return float(out[0]) if is_scalar else out

    def metadata(self) -> Dict[str, Any]:
        return {
            "engine": self.name,
            "coalesced_terms": self.n_terms,
            "beta_1": self.beta_1,
            "rho": self.rho,
            "C": self.C,
            "log_C": self.log_C,
            "b_max": self.b_max,
            "max_terms": self.max_terms,
            "actual_terms": self.actual_terms,
            "active_window": (self.k_min, self.k_max),
            "captured_mass": self.captured_mass,
        }


# ==============================================================================
# Engine 2: Moschopoulos PDF + Numerical Quadrature CDF
# ==============================================================================

class MoschopoulosQuadEngine(MoschopoulosEngine):
    """
    Uses the Moschopoulos (1985) gamma-series for the PDF and adaptive
    Gauss-Kronrod quadrature (`scipy.integrate.quad`) for the CDF.
    """

    name = "moschopoulos_quad"

    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        res = np.zeros_like(y, dtype=float)
        for i, val in enumerate(y):
            if val <= 0.0:
                res[i] = 0.0
            else:
                integral, _ = integrate.quad(
                    lambda t: float(self.pdf_unshifted(np.array([t]))[0]),
                    0.0,
                    float(val),
                    limit=100,
                )
                res[i] = min(1.0, max(0.0, integral))
        return res


# ==============================================================================
# Engine 3: Barnabani (2017) Generalized Negative Binomial (GNB) Mixture Engine
# ==============================================================================

class BarnabaniEngine(BaseGammaEngine):
    """
    Barnabani (2017) & Vellaisamy-Upadhye (2009) Generalized Negative Binomial (GNB)
    approximation to the convolution of independent Gamma distributions (also featured
    in the R `coga` package by Hu, Pozdnyakov, & Yan 2020).

    References:
    - Vellaisamy, P., & Upadhye, N. S. (2009). "On the sums of compound negative
      binomial and gamma random variables." Journal of Applied Probability, 46(1), 272-283.
    - Barnabani, M. (2017). "An approximation to the convolution of gamma distributions."
      Communications in Statistics - Simulation and Computation, 46(1), 331-343.
    - Jain, G. C., & Consul, P. C. (1971). "A generalized negative binomial distribution."
      SIAM Journal on Applied Mathematics, 21(4), 501-513.

    Key Idea:
    The Moschopoulos mixture weights w_k = C * delta_k form the exact PMF of a
    sum of independent Negative Binomial random variables K = sum_{i=2}^n NB(alpha_i, 1 - beta_1/beta_i).
    Instead of computing delta_k via an O(K^2) recurrence, Barnabani (2017) matches the
    exact first three cumulants (mu, mu_2, mu_3) of K with a single Jain-Consul
    Generalized Negative Binomial GNB(r, p, b) (or Negative Binomial NB(r, p) when A < 2),
    evaluating the weights in O(1) closed form per term!
    """

    name = "barnabani"

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        total_loc: float = 0.0,
        max_terms: int = 300,
    ):
        super().__init__(alphas, betas, total_loc)
        self.max_terms = int(max_terms)
        self.beta_1 = float(np.min(self.betas))
        self.rho = float(np.sum(self.alphas))
        self._compute_gnb_weights()

    def _compute_gnb_weights(self) -> None:
        # Exact cumulants of the discrete mixture index K (Barnabani 2017 / approxcoga.cpp):
        ratio_inv = self.betas / self.beta_1          # beta_i / beta_1 >= 1
        q_i = 1.0 - (self.beta_1 / self.betas)        # 1 - beta_1 / beta_i in [0, 1)

        self.mu = float(np.sum(self.alphas * ratio_inv * q_i))
        self.mu2 = float(np.sum(self.alphas * (ratio_inv ** 2) * q_i))
        self.mu3 = float(np.sum(self.alphas * (ratio_inv ** 3) * q_i * (2.0 - self.beta_1 / self.betas)))

        if self.mu <= 1e-14 or self.mu2 <= self.mu:
            # Degenerate equal-scale case: all mass at k = 0
            self.gnb_mode = "degenerate"
            self.A = 0.0
            self.k_min = 0
            self.k_max = 0
            self.active_weights = np.array([1.0])
            self.active_log_weights = np.array([0.0])
            self.active_rho_plus_k = np.array([self.rho])
            self.active_lgamma_rho_k = special.gammaln(self.active_rho_plus_k)
            return

        self.A = ((self.mu * self.mu3 - 3.0 * (self.mu2 ** 2)) ** 2) / (self.mu * (self.mu2 ** 3)) - 2.0
        k_std = math.sqrt(self.mu2)
        K = min(4000, max(self.max_terms, int(math.ceil(self.mu + 6.5 * k_std + 20))))
        k_arr = np.arange(K + 1, dtype=float)

        use_gnb = self.A >= 2.0
        if use_gnb:
            p = 1.0 - 0.5 * self.A + math.sqrt(max(0.0, 0.25 * (self.A ** 2) - 1.0))
            cart = self.mu * (1.0 - p) / self.mu2
            b = (1.0 - math.sqrt(max(0.0, cart))) / p if p > 0 else 1.0
            r = self.mu * (1.0 - p * b) / p if p > 0 else 1.0
            if not (0.0 < p < 1.0 and -1.0 <= p * b <= 1.0 and r > 0.0):
                use_gnb = False

        if use_gnb:
            self.gnb_mode = "GNB"
            # GNB(k; r, p, b) = (r / (r + b*k)) * choose(r + b*k, k) * p^k * (1 - p)^(r + (b - 1)*k)
            rbk = r + b * k_arr
            rbk_minus_k = rbk - k_arr
            valid = (rbk > 0.0) & (rbk_minus_k > -1.0)
            # Also apply Barnabani's support cutoff if r + b * mu < 0
            if (r + b * self.mu) < 0.0:
                valid = valid & (k_arr < self.mu)

            log_w = np.full(K + 1, -np.inf, dtype=float)
            kv = k_arr[valid]
            rbk_v = rbk[valid]
            rbk_mk_v = rbk_minus_k[valid]
            log_w[valid] = (
                math.log(r)
                - np.log(rbk_v)
                + special.gammaln(rbk_v + 1.0)
                - special.gammaln(kv + 1.0)
                - special.gammaln(rbk_mk_v + 1.0)
                + kv * math.log(p)
                + rbk_mk_v * math.log(1.0 - p)
            )
        else:
            self.gnb_mode = "NB"
            # Standard Negative Binomial NB(r, p) matching exact mean mu and variance mu2
            p = 1.0 - (self.mu / self.mu2)
            r = (1.0 - p) * self.mu / p
            log_w = (
                special.gammaln(r + k_arr)
                - special.gammaln(r)
                - special.gammaln(k_arr + 1.0)
                + k_arr * math.log(p)
                + r * math.log(1.0 - p)
            )

        weights = np.exp(np.clip(log_w, -745.0, 0.0))
        total_w = float(np.sum(weights))
        if total_w > 0.0:
            weights /= total_w
            log_w -= math.log(total_w)

        max_lw = float(np.max(log_w))
        sig_indices = np.flatnonzero(log_w >= (max_lw - 35.0))
        self.k_min = int(sig_indices[0]) if len(sig_indices) > 0 else 0
        self.k_max = int(sig_indices[-1]) if len(sig_indices) > 0 else K

        self.active_weights = weights[self.k_min:self.k_max + 1]
        self.active_log_weights = log_w[self.k_min:self.k_max + 1]
        active_k = np.arange(self.k_min, self.k_max + 1, dtype=float)
        self.active_rho_plus_k = self.rho + active_k
        self.active_lgamma_rho_k = special.gammaln(self.active_rho_plus_k)

    def pdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        log_y = np.log(y)[:, None]
        log_beta1 = math.log(self.beta_1)
        y_div_beta1 = (y / self.beta_1)[:, None]

        log_gamma_pdf = (
            (self.active_rho_plus_k[None, :] - 1.0) * log_y
            - y_div_beta1
            - self.active_lgamma_rho_k[None, :]
            - self.active_rho_plus_k[None, :] * log_beta1
        )
        log_total_terms = log_gamma_pdf + self.active_log_weights[None, :]
        max_log = np.max(log_total_terms, axis=1, keepdims=True)
        return np.exp(np.clip(max_log[:, 0], -745.0, 700.0)) * np.sum(
            np.exp(np.clip(log_total_terms - max_log, -745.0, 0.0)), axis=1
        )

    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        return eval_gamma_mixture_cdf_recurrence(
            y=y,
            beta_1=self.beta_1,
            rho=self.rho,
            k_min=self.k_min,
            weights_active=self.active_weights,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "engine": self.name,
            "gnb_mode": self.gnb_mode,
            "A_index": self.A,
            "active_window": (self.k_min, self.k_max),
        }


# ==============================================================================
# Engine 4: Ha & Provost (2013) Generalized Laguerre Polynomial Series Engine
# ==============================================================================

class LaguerreEngine(BaseGammaEngine):
    """
    Generalized Laguerre Orthogonal Polynomial Expansion (Ha & Provost 2013; Provost 2005).

    Expands the density and CDF around the Welch-Satterthwaite baseline Gamma
    distribution Gamma(alpha_0, beta_0) (where alpha_0 = mu^2 / sigma^2 and
    beta_0 = sigma^2 / mu match the exact first two moments):
        f(y) = g(y; alpha_0, beta_0) * [ 1 + sum_{k=3}^M c_k L_k^{(alpha_0 - 1)}(y / beta_0) ]

    Using the cumulant-excess generating function of Z = Y / beta_0 over Z_0 ~ Gamma(alpha_0, 1):
        sum_{k=0}^inf E[L_k^{(alpha_0 - 1)}(Z)] u^k = exp( sum_{r=3}^inf d_r (-u)^r (1 - u)^(-r) )
    where d_r = (kappa_r(Z) - (r - 1)! * alpha_0) / r!, the coefficients c_k are computed
    analytically up to order M = 6 without any catastrophic cancellation!

    Furthermore, by Rodrigues' formula for Generalized Laguerre polynomials, the CDF has
    an EXACT closed-form finite expression requiring only ONE incomplete gamma evaluation:
        F(y) = P(alpha_0, z) - (z^alpha_0 * exp(-z) / Gamma(alpha_0)) * sum_{k=3}^M (c_k / k) * L_{k-1}^{(alpha_0)}(z)
    where z = y / beta_0.
    """

    name = "laguerre"

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        total_loc: float = 0.0,
        order: int = 6,
    ):
        super().__init__(alphas, betas, total_loc)
        if not (3 <= order <= 6):
            raise ValueError(f"LaguerreEngine order must be between 3 and 6, got {order}")
        self.order = int(order)

        # Baseline Welch-Satterthwaite parameters matching kappa_1 and kappa_2
        self.alpha_0 = (self.unshifted_mean ** 2) / self.variance
        self.beta_0 = self.variance / self.unshifted_mean

        # Exact scaled cumulants of Z = Y / beta_0:
        # tilde_kappa_r = (r - 1)! * sum_i alpha_i * (beta_i / beta_0)^r
        b_scaled = self.betas / self.beta_0
        d = np.zeros(7, dtype=float)
        for r in range(3, self.order + 1):
            fact_r_minus_1 = float(math.factorial(r - 1))
            fact_r = float(math.factorial(r))
            tilde_kappa_r = fact_r_minus_1 * float(np.sum(self.alphas * (b_scaled ** r)))
            # Cumulant excess d_r = (tilde_kappa_r - (r - 1)! * alpha_0) / r!
            d[r] = (tilde_kappa_r - fact_r_minus_1 * self.alpha_0) / fact_r

        # Exact expectations mu_L[k] = E[L_k^{(alpha_0 - 1)}(Z)] from generating function
        # exp(-d_3 u^3 (1-u)^(-3) + d_4 u^4 (1-u)^(-4) - d_5 u^5 (1-u)^(-5) + d_6 u^6 (1-u)^(-6))
        mu_L = np.zeros(self.order + 1, dtype=float)
        mu_L[0] = 1.0
        if self.order >= 3:
            mu_L[3] = -d[3]
        if self.order >= 4:
            mu_L[4] = d[4] - 3.0 * d[3]
        if self.order >= 5:
            mu_L[5] = -d[5] + 4.0 * d[4] - 6.0 * d[3]
        if self.order >= 6:
            mu_L[6] = d[6] - 5.0 * d[5] + 10.0 * d[4] - 10.0 * d[3] + 0.5 * (d[3] ** 2)

        # Orthogonal expansion coefficients: c_k = (k! / (alpha_0)_k) * mu_L[k]
        self.coeffs = np.zeros(self.order + 1, dtype=float)
        self.coeffs[0] = 1.0
        poch = self.alpha_0 * (self.alpha_0 + 1.0)
        for k in range(3, self.order + 1):
            poch *= (self.alpha_0 + float(k - 1))
            self.coeffs[k] = (float(math.factorial(k)) / poch) * mu_L[k]

    def pdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        z = y / self.beta_0
        base_pdf = sts.gamma.pdf(y, self.alpha_0, scale=self.beta_0)
        nu = self.alpha_0 - 1.0
        corr = np.ones_like(z, dtype=float)
        for k in range(3, self.order + 1):
            if self.coeffs[k] != 0.0:
                corr += self.coeffs[k] * special.eval_genlaguerre(k, nu, z)
        return np.maximum(0.0, base_pdf * corr)

    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        z = y / self.beta_0
        base_cdf = special.gammainc(self.alpha_0, z)
        # Log-stabilized factor: z^alpha_0 * exp(-z) / Gamma(alpha_0)
        log_factor = self.alpha_0 * np.log(z) - z - special.gammaln(self.alpha_0)
        factor = np.exp(np.clip(log_factor, -745.0, 700.0))

        poly_sum = np.zeros_like(z, dtype=float)
        for k in range(3, self.order + 1):
            if self.coeffs[k] != 0.0:
                poly_sum += (self.coeffs[k] / float(k)) * special.eval_genlaguerre(k - 1, self.alpha_0, z)

        return np.clip(base_cdf + factor * poly_sum, 0.0, 1.0)

    def metadata(self) -> Dict[str, Any]:
        return {
            "engine": self.name,
            "order": self.order,
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
            "laguerre_coeffs_c3_to_cm": self.coeffs[3:].tolist(),
        }


# ==============================================================================
# Engine 5: Saddlepoint Approximation (Daniels PDF & Lugannani-Rice CDF)
# ==============================================================================

class SaddlepointEngine(BaseGammaEngine):
    """
    Saddlepoint approximation using the exact Cumulant Generating Function (CGF):
        K(t)   = -sum_{i=1}^n alpha_i * log(1 - beta_i * t),   for t < 1 / max(beta_i)
        K'(t)  =  sum_{i=1}^n (alpha_i * beta_i) / (1 - beta_i * t) = y
        K''(t) =  sum_{i=1}^n (alpha_i * beta_i^2) / (1 - beta_i * t)^2

    - PDF: Daniels (1954) saddlepoint density
        f_sp(y) = exp(K(t_hat) - t_hat * y) / sqrt(2 * pi * K''(t_hat))
    - CDF: Lugannani-Rice (1980) formula
        F_sp(y) = Phi(w) + phi(w) * (1/w - 1/u)
      where w = sgn(t_hat) * sqrt(2 * (t_hat * y - K(t_hat)))
      and   u = t_hat * sqrt(K''(t_hat)).
    """

    name = "saddlepoint"

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        total_loc: float = 0.0,
        normalize_pdf: bool = True,
    ):
        super().__init__(alphas, betas, total_loc)
        self.beta_max = float(np.max(self.betas))
        self.t_upper = 1.0 / self.beta_max
        self.normalize_pdf = bool(normalize_pdf)
        self._norm_factor = 1.0

        if self.normalize_pdf:
            c3_sq = (self.kappa3 ** 2) / (self.variance ** 3)
            c4 = self.kappa4 / (self.variance ** 2)
            corr = 1.0 + (3.0 * c4 - 5.0 * c3_sq) / 24.0
            if corr > 0.5:
                self._norm_factor = corr

    def _solve_saddlepoint(self, y: np.ndarray) -> np.ndarray:
        """
        Solve K'(t) = y for t in (-inf, 1/beta_max) using safeguarded Newton-Raphson.
        """
        t_hats = np.zeros_like(y, dtype=float)
        rho = float(np.sum(self.alphas))
        max_mask = np.isclose(self.betas, self.beta_max, rtol=1e-12)
        alpha_max = float(np.sum(self.alphas[max_mask]))

        for idx, y_val in enumerate(y):
            if np.isclose(y_val, self.unshifted_mean, rtol=1e-12):
                t_hats[idx] = 0.0
                continue

            if y_val < self.unshifted_mean:
                low = min(-rho / y_val * 2.0, (y_val - self.unshifted_mean) / self.variance * 2.0) - 1.0 / self.beta_max
                while float(np.sum(self.alphas * self.betas / (1.0 - self.betas * low))) > y_val:
                    low *= 2.0
                high = 0.0
                t = max(low * 0.5, (y_val - self.unshifted_mean) / self.variance)
                if t <= low or t >= high:
                    t = 0.5 * (low + high)
            else:
                low = 0.0
                high = self.t_upper * (1.0 - min(0.999999, (alpha_max * self.beta_max) / (y_val * 1.05)))
                while float(np.sum(self.alphas * self.betas / (1.0 - self.betas * high))) < y_val:
                    high = 0.5 * (high + self.t_upper)
                t = 0.5 * (low + high)

            for _ in range(35):
                denom = 1.0 - self.betas * t
                if np.any(denom <= 0.0):
                    t = 0.5 * (low + high)
                    denom = 1.0 - self.betas * t

                ratio = self.betas / denom
                kp = float(np.sum(self.alphas * ratio))
                diff = kp - y_val
                if abs(diff) <= 1e-11 * y_val:
                    break

                if diff < 0.0:
                    low = max(low, t)
                else:
                    high = min(high, t)

                kpp = float(np.sum(self.alphas * (ratio ** 2)))
                t_next = t - diff / kpp
                if low < t_next < high:
                    t = t_next
                else:
                    t = 0.5 * (low + high)

            t_hats[idx] = t

        return t_hats

    def pdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        t_hats = self._solve_saddlepoint(y)
        denom = 1.0 - self.betas[None, :] * t_hats[:, None]
        K_t = -np.sum(self.alphas[None, :] * np.log(denom), axis=1)
        K_pp = np.sum(self.alphas[None, :] * ((self.betas[None, :] / denom) ** 2), axis=1)

        exponent = K_t - t_hats * y
        raw_pdf = np.exp(exponent) / np.sqrt(2.0 * math.pi * K_pp)
        return raw_pdf / self._norm_factor

    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        t_hats = self._solve_saddlepoint(y)
        denom = 1.0 - self.betas[None, :] * t_hats[:, None]
        K_t = -np.sum(self.alphas[None, :] * np.log(denom), axis=1)
        K_pp = np.sum(self.alphas[None, :] * ((self.betas[None, :] / denom) ** 2), axis=1)

        cdf_vals = np.zeros_like(y, dtype=float)
        u = t_hats * np.sqrt(K_pp)
        deviance = np.maximum(0.0, 2.0 * (t_hats * y - K_t))
        w = np.sign(t_hats) * np.sqrt(deviance)

        near_zero = (np.abs(u) < 1e-4) | (np.abs(w) < 1e-4)
        far = ~near_zero

        if np.any(far):
            wf = w[far]
            uf = u[far]
            phi_w = sts.norm.pdf(wf)
            Phi_w = sts.norm.cdf(wf)
            cdf_vals[far] = Phi_w + phi_w * (1.0 / wf - 1.0 / uf)

        if np.any(near_zero):
            z = (y[near_zero] - self.unshifted_mean) / self.std
            cdf_vals[near_zero] = (
                sts.norm.cdf(z)
                - sts.norm.pdf(z) * (self.skewness / 6.0) * (z ** 2 - 1.0)
            )

        return np.clip(cdf_vals, 0.0, 1.0)


# ==============================================================================
# Engine 6: Characteristic Function / Fourier Inversion with Fast Spline Lookup
# ==============================================================================

class CharacteristicFunctionFFTEngine(BaseGammaEngine):
    """
    Inverts the exact Characteristic Function (CF) of the unshifted sum:
        phi(t) = prod_{i=1}^n (1 - i * beta_i * t)^(-alpha_i)
    using vectorized Fourier / Gil-Pelaez inversion on a high-resolution grid,
    then builds monotonic PCHIP cubic spline interpolators for O(1) microsecond
    evaluations of PDF, CDF, and PPF (percentile).
    """

    name = "fft"

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        total_loc: float = 0.0,
        grid_size: int = 512,
        n_freq: int = 1024,
    ):
        super().__init__(alphas, betas, total_loc)
        self.grid_size = int(grid_size)
        self.n_freq = int(n_freq)
        self._build_lookup_splines()

    def _build_lookup_splines(self) -> None:
        alpha_ws = (self.unshifted_mean ** 2) / self.variance
        beta_ws = self.variance / self.unshifted_mean
        y_min = max(1e-9, float(sts.gamma.ppf(1e-7, alpha_ws, scale=beta_ws)) * 0.5)
        y_max = float(sts.gamma.ppf(1.0 - 1e-7, alpha_ws, scale=beta_ws)) * 1.35

        t_max = 8.0 / self.std
        for _ in range(30):
            log_mag = -0.5 * float(np.sum(self.alphas * np.log1p((self.betas * t_max) ** 2)))
            if log_mag < -25.0:
                break
            t_max *= 1.8

        M = self.n_freq
        dt = t_max / M
        t_grid = (np.arange(M, dtype=float) + 0.5) * dt

        bt = t_grid[:, None] * self.betas[None, :]
        log_mod = -0.5 * np.dot(np.log1p(bt ** 2), self.alphas)
        phase = np.dot(np.arctan(bt), self.alphas)
        phi_t = np.exp(log_mod) * (np.cos(phase) + 1j * np.sin(phase))

        y_grid = np.linspace(y_min, y_max, self.grid_size)
        ty = np.outer(y_grid, t_grid)
        exp_neg_ity = np.cos(ty) - 1j * np.sin(ty)

        integrand_pdf = np.real(exp_neg_ity * phi_t[None, :])
        pdf_grid = np.maximum(0.0, (dt / math.pi) * np.sum(integrand_pdf, axis=1))

        phi_over_t = phi_t / t_grid
        integrand_cdf = np.imag(exp_neg_ity * phi_over_t[None, :])
        cdf_grid = np.clip(0.5 - (dt / math.pi) * np.sum(integrand_cdf, axis=1), 0.0, 1.0)
        cdf_grid = np.maximum.accumulate(cdf_grid)

        y_full = np.concatenate(([0.0], y_grid))
        pdf_0 = 0.0 if np.sum(self.alphas) > 1.0 else pdf_grid[0]
        pdf_full = np.concatenate(([pdf_0], pdf_grid))
        cdf_full = np.concatenate(([0.0], cdf_grid))
        cdf_full[-1] = max(cdf_full[-1], 1.0 - 1e-7)

        self.y_min = y_min
        self.y_max = y_max
        self._pdf_spline = interpolate.PchipInterpolator(y_full, pdf_full, extrapolate=False)
        self._cdf_spline = interpolate.PchipInterpolator(y_full, cdf_full, extrapolate=False)

        unique_cdf, unique_idx = np.unique(cdf_full, return_index=True)
        unique_y = y_full[unique_idx]
        if unique_cdf[0] > 0.0:
            unique_cdf = np.concatenate(([0.0], unique_cdf))
            unique_y = np.concatenate(([0.0], unique_y))
        self._ppf_spline = interpolate.PchipInterpolator(unique_cdf, unique_y, extrapolate=True)

    def pdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        vals = self._pdf_spline(y)
        out = np.where(np.isnan(vals), 0.0, np.maximum(0.0, vals))
        return out

    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        vals = self._cdf_spline(y)
        out = np.where(y <= 0.0, 0.0, np.where(y >= self.y_max, 1.0, vals))
        return np.clip(out, 0.0, 1.0)

    def ppf_unshifted(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros_like(q, dtype=float)
        interior = (q > 0.0) & (q < 1.0)
        out[q >= 1.0] = np.inf
        if np.any(interior):
            out[interior] = np.maximum(0.0, self._ppf_spline(q[interior]))
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "engine": self.name,
            "grid_size": self.grid_size,
            "n_freq": self.n_freq,
            "y_max": self.y_max,
        }


# ==============================================================================
# Engine 7: Moment Matching (2-Moment Welch-Satterthwaite & 3-Moment Shifted Gamma)
# ==============================================================================

class MomentMatchingEngine(BaseGammaEngine):
    """
    Closed-form single-Gamma approximation via exact cumulant matching:
    - order = 2 (Welch-Satterthwaite):
        Matches mean (kappa_1) and variance (kappa_2) with loc_eff = 0:
            alpha_eff = kappa_1^2 / kappa_2
            beta_eff  = kappa_2 / kappa_1
    - order = 3 (Pearson Type III / 3-Moment Skewness-Matched Shifted Gamma):
        Matches mean (kappa_1), variance (kappa_2), and 3rd cumulant (kappa_3):
            alpha_eff = 4 / skewness^2 = 4 * kappa_2^3 / kappa_3^2
            beta_eff  = kappa_3 / (2 * kappa_2)
            loc_eff   = kappa_1 - alpha_eff * beta_eff
    """

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        total_loc: float = 0.0,
        order: int = 3,
    ):
        super().__init__(alphas, betas, total_loc)
        if order not in (2, 3):
            raise ValueError(f"Moment matching order must be 2 or 3, got {order}")
        self.order = order
        self.name = "welch_satterthwaite" if order == 2 else "moment_matching"

        if order == 2 or self.kappa3 <= 0:
            self.alpha_eff = (self.unshifted_mean ** 2) / self.variance
            self.beta_eff = self.variance / self.unshifted_mean
            self.loc_eff = 0.0
        else:
            self.beta_eff = self.kappa3 / (2.0 * self.variance)
            self.alpha_eff = self.variance / (self.beta_eff ** 2)
            self.loc_eff = self.unshifted_mean - self.alpha_eff * self.beta_eff

    def pdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        return sts.gamma.pdf(y, self.alpha_eff, loc=self.loc_eff, scale=self.beta_eff)

    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        return sts.gamma.cdf(y, self.alpha_eff, loc=self.loc_eff, scale=self.beta_eff)

    def ppf_unshifted(self, q: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, sts.gamma.ppf(q, self.alpha_eff, loc=self.loc_eff, scale=self.beta_eff))

    def metadata(self) -> Dict[str, Any]:
        return {
            "engine": self.name,
            "moment_order": self.order,
            "alpha_eff": self.alpha_eff,
            "loc_eff": self.loc_eff + self.total_loc,
            "beta_eff": self.beta_eff,
        }


# ==============================================================================
# Engine 8: Monte Carlo Simulation & Metamodel Engine
# ==============================================================================

class MonteCarloEngine(BaseGammaEngine):
    """
    Stochastic Monte Carlo engine that simulates M independent draws of
        Y = sum_{i=1}^n Z_i,   Z_i ~ Gamma(alpha_i, beta_i)
    and supports either empirical ECDF/KDE ("empirical") or 3-parameter fitted
    Gamma ("metamodel").
    """

    name = "monte_carlo"

    def __init__(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        total_loc: float = 0.0,
        n_samples: int = 25000,
        mode: str = "empirical",
        random_state: Optional[int] = 42,
    ):
        super().__init__(alphas, betas, total_loc)
        if mode not in ("empirical", "metamodel"):
            raise ValueError(f"MonteCarloEngine mode must be 'empirical' or 'metamodel', got {mode!r}")

        self.n_samples = int(n_samples)
        self.mode = mode
        self.rng = np.random.default_rng(random_state)

        samples = np.zeros(self.n_samples, dtype=float)
        for a, b in zip(self.alphas, self.betas):
            samples += self.rng.gamma(shape=a, scale=b, size=self.n_samples)
        self.samples_unshifted = np.sort(samples)

        self.meta_alb: Optional[Tuple[float, float, float]] = None
        self._kde: Optional[sts.gaussian_kde] = None

        if self.mode == "metamodel":
            floc = max(0.0, float(self.samples_unshifted[0]) - 1e-4)
            shape, loc_fit, scale = sts.gamma.fit(self.samples_unshifted, floc=floc)
            self.meta_alb = (float(shape), float(loc_fit), float(scale))
        else:
            self._kde = sts.gaussian_kde(self.samples_unshifted)

    def pdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        if self.mode == "metamodel" and self.meta_alb is not None:
            return sts.gamma.pdf(y, self.meta_alb[0], loc=self.meta_alb[1], scale=self.meta_alb[2])
        return self._kde(y)

    def cdf_unshifted(self, y: np.ndarray) -> np.ndarray:
        if self.mode == "metamodel" and self.meta_alb is not None:
            return sts.gamma.cdf(y, self.meta_alb[0], loc=self.meta_alb[1], scale=self.beta_alb_scale)
        idx = np.searchsorted(self.samples_unshifted, y, side="right")
        return idx / float(self.n_samples)

    @property
    def beta_alb_scale(self) -> float:
        return self.meta_alb[2] if self.meta_alb is not None else 1.0

    def ppf_unshifted(self, q: np.ndarray) -> np.ndarray:
        if self.mode == "metamodel" and self.meta_alb is not None:
            return np.maximum(
                0.0,
                sts.gamma.ppf(q, self.meta_alb[0], loc=self.meta_alb[1], scale=self.meta_alb[2]),
            )
        out = np.zeros_like(q, dtype=float)
        interior = (q > 0.0) & (q < 1.0)
        out[q >= 1.0] = np.inf
        if np.any(interior):
            out[interior] = np.quantile(self.samples_unshifted, q[interior])
        return out

    def metadata(self) -> Dict[str, Any]:
        return {
            "engine": f"{self.name}_{self.mode}",
            "n_samples": self.n_samples,
            "mode": self.mode,
            "meta_alb": self.meta_alb,
        }
