"""
Core implementation of `toolbox.stats.GammaLinearCombination`
=============================================================
Implements the linear combination of independent Gamma random variables:
    Y = sum_{i=1}^n c_i X_i + L,   where c_i > 0 and X_i ~ Gamma(alpha_i, beta_i)

Reference:
- Documentation: https://liam-hsieh.github.io/liamhsieh-toolbox/toolbox/stats/gamma_linear_combination/
- Moschopoulos, P. G. (1985). "The distribution of the sum of independent gamma random variables."
  Annals of the Institute of Statistical Mathematics, 37(1), 541-544.
- Mathai, A. M. (1982). "Storage capacity of a dam with gamma type inputs."
  Annals of the Institute of Statistical Mathematics, 34(3), 591-597.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as sts

from .engines import (
    BarnabaniEngine,
    BaseGammaEngine,
    CharacteristicFunctionFFTEngine,
    LaguerreEngine,
    MomentMatchingEngine,
    MonteCarloEngine,
    MoschopoulosEngine,
    MoschopoulosQuadEngine,
    SaddlepointEngine,
    coalesce_equal_scales,
)

AVAILABLE_METHODS = (
    "moschopoulos",
    "moschopoulos_quad",
    "barnabani",
    "laguerre",
    "saddlepoint",
    "fft",
    "moment_matching",
    "welch_satterthwaite",
    "monte_carlo",
    "metamodel",
    "auto",
)

AVAILABLE_PRIORITIES = (
    "balanced",
    "accuracy",
    "speed",
    "tail_risk",
    "batch",
)


@dataclass(frozen=True)
class AutoSelectionDecision:
    """
    Plain-English and diagnostic record of how `method='auto'` selected the
    computational engine for a given input dataset.

    Designed for users without a statistical background to understand what
    characteristics of their data drove the automatic method selection.
    """

    selected_method: str
    regime_id: str
    regime_label: str
    priority: str
    plain_english_summary: str
    technical_rationale: str
    expected_accuracy: str
    expected_speed: str
    alternative_suggestion: str
    data_profile: Dict[str, Any]
    auto_tuned_params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the auto-selection decision into a plain dictionary."""
        return {
            "selected_method": self.selected_method,
            "regime_id": self.regime_id,
            "regime_label": self.regime_label,
            "priority": self.priority,
            "plain_english_summary": self.plain_english_summary,
            "technical_rationale": self.technical_rationale,
            "expected_accuracy": self.expected_accuracy,
            "expected_speed": self.expected_speed,
            "alternative_suggestion": self.alternative_suggestion,
            "data_profile": dict(self.data_profile),
            "auto_tuned_params": dict(self.auto_tuned_params),
        }

    def format_report(self) -> str:
        """
        Return a user-friendly, jargon-free explanation of the automatic method
        selection suitable for printing in notebooks or logs.
        """
        dp = self.data_profile
        lines = [
            "==========================================================================",
            " Automatic Method Selection Report (for Non-Statistical Users)",
            "==========================================================================",
            f"  Selected Engine      : {self.selected_method!r} (Priority: {self.priority!r})",
            f"  Detected Data Regime : {self.regime_label} [{self.regime_id}]",
            "--------------------------------------------------------------------------",
            "  What Your Input Data Looks Like:",
            f"    - Random steps / components : {dp.get('n_valid_components', 0)} valid "
            f"({dp.get('n_coalesced_components', 0)} unique scale groups after coalescing)",
            f"    - Step variability spread   : {dp.get('scale_ratio', 1.0):.2f}x ratio between "
            "widest and narrowest step scales",
            f"    - Largest bottleneck share  : {dp.get('max_variance_share_pct', 0.0):.1f}% of total variance",
            f"    - Distribution shape        : {dp.get('shape_description', 'Standard')} "
            f"(skewness = {dp.get('skewness', 0.0):.2f})",
            "--------------------------------------------------------------------------",
            "  Why This Method Was Chosen (Plain English):",
            f"    {self.plain_english_summary}",
            "",
            "  Expected Performance:",
            f"    - Accuracy : {self.expected_accuracy}",
            f"    - Speed    : {self.expected_speed}",
            "--------------------------------------------------------------------------",
            f"  Tip: {self.alternative_suggestion}",
            "==========================================================================",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class GammaComponent:
    """
    Represents a single term `c_i * X_i` in the linear combination, where
    `X_i ~ Gamma(alpha, beta)` with optional location shift `loc` and positive
    linear weight `weight` (c_i > 0).

    Under shape-scale parameterization:
        c_i * X_i ~ Gamma(alpha, weight * beta) with location shift weight * loc.
    """

    alpha: float
    beta: float
    loc: float = 0.0
    weight: float = 1.0
    name: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """True if alpha, beta, and weight are finite positive numbers."""
        return (
            not np.isnan(self.alpha)
            and not np.isnan(self.beta)
            and not np.isnan(self.weight)
            and self.alpha > 0.0
            and self.beta > 0.0
            and self.weight > 0.0
            and not np.isnan(self.loc)
        )

    @property
    def is_constant(self) -> bool:
        """True if shape/scale are NaN or 0 but loc is valid (degenerate point mass)."""
        return (
            (np.isnan(self.alpha) or self.alpha == 0.0)
            and not np.isnan(self.loc)
            and not np.isnan(self.weight)
            and self.weight > 0.0
        )

    @property
    def effective_alpha(self) -> float:
        """Effective shape parameter of c_i * X_i (invariant under positive scaling)."""
        return self.alpha

    @property
    def effective_beta(self) -> float:
        """Effective scale parameter c_i * beta_i of c_i * X_i."""
        return self.weight * self.beta

    @property
    def effective_loc(self) -> float:
        """Effective location shift c_i * loc_i of c_i * X_i."""
        return self.weight * self.loc

    @property
    def mean(self) -> float:
        if self.is_valid:
            return self.effective_alpha * self.effective_beta + self.effective_loc
        if self.is_constant:
            return self.effective_loc
        return np.nan

    @property
    def variance(self) -> float:
        if self.is_valid:
            return self.effective_alpha * (self.effective_beta ** 2)
        if self.is_constant:
            return 0.0
        return np.nan


class GammaLinearCombination:
    """
    Linear Combination of Independent Gamma Distributed Random Variables
    (`toolbox.stats.GammaLinearCombination`).

    Models the distribution of:
        Y = sum_{i=1}^n c_i X_i + L
    where X_i ~ Gamma(alpha_i, beta_i) are independent Gamma random variables
    and c_i > 0 are positive real linear weights.

    Supports multiple computational methods (`method` parameter) so users can
    select the optimal trade-off between **effectiveness (accuracy)** and
    **efficiency (speed/scalability)**:

    - `"moschopoulos"` (default):
        Exact single gamma-series representation (Moschopoulos 1985) with log-space
        underflow protection, adaptive truncation, and analytical incomplete gamma CDF.
    - `"moschopoulos_quad"`:
        Moschopoulos PDF paired with adaptive Gauss-Kronrod quadrature (`scipy.integrate.quad`)
        for CDF.
    - `"saddlepoint"`:
        Daniels (1954) PDF and Lugannani-Rice (1980) CDF saddlepoint approximations
        from the exact Cumulant Generating Function. Ideal for large scale ratios
        (`max(beta)/min(beta) >> 10`) or hundreds of segments (`n >= 100`).
    - `"fft"` (or `"characteristic_function"`):
        Exact Characteristic Function inversion via Fourier integration with cached
        monotonic cubic splines for O(1) microsecond batch queries.
    - `"moment_matching"`:
        3-moment skewness-matched shifted Gamma (Pearson Type III) matching exact
        mean, variance, and skewness in O(1) closed form.
    - `"welch_satterthwaite"`:
        2-moment mean-and-variance matched Gamma approximation in O(1) closed form.
    - `"monte_carlo"`:
        Empirical Monte Carlo simulation with ECDF quantiles and Gaussian KDE density.
    - `"metamodel"`:
        Monte Carlo simulation fitted to a 3-parameter Gamma metamodel.
    - `"auto"`:
        Automatically selects between `"moschopoulos"` (when series converges rapidly)
        and `"saddlepoint"` (when scale spread or term count makes series expansion slow).
    """

    def __init__(
        self,
        gammas: Optional[Sequence[Any]] = None,
        *,
        alphas: Optional[Sequence[float]] = None,
        betas: Optional[Sequence[float]] = None,
        weights: Optional[Sequence[float]] = None,
        locs: Optional[Sequence[float]] = None,
        parameterization: str = "scale",
        method: str = "auto",
        auto_priority: str = "balanced",
        accuracy_k: int = 120,
        tol: float = 1e-12,
        adaptive: bool = True,
        fft_grid_size: int = 512,
        fft_n_freq: int = 1024,
        mc_samples: int = 25000,
        random_state: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        gammas : Sequence, optional
            Sequence of Gamma component specifications. Each element can be:
            - `(alpha, beta)` tuple/list for a 2-parameter Gamma(alpha, beta)
            - `(alpha, loc, beta)` tuple/list for a 3-parameter shifted Gamma(alpha, loc, beta)
            - `(alpha, loc, beta, weight)` 4-tuple
            - `dict` with keys `'alpha'` (or `'shape'`, `'k'`), `'beta'` (or `'scale'`, `'theta'`, `'rate'`),
              and optional `'loc'`, `'weight'` (or `'c'`).
            - `GammaComponent` instance.
        alphas : Sequence[float], optional
            Array of shape parameters alpha_i > 0 (alternative to passing `gammas`).
        betas : Sequence[float], optional
            Array of scale (or rate) parameters beta_i > 0 (alternative to passing `gammas`).
        weights : Sequence[float], optional
            Positive linear combination weights c_i > 0 in Y = sum(c_i * X_i).
            Defaults to 1.0 for all components.
        locs : Sequence[float], optional
            Optional location shifts loc_i >= 0 for each component.
        parameterization : {'scale', 'rate'}, default 'scale'
            - `'scale'`: beta_i is the scale parameter theta_i, so c_i * X_i ~ Gamma(alpha_i, c_i * beta_i).
            - `'rate'`: beta_i is the rate parameter lambda_i = 1 / theta_i, so
              c_i * X_i ~ Gamma(alpha_i, rate = beta_i / c_i), i.e., scale = c_i / beta_i
              (see Appendix: "extend to a linear combination").
        method : str, default 'auto'
            Computational engine: `'auto'` (default, automatically selects the optimal engine
            based on input data characteristics), `'moschopoulos'`, `'moschopoulos_quad'`,
            `'barnabani'`, `'laguerre'`, `'saddlepoint'`, `'fft'`, `'moment_matching'`,
            `'welch_satterthwaite'`, `'monte_carlo'`, or `'metamodel'`.
        auto_priority : {'balanced', 'accuracy', 'speed', 'tail_risk', 'batch'}, default 'balanced'
            User goal when `method='auto'` (designed for users without a statistical background):
            - `'balanced'`: Automatically balances high accuracy and sub-millisecond speed.
            - `'accuracy'` (or `'exact'`): Highest possible analytical precision.
            - `'speed'` (or `'realtime'`): Fastest closed-form calculation for real-time loops.
            - `'tail_risk'` (or `'sla'`): Best relative accuracy in extreme tails (P99, P99.9).
            - `'batch'` (or `'large_grid'`): Fastest evaluation for > 5,000 query points.
        accuracy_k : int, default 120
            Maximum number of series terms K in Moschopoulos expansion.
        tol : float, default 1e-12
            Convergence tolerance for adaptive Moschopoulos series truncation.
        adaptive : bool, default True
            Whether to truncate Moschopoulos series early once captured probability mass >= 1 - tol.
        fft_grid_size : int, default 512
            Spatial spline grid resolution for the `'fft'` engine.
        fft_n_freq : int, default 1024
            Frequency quadrature points for the `'fft'` engine.
        mc_samples : int, default 25000
            Number of random draws for `'monte_carlo'` and `'metamodel'` engines.
        random_state : int, optional, default 42
            Random seed for reproducible simulation.
        """
        if parameterization not in ("scale", "rate"):
            raise ValueError(f"parameterization must be 'scale' or 'rate', got {parameterization!r}")
        if not isinstance(accuracy_k, int) or accuracy_k < 1:
            raise ValueError(f"accuracy_k must be a positive integer, got {accuracy_k}")

        self.parameterization = parameterization
        self.requested_method = self._normalize_method_name(method)
        self.auto_priority = self._normalize_priority(auto_priority)
        self.accuracy_k = accuracy_k
        self.tol = tol
        self.adaptive = adaptive
        self.fft_grid_size = fft_grid_size
        self.fft_n_freq = fft_n_freq
        self.mc_samples = mc_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Parse components
        self.components: List[GammaComponent] = self._parse_inputs(
            gammas=gammas,
            alphas=alphas,
            betas=betas,
            weights=weights,
            locs=locs,
            parameterization=parameterization,
        )

        self.valid_indices: List[int] = []
        self.constant_indices: List[int] = []
        for idx, comp in enumerate(self.components):
            if comp.is_valid:
                self.valid_indices.append(idx)
            elif comp.is_constant:
                self.constant_indices.append(idx)

        num_valid = len(self.valid_indices)
        num_const = len(self.constant_indices)

        # Distribution type classification (compatible with MfgCT / CycleTime):
        #   1  : >= 2 valid Gamma components (true linear combination / convolution)
        #   0  : 1 valid Gamma component + shift
        #  -1  : deterministic constant shift only (zero variance)
        #  -2  : invalid / empty
        if num_valid >= 2:
            self.dist_type = 1
        elif num_valid == 1:
            self.dist_type = 0
        elif num_const > 0:
            self.dist_type = -1
        else:
            self.dist_type = -2

        all_locs = [c.effective_loc for c in self.components if not np.isnan(c.effective_loc)]
        self.total_loc = float(np.sum(all_locs)) if len(all_locs) > 0 else 0.0

        self.valid_alphas = np.array(
            [self.components[i].effective_alpha for i in self.valid_indices], dtype=float
        )
        self.valid_betas = np.array(
            [self.components[i].effective_beta for i in self.valid_indices], dtype=float
        )

        # Build the automatic data-driven decision profile
        self._auto_decision: AutoSelectionDecision = self._build_auto_decision(self.auto_priority)

        # Cache of instantiated engines so users can compare methods on the fly
        self._engines: Dict[str, BaseGammaEngine] = {}
        self.active_method = self._resolve_method(self.requested_method)

        if self.dist_type == 1:
            self._engine: Optional[BaseGammaEngine] = self._get_engine(self.active_method)
        else:
            self._engine = None

    @staticmethod
    def _normalize_priority(priority: str) -> str:
        p = priority.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "default": "balanced",
            "auto": "balanced",
            "exact": "accuracy",
            "precision": "accuracy",
            "high_accuracy": "accuracy",
            "fast": "speed",
            "realtime": "speed",
            "real_time": "speed",
            "low_latency": "speed",
            "tail": "tail_risk",
            "sla": "tail_risk",
            "extreme_quantiles": "tail_risk",
            "large_grid": "batch",
            "vectorized": "batch",
            "grid": "batch",
        }
        p = aliases.get(p, p)
        if p not in AVAILABLE_PRIORITIES:
            raise ValueError(
                f"Unknown priority {priority!r}. Choose from {AVAILABLE_PRIORITIES}."
            )
        return p

    @staticmethod
    def _normalize_method_name(method: str) -> str:
        m = method.strip().lower().replace("-", "_")
        aliases = {
            "exact": "moschopoulos",
            "series": "moschopoulos",
            "moschopoulos_series": "moschopoulos",
            "quad": "moschopoulos_quad",
            "numerical": "moschopoulos_quad",
            "gnb": "barnabani",
            "coga_approx": "barnabani",
            "negative_binomial": "barnabani",
            "laguerre_series": "laguerre",
            "orthogonal_polynomial": "laguerre",
            "sp": "saddlepoint",
            "lugannani_rice": "saddlepoint",
            "cf": "fft",
            "characteristic_function": "fft",
            "fourier": "fft",
            "moment": "moment_matching",
            "moments": "moment_matching",
            "pearson3": "moment_matching",
            "ws": "welch_satterthwaite",
            "satterthwaite": "welch_satterthwaite",
            "mc": "monte_carlo",
            "simulation": "monte_carlo",
            "meta": "metamodel",
        }
        m = aliases.get(m, m)
        if m not in AVAILABLE_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Choose from {AVAILABLE_METHODS}."
            )
        return m

    @classmethod
    def _parse_inputs(
        cls,
        gammas: Optional[Sequence[Any]],
        alphas: Optional[Sequence[float]],
        betas: Optional[Sequence[float]],
        weights: Optional[Sequence[float]],
        locs: Optional[Sequence[float]],
        parameterization: str,
    ) -> List[GammaComponent]:
        if gammas is not None and (alphas is not None or betas is not None):
            raise ValueError("Provide either `gammas` OR (`alphas`, `betas`), not both.")

        raw_list: List[GammaComponent] = []

        if alphas is not None or betas is not None:
            if alphas is None or betas is None:
                raise ValueError("Both `alphas` and `betas` must be provided together.")
            a_arr = np.atleast_1d(np.asarray(alphas, dtype=float))
            b_arr = np.atleast_1d(np.asarray(betas, dtype=float))
            if len(a_arr) != len(b_arr):
                raise ValueError(f"Length mismatch: len(alphas)={len(a_arr)} != len(betas)={len(b_arr)}")
            n = len(a_arr)
            w_arr = np.ones(n, dtype=float) if weights is None else np.atleast_1d(np.asarray(weights, dtype=float))
            l_arr = np.zeros(n, dtype=float) if locs is None else np.atleast_1d(np.asarray(locs, dtype=float))
            if len(w_arr) != n or len(l_arr) != n:
                raise ValueError("`weights` and `locs` must have the same length as `alphas` and `betas`.")

            for i in range(n):
                b_val = float(b_arr[i])
                if parameterization == "rate" and b_val > 0 and not np.isnan(b_val):
                    b_val = 1.0 / b_val
                raw_list.append(
                    GammaComponent(
                        alpha=float(a_arr[i]),
                        beta=b_val,
                        loc=float(l_arr[i]),
                        weight=float(w_arr[i]),
                        name=f"Seg_{i+1}",
                    )
                )
            return raw_list

        if gammas is None:
            return []

        n = len(gammas)
        w_override = None if weights is None else np.atleast_1d(np.asarray(weights, dtype=float))
        l_override = None if locs is None else np.atleast_1d(np.asarray(locs, dtype=float))
        if w_override is not None and len(w_override) != n:
            raise ValueError(f"len(weights)={len(w_override)} must match len(gammas)={n}")
        if l_override is not None and len(l_override) != n:
            raise ValueError(f"len(locs)={len(l_override)} must match len(gammas)={n}")

        for i, item in enumerate(gammas):
            w_i = float(w_override[i]) if w_override is not None else None
            l_i = float(l_override[i]) if l_override is not None else None
            comp = cls._parse_single_component(item, idx=i, w_override=w_i, l_override=l_i, parameterization=parameterization)
            raw_list.append(comp)

        return raw_list

    @staticmethod
    def _parse_single_component(
        item: Any,
        idx: int,
        w_override: Optional[float],
        l_override: Optional[float],
        parameterization: str,
    ) -> GammaComponent:
        if isinstance(item, GammaComponent):
            beta = (1.0 / item.beta) if (parameterization == "rate" and item.beta > 0) else item.beta
            return GammaComponent(
                alpha=item.alpha,
                beta=beta,
                loc=item.loc if l_override is None else l_override,
                weight=item.weight if w_override is None else w_override,
                name=item.name or f"Seg_{idx+1}",
            )

        # Support objects from MfgCT/CycleTime (GammaParams / GammaDistribution)
        if hasattr(item, "alpha") and hasattr(item, "beta"):
            alpha = float(getattr(item, "alpha"))
            beta = float(getattr(item, "beta"))
            loc = float(getattr(item, "loc", 0.0))
            weight = float(getattr(item, "weight", 1.0))
            if parameterization == "rate" and beta > 0 and not np.isnan(beta):
                beta = 1.0 / beta
            return GammaComponent(
                alpha=alpha,
                beta=beta,
                loc=loc if l_override is None else l_override,
                weight=weight if w_override is None else w_override,
                name=getattr(item, "name", f"Seg_{idx+1}"),
            )

        if isinstance(item, dict):
            alpha = float(item.get("alpha", item.get("shape", item.get("k", np.nan))))
            if "rate" in item and "beta" not in item and "scale" not in item:
                r = float(item["rate"])
                beta = (1.0 / r) if r > 0 else np.nan
            else:
                beta = float(item.get("beta", item.get("scale", item.get("theta", np.nan))))
                if parameterization == "rate" and beta > 0 and not np.isnan(beta):
                    beta = 1.0 / beta
            loc = float(item.get("loc", item.get("shift", 0.0)))
            weight = float(item.get("weight", item.get("c", 1.0)))
            return GammaComponent(
                alpha=alpha,
                beta=beta,
                loc=loc if l_override is None else l_override,
                weight=weight if w_override is None else w_override,
                name=item.get("name", f"Seg_{idx+1}"),
            )

        if isinstance(item, (list, tuple, np.ndarray)):
            if len(item) == 2:
                # Standard 2-parameter (alpha, beta) from documentation
                alpha, beta = float(item[0]), float(item[1])
                loc, weight = 0.0, 1.0
            elif len(item) == 3:
                # 3-parameter (alpha, loc, beta)
                alpha, loc, beta = float(item[0]), float(item[1]), float(item[2])
                weight = 1.0
            elif len(item) == 4:
                # 4-parameter (alpha, loc, beta, weight)
                alpha, loc, beta, weight = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            elif len(item) == 1:
                alpha, loc, beta, weight = np.nan, float(item[0]), np.nan, 1.0
            else:
                alpha, loc, beta, weight = np.nan, np.nan, np.nan, 1.0

            if parameterization == "rate" and beta > 0 and not np.isnan(beta):
                beta = 1.0 / beta
            return GammaComponent(
                alpha=alpha,
                beta=beta,
                loc=loc if l_override is None else l_override,
                weight=weight if w_override is None else w_override,
                name=f"Seg_{idx+1}",
            )

        return GammaComponent(alpha=np.nan, beta=np.nan, loc=np.nan, weight=1.0, name=f"Seg_{idx+1}")

    def _build_auto_decision(self, priority: Optional[str] = None) -> AutoSelectionDecision:
        """
        Inspect the input Gamma components and construct a complete, transparent
        `AutoSelectionDecision` based on the Technical Note's Key Takeaways and
        Method Selection Summary.
        """
        prio = self.auto_priority if priority is None else self._normalize_priority(priority)

        if self.dist_type != 1:
            desc_map = {
                0: "Single Gamma component with location shift (exact single Gamma distribution).",
                -1: "Deterministic constant shift only (zero variance).",
                -2: "Empty or insufficient input data.",
            }
            return AutoSelectionDecision(
                selected_method="moschopoulos",
                regime_id="regime_0_trivial",
                regime_label="Single Component or Constant Shift",
                priority=prio,
                plain_english_summary=desc_map.get(self.dist_type, "Trivial distribution."),
                technical_rationale="No multi-component convolution required; evaluated directly via exact closed-form scipy.stats.gamma.",
                expected_accuracy="Exact (machine precision ~1e-15)",
                expected_speed="< 0.05 ms (closed-form O(1))",
                alternative_suggestion="No alternative needed; exact closed-form solution is already optimal.",
                data_profile={
                    "n_total_components": len(self.components),
                    "n_valid_components": len(self.valid_indices),
                    "n_coalesced_components": len(self.valid_indices),
                    "scale_ratio": 1.0 if len(self.valid_indices) == 1 else np.nan,
                    "total_shape": float(np.sum(self.valid_alphas)) if len(self.valid_alphas) > 0 else 0.0,
                    "skewness": self.skewness,
                    "excess_kurtosis": np.nan,
                    "max_variance_share_pct": 100.0 if len(self.valid_indices) == 1 else 0.0,
                    "estimated_series_terms": 1,
                    "shape_description": "Single Gamma / Constant",
                },
                auto_tuned_params={"effective_accuracy_k": 1},
            )

        # 1. Coalesce equal effective scales c_i * beta_i
        coalesced_a, coalesced_b = coalesce_equal_scales(self.valid_alphas, self.valid_betas)
        n_raw = len(self.valid_alphas)
        n_coal = len(coalesced_a)
        beta_min = float(np.min(coalesced_b))
        beta_max = float(np.max(coalesced_b))
        scale_ratio = beta_max / beta_min if beta_min > 0 else np.inf
        total_shape = float(np.sum(coalesced_a))
        skew = self.skewness
        var_total = self.variance
        k4 = float(6.0 * np.sum(coalesced_a * (coalesced_b ** 4)))
        excess_kurt = k4 / (var_total ** 2) if var_total > 0 else 0.0

        comp_vars = coalesced_a * (coalesced_b ** 2)
        max_var_share_pct = float(np.max(comp_vars) / var_total * 100.0) if var_total > 0 else 0.0

        # Estimate required Moschopoulos series terms from Barnabani GNB moments of k
        if n_coal <= 1 or scale_ratio <= 1.0 + 1e-11:
            mu_k = 0.0
            sigma_k = 0.0
            k_est = 1
        else:
            ratios = coalesced_b[1:] / beta_min
            mu_k = float(np.sum(coalesced_a[1:] * (ratios - 1.0)))
            sigma_k = float(math.sqrt(np.sum(coalesced_a[1:] * ratios * (ratios - 1.0))))
            k_est = int(math.ceil(mu_k + 8.0 * sigma_k + 15.0))

        if skew < 0.35:
            shape_desc = "Nearly symmetric / bell-shaped (Central Limit Theorem regime)"
        elif skew < 0.80:
            shape_desc = "Moderately right-skewed"
        else:
            shape_desc = "Strongly right-skewed (bottleneck-dominated tail)"

        data_profile = {
            "n_total_components": len(self.components),
            "n_valid_components": n_raw,
            "n_coalesced_components": n_coal,
            "scale_ratio": round(scale_ratio, 4),
            "total_shape": round(total_shape, 4),
            "skewness": round(skew, 4),
            "excess_kurtosis": round(excess_kurt, 4),
            "max_variance_share_pct": round(max_var_share_pct, 2),
            "estimated_series_terms": k_est,
            "shape_description": shape_desc,
        }

        # ----------------------------------------------------------------------
        # Regime 0: Identical effective scales after coalescing (n_coalesced == 1)
        # ----------------------------------------------------------------------
        if n_coal == 1 or scale_ratio <= 1.0 + 1e-11:
            return AutoSelectionDecision(
                selected_method="moschopoulos",
                regime_id="regime_0_identical_scales",
                regime_label="Regime 0: Identical Effective Scales (Exact Single Gamma)",
                priority=prio,
                plain_english_summary=(
                    f"All {n_raw} components share the exact same effective scale ({beta_min:.4g}). "
                    "They automatically merge into a single exact Gamma distribution requiring zero series iterations."
                ),
                technical_rationale=(
                    "By scale-coalescing (sum_{i} Gamma(alpha_i, beta) ~ Gamma(sum alpha_i, beta)), "
                    "MoschopoulosEngine collapses to K=1 term with O(1) exact evaluation."
                ),
                expected_accuracy="Exact (machine precision ~1e-15)",
                expected_speed="< 0.05 ms (O(1) closed-form)",
                alternative_suggestion="Already exact and O(1) fast; no other engine is needed.",
                data_profile=data_profile,
                auto_tuned_params={"effective_accuracy_k": 1},
            )

        # ----------------------------------------------------------------------
        # Priority Overrides for Specific User Goals
        # ----------------------------------------------------------------------
        if prio == "speed":
            if skew < 0.45:
                return AutoSelectionDecision(
                    selected_method="laguerre",
                    regime_id="priority_speed_low_skew",
                    regime_label="Speed Priority: Low-to-Moderate Skewness",
                    priority=prio,
                    plain_english_summary=(
                        "You prioritized maximum speed and your distribution has low-to-moderate asymmetry. "
                        "The 6-cumulant Laguerre polynomial expansion runs in closed form (< 0.1 ms) while "
                        "delivering ~100x higher accuracy than simple moment matching."
                    ),
                    technical_rationale=(
                        f"Skewness ({skew:.3f} < 0.45) guarantees rapid decay of standardized cumulants; "
                        "Ha & Provost (2013) 6-cumulant Laguerre series evaluates in O(1) vectorized operations."
                    ),
                    expected_accuracy="Very High (CDF error ~1e-5 to 1e-6)",
                    expected_speed="~0.05 - 0.15 ms",
                    alternative_suggestion="Use auto_priority='balanced' or 'accuracy' if you need exact machine precision.",
                    data_profile=data_profile,
                    auto_tuned_params={},
                )
            return AutoSelectionDecision(
                selected_method="moment_matching",
                regime_id="priority_speed_skewed",
                regime_label="Speed Priority: Skewed Distribution",
                priority=prio,
                plain_english_summary=(
                    "You prioritized maximum speed on a right-skewed dataset. The 3-moment Pearson Type III "
                    "(skewness-matched shifted Gamma) matches the exact mean, variance, and skewness in "
                    "instantaneous closed form (< 0.1 ms) and is guaranteed non-negative everywhere."
                ),
                technical_rationale=(
                    f"Skewness ({skew:.3f} >= 0.45); Pearson Type III 3-moment matching provides O(1) "
                    "closed-form PDF/CDF/PPF without polynomial oscillation."
                ),
                expected_accuracy="Moderate (CDF error ~1e-3 to 5e-3, quantile error < 0.5%)",
                expected_speed="< 0.10 ms",
                alternative_suggestion="Use auto_priority='balanced' to upgrade to ~1e-6 accuracy in ~1 ms.",
                data_profile=data_profile,
                auto_tuned_params={},
            )

        if prio == "batch":
            if n_coal > 25 and scale_ratio <= 8.0 and skew < 0.35:
                return AutoSelectionDecision(
                    selected_method="laguerre",
                    regime_id="priority_batch_large_line",
                    regime_label="Batch Priority: Large Multi-Step System",
                    priority=prio,
                    plain_english_summary=(
                        f"You prioritized fast batch evaluation over {n_raw} balanced steps. "
                        "The 6-cumulant Laguerre polynomial engine evaluates tens of thousands of points "
                        "directly in vectorized NumPy arrays in microseconds."
                    ),
                    technical_rationale=(
                        "Closed-form Rodrigues CDF and orthogonal polynomial recurrence evaluate N points in O(N) time "
                        "with zero grid-building overhead."
                    ),
                    expected_accuracy="Very High (CDF error ~1e-6)",
                    expected_speed="< 0.2 ms for 10,000 points",
                    alternative_suggestion="Use 'fft' if scale spread exceeds 8x.",
                    data_profile=data_profile,
                    auto_tuned_params={},
                )
            return AutoSelectionDecision(
                selected_method="fft",
                regime_id="priority_batch_spline",
                regime_label="Batch Priority: Precomputed Fourier Spline Grid",
                priority=prio,
                plain_english_summary=(
                    "You prioritized fast batch evaluation for large arrays. Characteristic Function FFT "
                    "inversion builds a smooth cubic spline table once (~1 ms) and then answers thousands of "
                    "PDF, CDF, and percentile queries in microseconds."
                ),
                technical_rationale=(
                    "Gil-Pelaez Fourier inversion with cached PCHIP monotonic splines achieves O(1) per-point "
                    "amortized complexity regardless of scale ratio."
                ),
                expected_accuracy="Very High (CDF error ~1e-6 to 1e-4)",
                expected_speed="~1 ms setup, < 0.05 ms per 1,000 queries",
                alternative_suggestion="Use auto_priority='balanced' for standard scalar or small-array queries.",
                data_profile=data_profile,
                auto_tuned_params={
                    "effective_fft_grid_size": max(512, self.fft_grid_size),
                    "effective_fft_n_freq": max(2048, self.fft_n_freq),
                },
            )

        if prio == "tail_risk":
            if n_coal <= 25 and scale_ratio <= 6.0 and k_est <= 200:
                tuned_k = max(self.accuracy_k, k_est)
                return AutoSelectionDecision(
                    selected_method="moschopoulos",
                    regime_id="priority_tail_exact_series",
                    regime_label="Tail-Risk Priority: Exact Convergent Series",
                    priority=prio,
                    plain_english_summary=(
                        "You prioritized extreme tail / SLA accuracy on a compact dataset. Because your step "
                        f"scales are within a {scale_ratio:.1f}x range, the exact Moschopoulos series converges "
                        "completely to 15-digit precision in the extreme tails."
                    ),
                    technical_rationale=(
                        f"Estimated series length K_est={k_est} <= 200; MoschopoulosEngine with incomplete-gamma "
                        "forward recurrence captures 1 - 1e-12 probability mass."
                    ),
                    expected_accuracy="Exact (machine precision ~1e-15 even at P99.9)",
                    expected_speed="< 0.5 ms",
                    alternative_suggestion="Automatically switches to 'saddlepoint' if scale ratio or step count grows.",
                    data_profile=data_profile,
                    auto_tuned_params={"effective_accuracy_k": tuned_k},
                )
            return AutoSelectionDecision(
                selected_method="saddlepoint",
                regime_id="priority_tail_saddlepoint",
                regime_label="Tail-Risk Priority: Uniform Relative Tail Accuracy",
                priority=prio,
                plain_english_summary=(
                    "You prioritized extreme tail / SLA percentiles (such as P95, P99, P99.9) on a dataset "
                    f"with {n_raw} steps and {scale_ratio:.1f}x scale spread. The Lugannani-Rice Saddlepoint "
                    "method is uniquely designed to maintain tiny relative error far out in the tails where "
                    "series truncate and polynomials oscillate."
                ),
                technical_rationale=(
                    "Lugannani-Rice (1980) and Daniels (1954) saddlepoint expansions achieve O(n^-1) uniform "
                    "relative tail error in O(n) time, completely independent of scale ratio."
                ),
                expected_accuracy="High Relative Tail Accuracy (< 0.05% quantile error at P99+)",
                expected_speed="~0.4 - 1.0 ms",
                alternative_suggestion="Use auto_priority='batch' if evaluating > 10,000 points simultaneously.",
                data_profile=data_profile,
                auto_tuned_params={},
            )

        if prio == "accuracy":
            if n_coal <= 50 and scale_ratio <= 10.0 and k_est <= 600:
                tuned_k = max(self.accuracy_k, min(800, k_est))
                return AutoSelectionDecision(
                    selected_method="moschopoulos",
                    regime_id="priority_accuracy_exact_series",
                    regime_label="Accuracy Priority: Exact Moschopoulos Series",
                    priority=prio,
                    plain_english_summary=(
                        "You prioritized maximum analytical accuracy. The Moschopoulos exact series has been "
                        f"automatically configured with up to {tuned_k} terms so it reaches near-machine "
                        "precision across your dataset."
                    ),
                    technical_rationale=(
                        f"Auto-scaled max_terms={tuned_k} >= K_est={k_est} guarantees full probability mass capture "
                        "with incomplete-gamma forward recurrence."
                    ),
                    expected_accuracy="Exact (~1e-13 to 1e-15 CDF error)",
                    expected_speed="~0.2 - 1.5 ms",
                    alternative_suggestion="Use auto_priority='balanced' for faster sub-millisecond approximations when n is large.",
                    data_profile=data_profile,
                    auto_tuned_params={"effective_accuracy_k": tuned_k},
                )
            return AutoSelectionDecision(
                selected_method="fft",
                regime_id="priority_accuracy_high_res_fft",
                regime_label="Accuracy Priority: High-Resolution Fourier Inversion",
                priority=prio,
                plain_english_summary=(
                    f"You prioritized maximum accuracy on a dataset with a wide {scale_ratio:.1f}x scale spread "
                    f"(or {n_raw} steps) where infinite series would need {k_est} terms. High-resolution "
                    "Characteristic Function Fourier inversion achieves ~1e-6 accuracy without series truncation."
                ),
                technical_rationale=(
                    "Gil-Pelaez exact Characteristic Function integration with 4096 frequency quadrature nodes "
                    "and 1024 PCHIP spline knots avoids geometric series slowdown."
                ),
                expected_accuracy="Very High (~1e-6 CDF error across body and tails)",
                expected_speed="~1.5 - 3.0 ms",
                alternative_suggestion="Use auto_priority='balanced' for sub-millisecond speed.",
                data_profile=data_profile,
                auto_tuned_params={
                    "effective_fft_grid_size": max(1024, self.fft_grid_size),
                    "effective_fft_n_freq": max(4096, self.fft_n_freq),
                },
            )

        # ----------------------------------------------------------------------
        # Default `auto_priority == 'balanced'`: 6-Regime Data-Driven Selector
        # Based on Key Takeaways & Method Selection Summary in TECH_NOTE.md
        # ----------------------------------------------------------------------

        # Regime I: Small-to-Moderate Component Count (n_coal <= 25) & Moderate Scale Ratio (<= 6.0)
        if n_coal <= 25 and scale_ratio <= 6.0 and k_est <= 180:
            tuned_k = max(self.accuracy_k, k_est)
            return AutoSelectionDecision(
                selected_method="moschopoulos",
                regime_id="regime_1_compact_homogeneous",
                regime_label="Regime I: Compact Step Count & Moderate Scale Spread",
                priority=prio,
                plain_english_summary=(
                    f"Your dataset has {n_raw} steps ({n_coal} unique scale groups) with a moderate "
                    f"{scale_ratio:.2f}x spread between the widest and narrowest steps. In this regime, "
                    "the exact Moschopoulos mathematical series converges rapidly to 15-digit precision "
                    "in a fraction of a millisecond."
                ),
                technical_rationale=(
                    f"n_coalesced={n_coal} <= 25, scale_ratio={scale_ratio:.2f} <= 6.0, and K_est={k_est} <= 180. "
                    "MoschopoulosEngine with incomplete-gamma forward recurrence and BLAS convolution is both exact and <0.5 ms fast."
                ),
                expected_accuracy="Exact (machine precision ~1e-15 CDF error)",
                expected_speed="~0.15 - 0.50 ms",
                alternative_suggestion=(
                    "Pass auto_priority='speed' if calling inside a tight real-time loop (>10,000 distinct distributions/sec)."
                ),
                data_profile=data_profile,
                auto_tuned_params={"effective_accuracy_k": tuned_k},
            )

        # Regime II-A: Moderate Component Count (n_coal <= 25) & Moderately High Scale Spread (6.0 < SR <= 12.0)
        if n_coal <= 25 and scale_ratio <= 12.0 and k_est <= 800:
            if k_est <= 200:
                tuned_k = max(self.accuracy_k, k_est)
                return AutoSelectionDecision(
                    selected_method="moschopoulos",
                    regime_id="regime_2a_moderate_spread_series",
                    regime_label="Regime II-A: Moderate Scale Spread (Auto-Extended Exact Series)",
                    priority=prio,
                    plain_english_summary=(
                        f"Your dataset has a {scale_ratio:.2f}x spread between step scales, requiring roughly "
                        f"{k_est} series terms. Auto mode automatically extended the exact Moschopoulos series "
                        f"limit to {tuned_k} terms so you get exact precision without truncation error."
                    ),
                    technical_rationale=(
                        f"Scale ratio ({scale_ratio:.2f}) is between 6x and 12x with K_est={k_est} <= 200; "
                        f"auto-tuning accuracy_k={tuned_k} preserves exactness in < 0.8 ms."
                    ),
                    expected_accuracy="Exact (~1e-13 to 1e-15 CDF error)",
                    expected_speed="~0.3 - 0.8 ms",
                    alternative_suggestion="Pass method='barnabani' or auto_priority='speed' for even faster approximate queries.",
                    data_profile=data_profile,
                    auto_tuned_params={"effective_accuracy_k": tuned_k},
                )
            return AutoSelectionDecision(
                selected_method="barnabani",
                regime_id="regime_2a_moderate_spread_gnb",
                regime_label="Regime II-A: Moderate Scale Spread (Barnabani GNB Mixture)",
                priority=prio,
                plain_english_summary=(
                    f"Your dataset has {n_raw} steps with a {scale_ratio:.2f}x scale spread (requiring ~{k_est} "
                    "series terms). Auto mode selected Barnabani's Generalized Negative Binomial (GNB) mixture, "
                    "which computes the series weights in closed form without slow recursive convolution."
                ),
                technical_rationale=(
                    f"For K_est={k_est} in (200, 800], Barnabani (2017) 3-moment GNB mixture replaces O(n K^2) "
                    "delta_k recursion with O(K) closed-form weights and active window trimming."
                ),
                expected_accuracy="High (CDF error ~1e-4, quantile error < 0.05%)",
                expected_speed="~0.2 - 0.5 ms",
                alternative_suggestion="Pass auto_priority='accuracy' if you require < 1e-6 error.",
                data_profile=data_profile,
                auto_tuned_params={"effective_accuracy_k": max(350, k_est)},
            )

        # Regime II-B: High Scale Disparity / Bottleneck-Dominated (n_coal <= 25 and (scale_ratio > 12 or k_est > 400))
        if n_coal <= 25:
            if skew > 0.80 or total_shape < 4.0:
                return AutoSelectionDecision(
                    selected_method="fft",
                    regime_id="regime_2b_bottleneck_fft",
                    regime_label="Regime II-B: Severe Bottleneck Disparity & High Skewness",
                    priority=prio,
                    plain_english_summary=(
                        f"Your dataset contains a dominant bottleneck step (accounting for {max_var_share_pct:.1f}% "
                        f"of total variance, with a {scale_ratio:.1f}x scale spread) and strong right-skewness "
                        f"({skew:.2f}). Infinite gamma series would stall here, so Auto mode selected "
                        "Characteristic Function FFT inversion to deliver ~99.9998% accuracy in ~1 ms."
                    ),
                    technical_rationale=(
                        f"High scale ratio ({scale_ratio:.2f}x, K_est={k_est}) stalls Moschopoulos/Barnabani series, "
                        f"and high skewness ({skew:.2f}) / small shape ({total_shape:.2f}) makes Gil-Pelaez FFT "
                        "inversion (~2e-6 CDF error) 350x more accurate than Saddlepoint near the mode."
                    ),
                    expected_accuracy="Very High (CDF error ~2e-6, quantile error ~0.0005%)",
                    expected_speed="~0.9 - 1.5 ms (and < 0.05 ms for subsequent queries)",
                    alternative_suggestion="Pass auto_priority='tail_risk' to use 'saddlepoint' for extreme P99.9+ tail analysis.",
                    data_profile=data_profile,
                    auto_tuned_params={
                        "effective_fft_grid_size": max(512, self.fft_grid_size),
                        "effective_fft_n_freq": max(2048, self.fft_n_freq),
                    },
                )
            return AutoSelectionDecision(
                selected_method="saddlepoint",
                regime_id="regime_2b_bottleneck_saddlepoint",
                regime_label="Regime II-B: High Scale Disparity (Saddlepoint Regime)",
                priority=prio,
                plain_english_summary=(
                    f"Your steps have a wide {scale_ratio:.1f}x scale spread where standard series expansions "
                    f"would need ~{k_est} terms. Auto mode selected the Saddlepoint method, which completely "
                    "bypasses series expansion and delivers < 0.05% error in under 0.5 ms."
                ),
                technical_rationale=(
                    f"Scale ratio {scale_ratio:.2f}x (K_est={k_est}) with smooth total shape ({total_shape:.2f} >= 4.0) "
                    f"and moderate skewness ({skew:.2f} <= 0.80): Lugannani-Rice Saddlepoint runs in O(n) time."
                ),
                expected_accuracy="High (CDF error ~5e-4, quantile error < 0.05%)",
                expected_speed="~0.3 - 0.6 ms",
                alternative_suggestion="Pass auto_priority='accuracy' to use high-resolution 'fft' (~1e-6 error).",
                data_profile=data_profile,
                auto_tuned_params={},
            )

        # Regime III-A: Large Multi-Step System (n_coal > 25), Balanced / Near-Gaussian (SR <= 8.0 and skew < 0.35)
        if scale_ratio <= 8.0 and skew < 0.35:
            return AutoSelectionDecision(
                selected_method="laguerre",
                regime_id="regime_3a_large_balanced",
                regime_label="Regime III-A: Large Multi-Step System (Central Limit / Laguerre Regime)",
                priority=prio,
                plain_english_summary=(
                    f"Your dataset combines {n_raw} process steps with balanced scales ({scale_ratio:.2f}x spread) "
                    f"and low asymmetry (skewness = {skew:.2f}). In large multi-step systems, higher-order "
                    "distortions fade rapidly, allowing the 6-cumulant Laguerre polynomial expansion to achieve "
                    "~99.9997% accuracy in ~0.1 ms (10x faster than series or root-finding methods)."
                ),
                technical_rationale=(
                    f"Large component count (n={n_coal} > 25) with skewness={skew:.3f} < 0.35 and scale_ratio={scale_ratio:.2f} <= 8.0: "
                    "standardized cumulants kappa_m / kappa_2^(m/2) = O(n^(1-m/2)) decay rapidly, making Ha & Provost (2013) "
                    "6-cumulant Laguerre expansion (~2.8e-6 CDF error in 0.1 ms) Pareto-optimal."
                ),
                expected_accuracy="Very High (CDF error ~3e-6, quantile error ~0.0001%)",
                expected_speed="~0.05 - 0.15 ms",
                alternative_suggestion="Pass auto_priority='tail_risk' to use 'saddlepoint' if querying extreme 1-in-1,000,000 tails.",
                data_profile=data_profile,
                auto_tuned_params={},
            )

        # Regime III-B: Large Multi-Step System (n_coal > 25) with Wide Scale Ratio (> 8.0) or Skewness (>= 0.35)
        return AutoSelectionDecision(
            selected_method="saddlepoint",
            regime_id="regime_3b_large_heterogeneous",
            regime_label="Regime III-B: Large Heterogeneous Multi-Step System (Saddlepoint Regime)",
            priority=prio,
            plain_english_summary=(
                f"Your dataset has a large number of steps ({n_raw}) combined with a {scale_ratio:.2f}x scale "
                f"spread (skewness = {skew:.2f}). Auto mode selected the Lugannani-Rice Saddlepoint method, "
                "which scales linearly with the number of steps, cannot underflow, and accurately captures "
                "both the center and extreme tails in under 1 ms."
            ),
            technical_rationale=(
                f"Large n_coalesced={n_coal} > 25 with scale_ratio={scale_ratio:.2f} > 8.0 or skewness={skew:.3f} >= 0.35: "
                "SaddlepointEngine avoids O(n K^2) series cost and polynomial tail oscillation while maintaining O(n) complexity."
            ),
            expected_accuracy="High (CDF error ~3e-5, quantile error < 0.005%)",
            expected_speed="~0.5 - 1.0 ms",
            alternative_suggestion="Large array queries (>= 5,000 points) are automatically routed to 'fft' splines for speed.",
            data_profile=data_profile,
            auto_tuned_params={},
        )

    def _resolve_method(self, method: str) -> str:
        """Resolves `'auto'` to the best concrete engine for the current parameters."""
        if method != "auto":
            return method
        return self._auto_decision.selected_method

    def _get_engine(self, method: str) -> BaseGammaEngine:
        norm_req = self._normalize_method_name(method)
        m = self._resolve_method(norm_req)
        is_auto = norm_req == "auto"
        tuned = self._auto_decision.auto_tuned_params if is_auto else {}

        if m not in self._engines:
            if m == "moschopoulos":
                eff_k = int(tuned.get("effective_accuracy_k", self.accuracy_k))
                self._engines[m] = MoschopoulosEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    max_terms=eff_k,
                    tol=self.tol,
                    adaptive=self.adaptive,
                )
            elif m == "moschopoulos_quad":
                eff_k = int(tuned.get("effective_accuracy_k", self.accuracy_k))
                self._engines[m] = MoschopoulosQuadEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    max_terms=eff_k,
                    tol=self.tol,
                    adaptive=self.adaptive,
                )
            elif m == "barnabani":
                eff_k = int(tuned.get("effective_accuracy_k", max(300, self.accuracy_k)))
                self._engines[m] = BarnabaniEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    max_terms=eff_k,
                )
            elif m == "laguerre":
                self._engines[m] = LaguerreEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    order=6,
                )
            elif m == "saddlepoint":
                self._engines[m] = SaddlepointEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                )
            elif m == "fft":
                eff_grid = int(tuned.get("effective_fft_grid_size", self.fft_grid_size))
                eff_freq = int(tuned.get("effective_fft_n_freq", self.fft_n_freq))
                self._engines[m] = CharacteristicFunctionFFTEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    grid_size=eff_grid,
                    n_freq=eff_freq,
                )
            elif m == "moment_matching":
                self._engines[m] = MomentMatchingEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    order=3,
                )
            elif m == "welch_satterthwaite":
                self._engines[m] = MomentMatchingEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    order=2,
                )
            elif m == "monte_carlo":
                self._engines[m] = MonteCarloEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    n_samples=self.mc_samples,
                    mode="empirical",
                    random_state=self.random_state,
                )
            elif m == "metamodel":
                self._engines[m] = MonteCarloEngine(
                    self.valid_alphas,
                    self.valid_betas,
                    total_loc=self.total_loc,
                    n_samples=self.mc_samples,
                    mode="metamodel",
                    random_state=self.random_state,
                )
            else:
                raise ValueError(f"Unsupported engine {m!r}")
        return self._engines[m]

    def _get_engine_for_query(
        self,
        query: Union[float, Sequence[float], np.ndarray],
        method: Optional[str] = None,
    ) -> BaseGammaEngine:
        """
        Return the engine for a query, automatically routing massive batch arrays
        (>= 5,000 points) to `'fft'` or `'laguerre'` when in `'auto'` mode so users
        never experience slow per-point root-finding on large grids.
        """
        if method is not None:
            return self._get_engine(method)
        if (
            self.requested_method == "auto"
            and self.auto_priority == "balanced"
            and self.active_method in ("saddlepoint", "moschopoulos_quad")
            and not np.isscalar(query)
            and np.size(query) >= 5000
        ):
            if self.skewness < 0.35 and self.scale_ratio <= 8.0:
                return self._get_engine("laguerre")
            return self._get_engine("fft")
        assert self._engine is not None
        return self._engine

    # ==========================================================================
    # Core Public Methods Specified in `toolbox.stats.GammaLinearCombination`
    # ==========================================================================

    def pdf(
        self,
        x: Union[float, Sequence[float], np.ndarray],
        method: Optional[str] = None,
    ) -> Union[float, np.ndarray]:
        """
        Return probability density f(x) at point(s) `x`.

        Parameters
        ----------
        x : float or array-like
            Evaluation point(s).
        method : str, optional
            Override the instance's default engine for this call
            (e.g., `'moschopoulos'`, `'saddlepoint'`, `'fft'`, `'moment_matching'`).

        Returns
        -------
        float or np.ndarray
        """
        if self.dist_type == 1:
            eng = self._get_engine_for_query(x, method=method)
            return eng.pdf(x)

        is_scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        if self.dist_type == 0:
            single = self.components[self.valid_indices[0]]
            res = sts.gamma.pdf(
                x_arr, single.effective_alpha, loc=self.total_loc, scale=single.effective_beta
            )
        elif self.dist_type == -1:
            res = np.where(np.isclose(x_arr, self.total_loc, atol=1e-5), 1.0, 0.0)
        else:
            res = np.full_like(x_arr, np.nan)
        return float(res[0]) if is_scalar else res

    def cdf(
        self,
        x: Union[float, Sequence[float], np.ndarray],
        method: Optional[str] = None,
    ) -> Union[float, np.ndarray]:
        """
        Return cumulative probability F(x) = Pr(Y <= x) at point(s) `x`.

        Parameters
        ----------
        x : float or array-like
            Evaluation point(s).
        method : str, optional
            Override the instance's default engine for this call.

        Returns
        -------
        float or np.ndarray
        """
        if self.dist_type == 1:
            eng = self._get_engine_for_query(x, method=method)
            return eng.cdf(x)

        is_scalar = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        if self.dist_type == 0:
            single = self.components[self.valid_indices[0]]
            res = sts.gamma.cdf(
                x_arr, single.effective_alpha, loc=self.total_loc, scale=single.effective_beta
            )
        elif self.dist_type == -1:
            res = np.where(x_arr >= self.total_loc, 1.0, 0.0)
        else:
            res = np.full_like(x_arr, np.nan)
        return float(res[0]) if is_scalar else res

    def percentile(
        self,
        q: Union[float, Sequence[float], np.ndarray],
        method: Optional[str] = None,
    ) -> Union[float, np.ndarray]:
        """
        Return percentage value (quantile / inverse CDF) for probability `q` in [0, 1].

        Parameters
        ----------
        q : float or array-like in [0, 1]
            Cumulative probability / quantile level(s), e.g., `0.5` for median,
            `0.95` for 95th percentile.
        method : str, optional
            Override the instance's default engine for this call.

        Returns
        -------
        float or np.ndarray
        """
        if self.dist_type == 1:
            eng = self._get_engine_for_query(q, method=method)
            return eng.ppf(q)

        is_scalar = np.isscalar(q)
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValueError("Quantiles q must be in [0, 1].")

        if self.dist_type == 0:
            single = self.components[self.valid_indices[0]]
            res = sts.gamma.ppf(
                q_arr, single.effective_alpha, loc=self.total_loc, scale=single.effective_beta
            )
        elif self.dist_type == -1:
            res = np.full_like(q_arr, self.total_loc)
        else:
            res = np.full_like(q_arr, np.nan)
        return float(res[0]) if is_scalar else res

    def ppf(
        self,
        q: Union[float, Sequence[float], np.ndarray],
        method: Optional[str] = None,
    ) -> Union[float, np.ndarray]:
        """Alias for `percentile(q)` following `scipy.stats` convention."""
        return self.percentile(q, method=method)

    def rv(
        self,
        size: int = 1,
        random_state: Optional[int] = None,
    ) -> Union[float, np.ndarray]:
        """
        Generate exact random variable(s) from the linear combination
            Y = sum_{i=1}^n c_i X_i + L.

        Parameters
        ----------
        size : int, default 1
            Number of random variates to draw.
        random_state : int, optional
            Optional seed for reproducibility. If None, uses the instance's RNG.

        Returns
        -------
        np.ndarray
            1D array of `size` simulated values.
        """
        rng = self.rng if random_state is None else np.random.default_rng(random_state)
        if self.dist_type in (0, 1):
            samples = np.full(size, self.total_loc, dtype=float)
            for a, b in zip(self.valid_alphas, self.valid_betas):
                samples += rng.gamma(shape=a, scale=b, size=size)
            return samples
        if self.dist_type == -1:
            return np.full(size, self.total_loc, dtype=float)
        return np.full(size, np.nan, dtype=float)

    def rvs(
        self,
        size: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Alias for `rv(size, random_state)` following `scipy.stats` convention."""
        return self.rv(size=size, random_state=random_state)

    def plot_pdf(
        self,
        ax=None,
        resolution: int = 400,
        show_components: bool = False,
        show_uniform_bound: bool = False,
        compare_methods: Optional[Sequence[str]] = None,
        show_histogram: bool = False,
        label: Optional[str] = None,
    ):
        """
        Plot the probability density function (PDF) of the Gamma linear combination.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. Creates a new figure if None.
        resolution : int, default 400
            Number of grid points on the horizontal axis.
        show_components : bool, default False
            If True, overlays individual scaled component densities `c_i * X_i`.
        show_uniform_bound : bool, default False
            If True, overlays Moschopoulos's uniform convergence envelope (Eq. 245).
        compare_methods : Sequence[str], optional
            List of additional engine names (e.g. `['saddlepoint', 'moment_matching']`)
            to overlay for visual comparison.
        show_histogram : bool, default False
            If True, overlays a Monte Carlo histogram of `rv(20000)`.
        label : str, optional
            Custom legend label for the primary PDF curve.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8.5, 4.8))

        if self.dist_type not in (0, 1):
            ax.set_title(f"GammaLinearCombination (type={self.dist_type}, loc={self.total_loc})")
            return ax

        upper_x = float(self.percentile(0.995, method="moment_matching" if self.dist_type == 1 else None))
        if show_components:
            lower_x = max(0.0, self.total_loc - (upper_x - self.total_loc) * 0.04)
        else:
            p001 = float(self.percentile(0.001, method="moment_matching" if self.dist_type == 1 else None))
            span = max(1e-6, upper_x - p001)
            lower_x = max(0.0, self.total_loc, p001 - 0.1 * span)
        x_vals = np.linspace(lower_x, upper_x, resolution)

        if show_histogram:
            mc_draws = self.rv(size=20000, random_state=42)
            ax.hist(mc_draws, bins=60, density=True, alpha=0.25, color="steelblue", label="Monte Carlo (N=20k)")

        primary_label = label or f"GammaLinearCombination ({self.active_method})"
        y_vals = self.pdf(x_vals)
        ax.plot(x_vals, y_vals, "r-", lw=2.4, label=primary_label)

        if compare_methods and self.dist_type == 1:
            styles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
            for i, m in enumerate(compare_methods):
                y_m = self.pdf(x_vals, method=m)
                ax.plot(x_vals, y_m, linestyle=styles[i % len(styles)], lw=1.8, label=f"Method: {m}")

        if show_uniform_bound and self.dist_type == 1:
            bound_y = self.pdf_uniform_bound(x_vals)
            ax.plot(x_vals, bound_y, "k:", lw=1.5, alpha=0.75, label="Moschopoulos Uniform Bound")

        if show_components and self.dist_type == 1:
            for idx in self.valid_indices:
                comp = self.components[idx]
                comp_y = sts.gamma.pdf(
                    x_vals, comp.effective_alpha, loc=comp.effective_loc, scale=comp.effective_beta
                )
                ax.plot(
                    x_vals,
                    comp_y,
                    "--",
                    alpha=0.45,
                    label=f"{comp.name}: Gamma(α={comp.effective_alpha:.2f}, cβ={comp.effective_beta:.2f})",
                )

        ax.set_xlabel("y (Linear Combination Value)")
        ax.set_ylabel("Probability Density f(y)")
        ax.set_title("GammaLinearCombination Probability Density Function")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        return ax

    # ==========================================================================
    # Theoretical Functions from the Document (MGF, CGF, Uniform Bound)
    # ==========================================================================

    def mgf(self, t: Union[float, Sequence[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Exact Moment Generating Function (MGF) from Eq. (191) of the documentation:
            M(t) = exp(total_loc * t) * prod_{i=1}^n (1 - c_i * beta_i * t)^(-alpha_i)
        valid for t < 1 / max(c_i * beta_i).
        """
        return np.exp(self.cgf(t))

    def cgf(self, t: Union[float, Sequence[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Exact Cumulant Generating Function (CGF) log M(t):
            K(t) = total_loc * t - sum_{i=1}^n alpha_i * log(1 - c_i * beta_i * t)
        valid for t < 1 / max(c_i * beta_i).
        """
        is_scalar = np.isscalar(t)
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.full_like(t_arr, np.nan, dtype=float)
        if self.dist_type not in (0, 1):
            return float(out[0]) if is_scalar else out

        beta_max = float(np.max(self.valid_betas))
        valid = t_arr < (1.0 / beta_max)
        if np.any(valid):
            tv = t_arr[valid]
            denom = 1.0 - tv[:, None] * self.valid_betas[None, :]
            out[valid] = self.total_loc * tv - np.dot(np.log(denom), self.valid_alphas)
        return float(out[0]) if is_scalar else out

    def pdf_uniform_bound(self, x: Union[float, Sequence[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the Moschopoulos uniform convergence upper bound from Eq. (245)
        of the documentation:
            f_bound(y) = (C * beta_1^(-rho) / Gamma(rho)) * y^(rho - 1) * exp(-y * (1 - b) / beta_1)
        where b = max_{2 <= j <= n} (1 - beta_1 / beta_j).
        """
        if self.dist_type != 1:
            return self.pdf(x)
        mosch: MoschopoulosEngine = self._get_engine("moschopoulos")  # type: ignore[assignment]
        return mosch.pdf_uniform_bound(x)

    # ==========================================================================
    # Exact Cumulants, Moments & Diagnostics
    # ==========================================================================

    @property
    def mean(self) -> float:
        """Exact mean E[Y] = sum_{i=1}^n c_i * alpha_i * beta_i + total_loc."""
        if self.dist_type in (0, 1):
            return float(np.sum(self.valid_alphas * self.valid_betas)) + self.total_loc
        if self.dist_type == -1:
            return self.total_loc
        return np.nan

    @property
    def variance(self) -> float:
        """Exact variance Var[Y] = sum_{i=1}^n c_i^2 * alpha_i * beta_i^2."""
        if self.dist_type in (0, 1):
            return float(np.sum(self.valid_alphas * (self.valid_betas ** 2)))
        if self.dist_type == -1:
            return 0.0
        return np.nan

    @property
    def std(self) -> float:
        """Exact standard deviation sqrt(Var[Y])."""
        v = self.variance
        return math.sqrt(v) if (not np.isnan(v) and v >= 0.0) else np.nan

    @property
    def skewness(self) -> float:
        """Exact skewness gamma_1 = kappa_3 / kappa_2^(3/2)."""
        if self.dist_type in (0, 1) and self.variance > 0:
            k3 = float(2.0 * np.sum(self.valid_alphas * (self.valid_betas ** 3)))
            return k3 / (self.variance ** 1.5)
        return np.nan

    @property
    def scale_ratio(self) -> float:
        """Ratio max(c_i * beta_i) / min(c_i * beta_i) governing Moschopoulos convergence rate."""
        if self.dist_type in (0, 1) and len(self.valid_betas) > 0:
            return float(np.max(self.valid_betas) / np.min(self.valid_betas))
        return np.nan

    @property
    def auto_decision(self) -> AutoSelectionDecision:
        """
        Return the structured `AutoSelectionDecision` describing the input data
        profile, detected regime, chosen engine, and plain-English explanation.
        """
        return self._auto_decision

    def explain_auto_selection(self, priority: Optional[str] = None) -> str:
        """
        Return a plain-English report explaining which method is automatically
        selected for this dataset and why (designed for users without a
        statistical background).

        Parameters
        ----------
        priority : {'balanced', 'accuracy', 'speed', 'tail_risk', 'batch'}, optional
            Optionally preview the automatic selection for a different user goal.

        Returns
        -------
        str
            Formatted human-readable report.
        """
        dec = self._auto_decision if priority is None else self._build_auto_decision(priority)
        return dec.format_report()

    def overview(self) -> Dict[str, Any]:
        """Return a structured dictionary summarizing the distribution and active engine."""
        quality_map = {
            1: f"Linear combination of {len(self.valid_indices)} independent Gamma variables ({self.active_method})",
            0: "Single valid Gamma variable with aggregated location shift",
            -1: "Deterministic constant shift only (zero variance)",
            -2: "Insufficient information / empty parameters",
        }
        info: Dict[str, Any] = {
            "type": self.dist_type,
            "quality_description": quality_map[self.dist_type],
            "requested_method": self.requested_method,
            "active_method": self.active_method,
            "auto_priority": self.auto_priority,
            "auto_regime": self._auto_decision.regime_id,
            "auto_plain_english": self._auto_decision.plain_english_summary,
            "parameterization": self.parameterization,
            "total_segments": len(self.components),
            "valid_segments_count": len(self.valid_indices),
            "total_location_shift": self.total_loc,
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
            "skewness": self.skewness,
            "scale_ratio": self.scale_ratio,
        }
        if self._engine is not None:
            info["engine_metadata"] = self._engine.metadata()
        return info

    # ==========================================================================
    # Effectiveness vs. Efficiency Benchmarking & Method Recommendation
    # ==========================================================================

    def compare_methods(
        self,
        methods: Sequence[str] = (
            "moschopoulos",
            "moschopoulos_quad",
            "barnabani",
            "laguerre",
            "saddlepoint",
            "fft",
            "moment_matching",
            "welch_satterthwaite",
            "monte_carlo",
            "metamodel",
        ),
        n_eval_points: int = 100,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90, 0.95, 0.99),
    ) -> List[Dict[str, Any]]:
        """
        Benchmark multiple implementation engines on this distribution instance
        to compare **Effectiveness (accuracy)** vs. **Efficiency (execution speed)**.

        Uses a high-precision reference (Moschopoulos with K=500 terms if scale_ratio <= 15,
        or high-resolution Characteristic Function inversion if scale_ratio > 15) as ground truth.

        Returns
        -------
        List[Dict[str, Any]]
            List of benchmark records with timing (ms) and error metrics for each method.
        """
        if self.dist_type != 1:
            raise ValueError("compare_methods requires at least 2 valid Gamma components (dist_type == 1).")

        # Build high-accuracy reference ground truth
        if self.scale_ratio <= 15.0:
            ref_engine: BaseGammaEngine = MoschopoulosEngine(
                self.valid_alphas, self.valid_betas, total_loc=self.total_loc, max_terms=600, tol=1e-15
            )
        else:
            ref_engine = CharacteristicFunctionFFTEngine(
                self.valid_alphas, self.valid_betas, total_loc=self.total_loc, grid_size=1024, n_freq=4096
            )

        q_arr = np.asarray(quantiles, dtype=float)
        ref_ppf = ref_engine.ppf(q_arr)
        x_grid = np.linspace(float(ref_ppf[0]) * 0.7, float(ref_ppf[-1]) * 1.15, n_eval_points)
        ref_pdf = ref_engine.pdf(x_grid)
        ref_cdf = ref_engine.cdf(x_grid)

        results: List[Dict[str, Any]] = []
        for m in methods:
            # Clear cached engine to measure cold initialization time accurately
            norm_m = self._normalize_method_name(m)
            self._engines.pop(norm_m, None)

            t0 = time.perf_counter()
            eng = self._get_engine(norm_m)
            init_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            pdf_vals = eng.pdf(x_grid)
            pdf_ms = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            cdf_vals = eng.cdf(x_grid)
            cdf_ms = (time.perf_counter() - t2) * 1000.0

            t3 = time.perf_counter()
            ppf_vals = eng.ppf(q_arr)
            ppf_ms = (time.perf_counter() - t3) * 1000.0

            pdf_max_err = float(np.max(np.abs(pdf_vals - ref_pdf)))
            cdf_max_err = float(np.max(np.abs(cdf_vals - ref_cdf)))
            ppf_rel_err_pct = float(np.max(np.abs(ppf_vals - ref_ppf) / np.maximum(ref_ppf, 1e-12)) * 100.0)

            results.append(
                {
                    "method": norm_m,
                    "init_ms": round(init_ms, 3),
                    "pdf_ms": round(pdf_ms, 3),
                    "cdf_ms": round(cdf_ms, 3),
                    "ppf_ms": round(ppf_ms, 3),
                    "total_ms": round(init_ms + pdf_ms + cdf_ms + ppf_ms, 3),
                    "pdf_max_abs_err": pdf_max_err,
                    "cdf_max_abs_err": cdf_max_err,
                    "ppf_max_rel_err_pct": ppf_rel_err_pct,
                    "p50": float(ppf_vals[list(q_arr).index(0.50)]) if 0.50 in q_arr else float(ppf_vals[0]),
                    "p95": float(ppf_vals[list(q_arr).index(0.95)]) if 0.95 in q_arr else float(ppf_vals[-1]),
                }
            )

        return results

    def recommend_method(self, priority: str = "balanced") -> Dict[str, Any]:
        """
        Recommend the optimal computational engine based on user priority and
        the mathematical properties of the current components.

        Parameters
        ----------
        priority : {'balanced', 'accuracy', 'exact', 'speed', 'tail_risk', 'batch'}, default 'balanced'
            - `'balanced'`: Best automatic trade-off across all 6 data regimes.
            - `'accuracy'` (or `'exact'`): Highest analytical precision regardless of cost.
            - `'speed'` (or `'realtime'`): Fastest closed-form evaluation (< 0.1 ms).
            - `'tail_risk'` (or `'sla'`): Best relative accuracy at extreme quantiles (P99, P99.9).
            - `'batch'` (or `'large_grid'`): Fastest amortized queries for > 5,000 evaluation points.

        Returns
        -------
        Dict[str, Any]
            Recommendation containing `'recommended_method'`, `'plain_english_summary'`,
            `'rationale'`, `'expected_accuracy'`, `'expected_speed'`, and `'data_profile'`.
        """
        dec = self._build_auto_decision(priority)
        return {
            "priority": priority,
            "normalized_priority": dec.priority,
            "recommended_method": dec.selected_method,
            "regime_id": dec.regime_id,
            "regime_label": dec.regime_label,
            "plain_english_summary": dec.plain_english_summary,
            "rationale": dec.technical_rationale,
            "expected_accuracy": dec.expected_accuracy,
            "expected_speed": dec.expected_speed,
            "alternative_suggestion": dec.alternative_suggestion,
            "n_valid_components": len(self.valid_indices),
            "scale_ratio": self.scale_ratio,
            "data_profile": dict(dec.data_profile),
            "auto_tuned_params": dict(dec.auto_tuned_params),
        }

    def __repr__(self) -> str:
        mean_str = f"{self.mean:.4f}" if not np.isnan(self.mean) else "NaN"
        std_str = f"{self.std:.4f}" if not np.isnan(self.std) else "NaN"
        mode_tag = (
            f"auto->{self.active_method}"
            if self.requested_method == "auto"
            else self.active_method
        )
        return (
            f"GammaLinearCombination(method={mode_tag!r}, "
            f"type={self.dist_type}, terms={len(self.valid_indices)}/{len(self.components)}, "
            f"mean={mean_str}, std={std_str})"
        )


# ==============================================================================
# Helper Functions for Empirical Fitting & Convenience Construction
# ==============================================================================

def clean_samples_iqr(
    data: Sequence[float],
    iqr_rate: float = 1.5,
    min_val: Optional[float] = 0.0,
    max_val: Optional[float] = None,
) -> np.ndarray:
    """Filter empirical observations using Interquartile Range (IQR) fences."""
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if min_val is not None:
        arr = arr[arr > min_val]
    if max_val is not None:
        arr = arr[arr <= max_val]
    if len(arr) < 4 or iqr_rate is None or iqr_rate <= 0:
        return arr
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = q75 - q25
    return arr[(arr >= q25 - iqr_rate * iqr) & (arr <= q75 + iqr_rate * iqr)]


def fit_gamma_component(
    data: Sequence[float],
    weight: float = 1.0,
    floc: Optional[Union[float, str]] = "min_offset",
    floc_offset: float = 0.0001,
    min_samples: int = 2,
    round_digits: int = 4,
    clean_iqr: bool = False,
    iqr_rate: float = 1.5,
    name: Optional[str] = None,
) -> GammaComponent:
    """
    Fit a GammaComponent `Gamma(alpha, beta, loc, weight)` to empirical sample data.
    Gracefully handles empty arrays, single observations, and zero-variance constants.
    """
    if data is None or len(data) == 0:
        return GammaComponent(alpha=np.nan, beta=np.nan, loc=np.nan, weight=weight, name=name)

    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return GammaComponent(alpha=np.nan, beta=np.nan, loc=np.nan, weight=weight, name=name)

    if clean_iqr:
        arr = clean_samples_iqr(arr, iqr_rate=iqr_rate)
        if len(arr) == 0:
            return GammaComponent(alpha=np.nan, beta=np.nan, loc=np.nan, weight=weight, name=name)

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    if len(arr) < min_samples or np.isclose(min_val, max_val, atol=1e-5):
        loc_val = round(max(0.0, min_val), round_digits) if round_digits > 0 else max(0.0, min_val)
        return GammaComponent(alpha=np.nan, beta=np.nan, loc=loc_val, weight=weight, name=name)

    floc_arg: Optional[float] = None
    if floc == "min_offset":
        floc_arg = max(0.0, min_val - floc_offset)
    elif floc is not None:
        floc_arg = float(floc)

    try:
        if floc_arg is not None:
            shape, loc_fit, scale = sts.gamma.fit(arr, floc=floc_arg)
        else:
            shape, loc_fit, scale = sts.gamma.fit(arr)
        loc_fit = max(0.0, float(loc_fit))
        shape, scale = float(shape), float(scale)
        if round_digits > 0:
            shape = round(shape, round_digits)
            loc_fit = round(loc_fit, round_digits)
            scale = round(scale, round_digits)
        return GammaComponent(alpha=shape, beta=scale, loc=loc_fit, weight=weight, name=name)
    except Exception:
        loc_val = round(max(0.0, min_val), round_digits) if round_digits > 0 else max(0.0, min_val)
        return GammaComponent(alpha=np.nan, beta=np.nan, loc=loc_val, weight=weight, name=name)


def select_auto_method(
    gammas: Optional[Sequence[Any]] = None,
    *,
    alphas: Optional[Sequence[float]] = None,
    betas: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
    locs: Optional[Sequence[float]] = None,
    parameterization: str = "scale",
    priority: str = "balanced",
) -> AutoSelectionDecision:
    """
    Convenience function for users without a statistical background: inspects
    the input Gamma components and returns an `AutoSelectionDecision` containing
    the selected engine, plain-English explanation, and data regime profile.
    """
    glc = GammaLinearCombination(
        gammas=gammas,
        alphas=alphas,
        betas=betas,
        weights=weights,
        locs=locs,
        parameterization=parameterization,
        method="auto",
        auto_priority=priority,
    )
    return glc.auto_decision

