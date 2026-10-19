"""
Comprehensive test suite for `toolbox.stats.GammaLinearCombination`
and `gamma_linear_combination`.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import scipy.stats as sts

from gamma_linear_combination import (
    AVAILABLE_METHODS,
    AVAILABLE_PRIORITIES,
    AutoSelectionDecision,
    GammaComponent,
    GammaLinearCombination,
    clean_samples_iqr,
    coalesce_equal_scales,
    fit_gamma_component,
    select_auto_method,
)
from toolbox.stats import GammaLinearCombination as ToolboxGLC
from toolbox.stats.gamma_linear_combination import GammaLinearCombination as ToolboxSubmoduleGLC


class TestGammaLinearCombination:

    def test_namespace_exports_match_documentation(self):
        """Verify `toolbox.stats.GammaLinearCombination` matches the documented import path."""
        assert ToolboxGLC is GammaLinearCombination
        assert ToolboxSubmoduleGLC is GammaLinearCombination

        glc = ToolboxGLC([(2.0, 1.5), (3.0, 2.0)])
        assert hasattr(glc, "pdf")
        assert hasattr(glc, "cdf")
        assert hasattr(glc, "plot_pdf")
        assert hasattr(glc, "percentile")
        assert hasattr(glc, "rv")

    def test_scale_coalescing_engineering_optimization(self):
        """
        Verify that components with identical effective scales c_i * beta_i are
        automatically coalesced into a single Gamma(sum(alpha_i), beta) term.
        """
        alphas = np.array([1.5, 2.5, 3.0, 1.0])
        betas = np.array([2.0, 4.0, 2.0, 4.0])
        merged_a, merged_b = coalesce_equal_scales(alphas, betas)
        np.testing.assert_allclose(merged_b, [2.0, 4.0])
        np.testing.assert_allclose(merged_a, [4.5, 3.5])

    def test_equal_effective_scale_reduces_to_exact_gamma(self):
        """
        When c_i * beta_i = theta_common for all i, the linear combination
        Y = sum(c_i * X_i) follows an exact Gamma(sum(alpha_i), theta_common).
        Here:
          c_1 * beta_1 = 2.0 * 1.5 = 3.0
          c_2 * beta_2 = 0.5 * 6.0 = 3.0
          c_3 * beta_3 = 1.0 * 3.0 = 3.0
        Sum ~ Gamma(alpha = 2 + 3.5 + 1.5 = 7.0, scale = 3.0, loc = 2.0).
        """
        alphas = [2.0, 3.5, 1.5]
        betas = [1.5, 6.0, 3.0]
        weights = [2.0, 0.5, 1.0]
        locs = [0.5, 1.0, 0.5]  # effective loc = 2*0.5 + 0.5*1.0 + 1*0.5 = 2.0

        glc = GammaLinearCombination(
            alphas=alphas,
            betas=betas,
            weights=weights,
            locs=locs,
            method="moschopoulos",
        )
        assert math.isclose(glc.total_loc, 2.0)
        assert math.isclose(glc.mean, 7.0 * 3.0 + 2.0)
        assert math.isclose(glc.variance, 7.0 * (3.0 ** 2))
        # Coalescing reduces 3 equal-scale terms to 1 term with 0 series iterations!
        assert glc.overview()["engine_metadata"]["coalesced_terms"] == 1

        x_pts = np.array([8.0, 15.0, 23.0, 35.0])
        exact_pdf = sts.gamma.pdf(x_pts, a=7.0, loc=2.0, scale=3.0)
        exact_cdf = sts.gamma.cdf(x_pts, a=7.0, loc=2.0, scale=3.0)
        q_pts = np.array([0.1, 0.5, 0.9, 0.99])
        exact_ppf = sts.gamma.ppf(q_pts, a=7.0, loc=2.0, scale=3.0)

        for method in (
            "moschopoulos",
            "moschopoulos_quad",
            "barnabani",
            "laguerre",
            "moment_matching",
            "welch_satterthwaite",
        ):
            np.testing.assert_allclose(glc.pdf(x_pts, method=method), exact_pdf, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(glc.cdf(x_pts, method=method), exact_cdf, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(glc.percentile(q_pts, method=method), exact_ppf, rtol=1e-5)

        # Saddlepoint and FFT are also highly accurate on identical-scale Gamma
        np.testing.assert_allclose(glc.cdf(x_pts, method="saddlepoint"), exact_cdf, atol=2e-3)
        np.testing.assert_allclose(glc.cdf(x_pts, method="fft"), exact_cdf, atol=2e-3)

    def test_shape_rate_parameterization_appendix(self):
        """
        Verify Appendix formula:
        Under shape-rate parameterization X ~ Gamma(alpha, rate),
        c * X ~ Gamma(alpha, rate / c), which has effective scale = c / rate.
        """
        alphas = [2.5, 4.0]
        rates = [0.5, 2.0]   # scales are 2.0 and 0.5
        weights = [1.5, 3.0] # effective scales are 3.0 and 1.5

        glc_rate = GammaLinearCombination(
            alphas=alphas,
            betas=rates,
            weights=weights,
            parameterization="rate",
        )
        glc_scale = GammaLinearCombination(
            alphas=alphas,
            betas=[2.0, 0.5],
            weights=weights,
            parameterization="scale",
        )

        assert math.isclose(glc_rate.mean, glc_scale.mean)
        assert math.isclose(glc_rate.variance, glc_scale.variance)
        x_grid = np.linspace(2.0, 25.0, 15)
        np.testing.assert_allclose(glc_rate.pdf(x_grid), glc_scale.pdf(x_grid), rtol=1e-12)
        np.testing.assert_allclose(glc_rate.cdf(x_grid), glc_scale.cdf(x_grid), rtol=1e-12)

    def test_all_ten_engines_consistency_on_heterogeneous_gammas(self):
        """
        Verify that all 10 engines (including `barnabani` and `laguerre`) agree
        closely on PDF, CDF, and Percentiles for a heterogeneous linear combination.
        """
        gammas = [
            {"alpha": 2.0, "beta": 1.2, "weight": 1.5, "loc": 0.5},
            {"alpha": 3.5, "beta": 2.0, "weight": 1.0, "loc": 1.0},
            {"alpha": 1.8, "beta": 1.5, "weight": 2.0, "loc": 0.0},
        ]
        glc = GammaLinearCombination(gammas, method="moschopoulos", accuracy_k=150, mc_samples=40000)

        x_eval = np.array([6.0, 12.0, 18.0, 26.0, 35.0])
        q_eval = np.array([0.10, 0.50, 0.90, 0.95])
        ref_pdf = glc.pdf(x_eval, method="moschopoulos")
        ref_cdf = glc.cdf(x_eval, method="moschopoulos")
        ref_ppf = glc.percentile(q_eval, method="moschopoulos")

        # 1. Moschopoulos Quadrature
        np.testing.assert_allclose(glc.cdf(x_eval, method="moschopoulos_quad"), ref_cdf, atol=1e-6)
        np.testing.assert_allclose(glc.percentile(q_eval, method="moschopoulos_quad"), ref_ppf, rtol=1e-5)

        # 2. Barnabani (2017) Generalized Negative Binomial Mixture
        np.testing.assert_allclose(glc.pdf(x_eval, method="barnabani"), ref_pdf, atol=1e-4)
        np.testing.assert_allclose(glc.cdf(x_eval, method="barnabani"), ref_cdf, atol=5e-4)
        np.testing.assert_allclose(glc.percentile(q_eval, method="barnabani"), ref_ppf, rtol=5e-4)

        # 3. Ha & Provost (2013) 6-Cumulant Generalized Laguerre Expansion
        np.testing.assert_allclose(glc.pdf(x_eval, method="laguerre"), ref_pdf, atol=2e-4)
        np.testing.assert_allclose(glc.cdf(x_eval, method="laguerre"), ref_cdf, atol=2e-4)
        np.testing.assert_allclose(glc.percentile(q_eval, method="laguerre"), ref_ppf, rtol=5e-4)

        # 4. Saddlepoint (Lugannani-Rice / Daniels)
        np.testing.assert_allclose(glc.pdf(x_eval, method="saddlepoint"), ref_pdf, atol=2e-3)
        np.testing.assert_allclose(glc.cdf(x_eval, method="saddlepoint"), ref_cdf, atol=2e-3)
        np.testing.assert_allclose(glc.percentile(q_eval, method="saddlepoint"), ref_ppf, rtol=5e-3)

        # 5. Characteristic Function FFT
        np.testing.assert_allclose(glc.pdf(x_eval, method="fft"), ref_pdf, atol=2e-3)
        np.testing.assert_allclose(glc.cdf(x_eval, method="fft"), ref_cdf, atol=2e-3)
        np.testing.assert_allclose(glc.percentile(q_eval, method="fft"), ref_ppf, rtol=5e-3)

        # 6. 3-Moment Skewness-Matched Gamma (Pearson Type III)
        np.testing.assert_allclose(glc.cdf(x_eval, method="moment_matching"), ref_cdf, atol=5e-3)
        np.testing.assert_allclose(glc.percentile(q_eval, method="moment_matching"), ref_ppf, rtol=1e-2)

        # 7. 2-Moment Welch-Satterthwaite
        np.testing.assert_allclose(glc.cdf(x_eval, method="welch_satterthwaite"), ref_cdf, atol=1.5e-2)
        np.testing.assert_allclose(glc.percentile(q_eval, method="welch_satterthwaite"), ref_ppf, rtol=2.5e-2)

        # 8. Monte Carlo & Metamodel
        np.testing.assert_allclose(glc.cdf(x_eval, method="monte_carlo"), ref_cdf, atol=1.5e-2)
        np.testing.assert_allclose(glc.percentile(q_eval, method="metamodel"), ref_ppf, rtol=3e-2)

    def test_wafer_fab_hundreds_of_segments_no_underflow(self):
        """
        Test the wafer fab scenario from the documentation:
        150 independent Gamma processing & transportation time segments.
        """
        rng = np.random.default_rng(2026)
        n_steps = 150
        alphas = rng.uniform(2.0, 5.0, size=n_steps)
        betas = rng.uniform(0.5, 4.0, size=n_steps)
        weights = rng.uniform(0.8, 1.2, size=n_steps)

        glc = GammaLinearCombination(
            alphas=alphas,
            betas=betas,
            weights=weights,
            method="auto",
            accuracy_k=250,
        )
        assert glc.active_method == "saddlepoint"

        pts = np.array([glc.mean - 1.5 * glc.std, glc.mean, glc.mean + 1.5 * glc.std])
        for m in ("saddlepoint", "moschopoulos", "barnabani", "laguerre", "fft", "moment_matching"):
            p_vals = glc.pdf(pts, method=m)
            c_vals = glc.cdf(pts, method=m)
            assert np.all(np.isfinite(p_vals)) and np.all(p_vals > 0.0)
            assert np.all(np.isfinite(c_vals)) and np.all((c_vals > 0.0) & (c_vals < 1.0))

        p95_mosch = glc.percentile(0.95, method="moschopoulos")
        for m in ("saddlepoint", "barnabani", "laguerre", "fft", "moment_matching"):
            assert math.isclose(glc.percentile(0.95, method=m), p95_mosch, rel_tol=5e-3)

    def test_mgf_cgf_and_uniform_convergence_bound(self):
        """
        Verify exact MGF/CGF equations (191)-(199) and the uniform convergence
        upper bound equation (245) from the documentation.
        """
        glc = GammaLinearCombination([(2.5, 1.2), (3.0, 2.0), (1.5, 1.8)])

        assert math.isclose(glc.mgf(0.0), 1.0, abs_tol=1e-12)
        assert math.isclose(glc.cgf(0.0), 0.0, abs_tol=1e-12)

        eps = 1e-6
        k1_num = (glc.cgf(eps) - glc.cgf(-eps)) / (2.0 * eps)
        k2_num = (glc.cgf(eps) - 2.0 * glc.cgf(0.0) + glc.cgf(-eps)) / (eps ** 2)
        assert math.isclose(k1_num, glc.mean, rel_tol=1e-6)
        assert math.isclose(k2_num, glc.variance, rel_tol=1e-5)

        y_grid = np.linspace(0.5, 35.0, 50)
        pdf_vals = glc.pdf(y_grid, method="moschopoulos")
        bound_vals = glc.pdf_uniform_bound(y_grid)
        assert np.all(bound_vals >= pdf_vals - 1e-12)

    def test_rv_and_plot_pdf(self):
        """Test random variate generation `rv` and `plot_pdf`."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        glc = GammaLinearCombination([(2.0, 1.5), (3.0, 2.5)], weights=[1.2, 0.8])
        draws = glc.rv(size=30000, random_state=123)
        assert len(draws) == 30000
        assert math.isclose(float(np.mean(draws)), glc.mean, rel_tol=2e-2)
        assert math.isclose(float(np.var(draws)), glc.variance, rel_tol=4e-2)

        ax = glc.plot_pdf(
            show_components=True,
            show_uniform_bound=True,
            compare_methods=["barnabani", "laguerre", "saddlepoint"],
        )
        assert ax is not None
        plt.close("all")

    def test_compare_and_recommend_methods(self):
        """Test `compare_methods()` and `recommend_method()`."""
        glc = GammaLinearCombination([(2.0, 1.0), (3.0, 2.0), (1.5, 4.0)])
        bench = glc.compare_methods(n_eval_points=30)
        assert len(bench) == 10
        for row in bench:
            assert "method" in row
            assert row["pdf_max_abs_err"] >= 0.0
            assert row["cdf_max_abs_err"] >= 0.0

        for prio in ("exact", "balanced", "speed", "batch"):
            rec = glc.recommend_method(priority=prio)
            assert rec["recommended_method"] in AVAILABLE_METHODS

    def test_edge_cases_and_empirical_fitting(self):
        """Test Type 0 (single Gamma), Type -1 (constant), Type -2 (empty), and sample fitting."""
        glc_t0 = GammaLinearCombination([(3.0, 1.0, 2.0), (np.nan, 4.0, np.nan)])
        assert glc_t0.dist_type == 0
        assert math.isclose(glc_t0.total_loc, 5.0)
        assert math.isclose(glc_t0.mean, 3.0 * 2.0 + 5.0)

        glc_const = GammaLinearCombination([(np.nan, 3.5, np.nan), (np.nan, 2.5, np.nan)])
        assert glc_const.dist_type == -1
        assert math.isclose(glc_const.mean, 6.0)
        assert glc_const.variance == 0.0
        assert math.isclose(glc_const.percentile(0.5), 6.0)

        glc_empty = GammaLinearCombination([])
        assert glc_empty.dist_type == -2
        assert np.isnan(glc_empty.mean)

        raw = [2.1, 2.3, 2.5, 2.2, 2.8, 3.1, 2.4, 150.0, -5.0]
        cleaned = clean_samples_iqr(raw)
        assert 150.0 not in cleaned and -5.0 not in cleaned
        comp = fit_gamma_component(cleaned, weight=1.5, name="FittedStep")
        assert comp.is_valid
        assert comp.weight == 1.5

    def test_auto_mode_regime_selection_for_non_statistical_users(self):
        """
        Verify that `method='auto'` (the default) automatically inspects input data
        and selects the optimal method across all regimes and user priorities,
        with plain-English explanations via `explain_auto_selection()`.
        """
        # Regime 0: Identical effective scales -> collapses to exact single Gamma ('moschopoulos')
        glc_r0 = GammaLinearCombination(alphas=[2.0, 3.0, 1.5], betas=[2.0, 4.0, 1.0], weights=[2.0, 1.0, 4.0])
        assert glc_r0.active_method == "moschopoulos"
        assert glc_r0.auto_decision.regime_id == "regime_0_identical_scales"
        assert glc_r0.auto_decision.auto_tuned_params["effective_accuracy_k"] == 1

        # Regime I: Small component count & moderate scale spread -> 'moschopoulos'
        glc_r1 = GammaLinearCombination([(2.2, 1.0), (3.5, 1.8), (1.8, 2.5), (4.0, 1.2)], weights=[1.2, 0.8, 1.5, 1.0])
        assert glc_r1.active_method == "moschopoulos"
        assert glc_r1.auto_decision.regime_id == "regime_1_compact_homogeneous"
        report_r1 = glc_r1.explain_auto_selection()
        assert "Automatic Method Selection Report" in report_r1
        assert "moschopoulos" in report_r1

        # Regime II-A: Moderate scale spread (6x-12x) -> 'moschopoulos' (auto-extended) or 'barnabani'
        glc_r2a = GammaLinearCombination(alphas=[8.0, 10.0, 12.0], betas=[1.0, 6.0, 9.5])
        assert glc_r2a.active_method in ("moschopoulos", "barnabani")
        assert glc_r2a.auto_decision.regime_id.startswith("regime_2a_")

        # Regime II-B: Severe bottleneck disparity (20x scale spread, high skewness) -> 'fft'
        glc_r2b_skew = GammaLinearCombination(
            [(1.5, 0.5), (2.0, 1.2), (1.0, 3.5), (2.5, 6.0), (1.2, 10.0)]
        )
        assert glc_r2b_skew.active_method == "fft"
        assert glc_r2b_skew.auto_decision.regime_id == "regime_2b_bottleneck_fft"

        # Regime II-B: High scale spread (15x) but smooth shape / lower skewness -> 'saddlepoint'
        glc_r2b_smooth = GammaLinearCombination(
            alphas=[3.0, 6.0, 8.0, 10.0], betas=[1.0, 8.0, 12.0, 15.0]
        )
        assert glc_r2b_smooth.active_method == "saddlepoint"
        assert glc_r2b_smooth.auto_decision.regime_id == "regime_2b_bottleneck_saddlepoint"

        # Regime III-A: Large balanced multi-step line (n=100, scale_ratio <= 8, low skewness) -> 'laguerre'
        rng = np.random.default_rng(2026)
        glc_r3a = GammaLinearCombination(
            alphas=rng.uniform(1.5, 4.5, size=100),
            betas=rng.uniform(0.8, 2.5, size=100),
            weights=rng.uniform(0.8, 1.5, size=100),
        )
        assert glc_r3a.active_method == "laguerre"
        assert glc_r3a.auto_decision.regime_id == "regime_3a_large_balanced"

        # Regime III-B: Large heterogeneous multi-step line (n=150, scale_ratio > 8) -> 'saddlepoint'
        rng2 = np.random.default_rng(2026)
        glc_r3b = GammaLinearCombination(
            alphas=rng2.uniform(2.0, 5.0, size=150),
            betas=rng2.uniform(0.5, 4.0, size=150),
            weights=rng2.uniform(0.8, 1.2, size=150),
        )
        assert glc_r3b.active_method == "saddlepoint"
        assert glc_r3b.auto_decision.regime_id == "regime_3b_large_heterogeneous"

        # Test dynamic large-batch routing (>= 5,000 points) on a saddlepoint instance
        large_grid = np.linspace(glc_r3b.mean - 2 * glc_r3b.std, glc_r3b.mean + 2 * glc_r3b.std, 5500)
        batch_cdf = glc_r3b.cdf(large_grid)
        assert len(batch_cdf) == 5500 and np.all(np.isfinite(batch_cdf))

        # Test all user-friendly priorities via `select_auto_method`
        for prio in AVAILABLE_PRIORITIES:
            dec = select_auto_method([(1.5, 0.5), (2.0, 1.2), (1.2, 10.0)], priority=prio)
            assert isinstance(dec, AutoSelectionDecision)
            assert dec.selected_method in AVAILABLE_METHODS
            assert len(dec.format_report()) > 100

