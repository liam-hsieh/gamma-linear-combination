"""
Command-line benchmark and visual comparison utility for `GammaLinearCombination`.
Demonstrates the effectiveness vs. efficiency trade-off across all implemented methods.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import GammaLinearCombination


def format_benchmark_table(rows: List[Dict[str, Any]]) -> str:
    """Format benchmark results into a clean Markdown-compatible table."""
    header = (
        f"{'Method':<22} | {'Init(ms)':>8} | {'PDF(ms)':>8} | {'CDF(ms)':>8} | "
        f"{'PPF(ms)':>8} | {'PDF MaxErr':>11} | {'CDF MaxErr':>11} | {'PPF RelErr%':>11} | {'P95 Est':>9}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['method']:<22} | "
            f"{r['init_ms']:>8.3f} | "
            f"{r['pdf_ms']:>8.3f} | "
            f"{r['cdf_ms']:>8.3f} | "
            f"{r['ppf_ms']:>8.3f} | "
            f"{r['pdf_max_abs_err']:>11.2e} | "
            f"{r['cdf_max_abs_err']:>11.2e} | "
            f"{r['ppf_max_rel_err_pct']:>10.4f}% | "
            f"{r['p95']:>9.4f}"
        )
    return "\n".join(lines)


def generate_comparison_figure(output_path: pathlib.Path) -> None:
    """Generate a 3-panel visual comparison figure of all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # Panel 1: PDF & Component Breakdown + Uniform Bound (Weighted 4-term linear combination)
    glc_s1 = GammaLinearCombination(
        alphas=[2.2, 3.5, 1.8, 4.0],
        betas=[1.2, 1.8, 2.5, 1.5],
        weights=[1.0, 1.5, 0.8, 1.2],
        locs=[0.5, 1.0, 0.0, 0.5],
        method="moschopoulos",
    )
    glc_s1.plot_pdf(
        ax=axes[0],
        show_components=True,
        show_histogram=True,
        compare_methods=["barnabani", "laguerre", "saddlepoint"],
        label="Moschopoulos (Exact Series)",
    )
    axes[0].set_title("Scenario 1: Weighted Linear Combination\n$Y = \\sum_{i=1}^4 c_i X_i + L$")

    # Panel 2: CDF Comparison on High Scale-Ratio Regime (scale ratio = 25x)
    glc_s2 = GammaLinearCombination(
        alphas=[1.5, 2.0, 1.2, 0.8],
        betas=[0.4, 1.5, 4.0, 10.0],
        weights=[1.0, 1.0, 1.0, 1.0],
        method="fft",
    )
    x_s2 = np.linspace(0.1, glc_s2.percentile(0.99, method="fft"), 250)
    for m, style in [
        ("fft", "k-"),
        ("moschopoulos", "r--"),
        ("barnabani", "c--"),
        ("laguerre", "g:"),
        ("saddlepoint", "b-."),
        ("moment_matching", "m:"),
    ]:
        axes[1].plot(x_s2, glc_s2.cdf(x_s2, method=m), style, lw=2.0, label=m)
    axes[1].set_title("Scenario 2: High Scale Spread ($\\beta_{\\max}/\\beta_{\\min} = 25\\times$)\nCumulative Distribution F(y)")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Cumulative Probability F(y)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Panel 3: Wafer Fab Cycle Time (100 independent Gamma segments)
    rng = np.random.default_rng(123)
    fab_alphas = rng.uniform(1.5, 5.0, size=100)
    fab_betas = rng.uniform(0.5, 3.0, size=100)
    fab_weights = rng.uniform(0.8, 1.5, size=100)
    glc_s3 = GammaLinearCombination(
        alphas=fab_alphas,
        betas=fab_betas,
        weights=fab_weights,
        method="saddlepoint",
    )
    glc_s3.plot_pdf(
        ax=axes[2],
        show_histogram=True,
        compare_methods=["moschopoulos", "barnabani", "laguerre"],
        label="Saddlepoint (Daniels/Lugannani-Rice)",
    )
    axes[2].set_title("Scenario 3: Wafer Fab End-to-End Cycle Time\n($n = 100$ Independent Gamma Segments)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    print("=" * 105)
    print(" toolbox.stats.GammaLinearCombination — Effectiveness vs. Efficiency Benchmark")
    print("=" * 105)

    # Scenario 1: Moderate Weighted Linear Combination (n=4, scale ratio ~ 2.7x)
    s1 = GammaLinearCombination(
        alphas=[2.2, 3.5, 1.8, 4.0],
        betas=[1.2, 1.8, 2.5, 1.5],
        weights=[1.0, 1.5, 0.8, 1.2],
        locs=[0.5, 1.0, 0.0, 0.5],
        method="moschopoulos",
    )
    print(f"\n[Scenario 1] 4-Segment Weighted Combination (mean={s1.mean:.3f}, std={s1.std:.3f}, scale_ratio={s1.scale_ratio:.2f}x)")
    print(format_benchmark_table(s1.compare_methods()))
    rec1 = s1.recommend_method("balanced")
    print(f" -> Recommended ('balanced'): {rec1['recommended_method']} ({rec1['rationale']})")

    # Scenario 2: High Scale Spread (n=5, scale ratio = 30x)
    s2 = GammaLinearCombination(
        alphas=[1.5, 2.0, 1.2, 0.9, 1.1],
        betas=[0.5, 1.5, 4.0, 8.0, 15.0],
        method="auto",
    )
    print(f"\n[Scenario 2] High Scale Spread (mean={s2.mean:.3f}, std={s2.std:.3f}, scale_ratio={s2.scale_ratio:.2f}x)")
    print(format_benchmark_table(s2.compare_methods()))
    rec2 = s2.recommend_method("balanced")
    print(f" -> Recommended ('balanced'): {rec2['recommended_method']} ({rec2['rationale']})")

    # Scenario 3: Wafer Fab Cycle Time (n=100 segments)
    rng = np.random.default_rng(123)
    s3 = GammaLinearCombination(
        alphas=rng.uniform(1.5, 5.0, size=100),
        betas=rng.uniform(0.5, 3.0, size=100),
        weights=rng.uniform(0.8, 1.5, size=100),
        method="auto",
    )
    print(f"\n[Scenario 3] Wafer Fab Cycle Time n=100 Segments (mean={s3.mean:.3f}, std={s3.std:.3f}, scale_ratio={s3.scale_ratio:.2f}x)")
    print(format_benchmark_table(s3.compare_methods()))
    rec3 = s3.recommend_method("balanced")
    print(f" -> Recommended ('balanced'): {rec3['recommended_method']} ({rec3['rationale']})")

    out_img = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "method_comparison.png"
    generate_comparison_figure(out_img)
    print(f"\nSaved visual comparison figure to: {out_img}")


if __name__ == "__main__":
    main()
