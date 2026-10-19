"""
toolbox.stats.gamma_linear_combination
======================================
Direct module alias for `toolbox.stats.GammaLinearCombination`.
"""

from gamma_linear_combination import (
    AVAILABLE_METHODS,
    GammaComponent,
    GammaLinearCombination,
    clean_samples_iqr,
    fit_gamma_component,
)

__all__ = [
    "GammaLinearCombination",
    "GammaComponent",
    "AVAILABLE_METHODS",
    "fit_gamma_component",
    "clean_samples_iqr",
]
