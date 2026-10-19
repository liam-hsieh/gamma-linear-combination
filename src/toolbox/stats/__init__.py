"""
toolbox.stats
=============
Statistical distributions and linear combination tools.
"""

from gamma_linear_combination import (
    AVAILABLE_METHODS,
    AVAILABLE_PRIORITIES,
    AutoSelectionDecision,
    GammaComponent,
    GammaLinearCombination,
    clean_samples_iqr,
    fit_gamma_component,
    select_auto_method,
)

__all__ = [
    "GammaLinearCombination",
    "GammaComponent",
    "AutoSelectionDecision",
    "AVAILABLE_METHODS",
    "AVAILABLE_PRIORITIES",
    "select_auto_method",
    "fit_gamma_component",
    "clean_samples_iqr",
]
