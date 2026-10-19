"""
gamma_linear_combination
========================
Multi-engine implementation of `toolbox.stats.GammaLinearCombination` for computing
the probability density (PDF), cumulative probability (CDF), quantiles (percentile/PPF),
and random variates (RV) of a linear combination of independent Gamma random variables:
    Y = sum_{i=1}^n c_i X_i + L,   where X_i ~ Gamma(alpha_i, beta_i), c_i > 0.
"""

from .core import (
    AVAILABLE_METHODS,
    AVAILABLE_PRIORITIES,
    AutoSelectionDecision,
    GammaComponent,
    GammaLinearCombination,
    clean_samples_iqr,
    fit_gamma_component,
    select_auto_method,
)
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

__all__ = [
    "AVAILABLE_METHODS",
    "AVAILABLE_PRIORITIES",
    "AutoSelectionDecision",
    "GammaComponent",
    "GammaLinearCombination",
    "clean_samples_iqr",
    "fit_gamma_component",
    "select_auto_method",
    "coalesce_equal_scales",
    "BaseGammaEngine",
    "MoschopoulosEngine",
    "MoschopoulosQuadEngine",
    "BarnabaniEngine",
    "LaguerreEngine",
    "SaddlepointEngine",
    "CharacteristicFunctionFFTEngine",
    "MomentMatchingEngine",
    "MonteCarloEngine",
]

__version__ = "0.3.0"
