"""
uncertainty_metrics.py
----------------------
Implements evaluation metrics for prediction intervals,
including PICP, MIW, and Interval Score.
"""

import numpy as np

def calculate_picp(actuals: np.ndarray, lowers: np.ndarray, uppers: np.ndarray) -> float:
    """
    Computes Prediction Interval Coverage Probability (PICP).
    """
    covered = (actuals >= lowers) & (actuals <= uppers)
    return float(np.mean(covered))

def calculate_miw(lowers: np.ndarray, uppers: np.ndarray) -> float:
    """
    Computes Mean Interval Width (MIW).
    """
    return float(np.mean(uppers - lowers))

def calculate_interval_score(actuals: np.ndarray, lowers: np.ndarray, uppers: np.ndarray, alpha: float) -> float:
    """
    Computes the standard Winkler Interval Score.
    """
    widths = uppers - lowers
    lower_penalties = (2.0 / alpha) * (lowers - actuals) * (actuals < lowers)
    upper_penalties = (2.0 / alpha) * (actuals - uppers) * (actuals > uppers)
    return float(np.mean(widths + lower_penalties + upper_penalties))

def calculate_coverage_error(empirical_coverage: float, nominal_coverage: float) -> float:
    """
    Computes absolute coverage error.
    """
    return abs(empirical_coverage - nominal_coverage)
