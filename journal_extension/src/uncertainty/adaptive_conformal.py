"""
adaptive_conformal.py
---------------------
Implements Scale-Normalized and Volume-Normalized Conformal Prediction
for multi-step lead-time demand uncertainty calibration.
Optimized with vectorized NumPy/Pandas array operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class AdaptiveConformalCalibrator:
    """
    Manages rolling-origin adaptive conformal prediction with scale and volume normalization.
    """

    def __init__(
        self,
        window_size: int = 100,
        scale_window: int = 50,
        min_observations: int = 20,
        epsilon: float = 1e-4
    ):
        self.window_size = window_size
        self.scale_window = scale_window
        self.min_observations = min_observations
        self.epsilon = epsilon

    def compute_conformal_quantiles(self, scores: np.ndarray, nominal_coverages: list) -> Dict[float, float]:
        """
        Computes the standard finite-sample conformal quantiles for multiple coverages simultaneously:
        index = ceil((n + 1) * (1 - alpha))
        """
        n = len(scores)
        quantiles = {}
        if n < self.min_observations:
            for cov in nominal_coverages:
                quantiles[cov] = float(np.quantile(scores, cov)) if n > 0 else 0.0
            return quantiles

        sorted_scores = np.sort(scores)
        for cov in nominal_coverages:
            alpha = 1.0 - cov
            idx_1indexed = int(np.ceil((n + 1) * (1.0 - alpha)))
            idx_0indexed = min(max(idx_1indexed - 1, 0), n - 1)
            quantiles[cov] = float(sorted_scores[idx_0indexed])
        return quantiles

    def compute_conformal_quantile(self, scores: np.ndarray, alpha: float) -> float:
        """
        Computes single finite-sample conformal quantile.
        """
        cov = 1.0 - alpha
        return self.compute_conformal_quantiles(scores, [cov])[cov]

    def estimate_residual_scale(self, eligible_errors: np.ndarray) -> float:
        """
        Estimates the local residual scale (sigma) from historical eligible OOS errors.
        Uses the mean absolute error of the most recent scale_window eligible residuals.
        """
        if len(eligible_errors) == 0:
            return 1.0
        recent = eligible_errors[-self.scale_window:]
        scale = float(np.mean(recent))
        return max(scale, self.epsilon)

    def calibrate_raw(
        self,
        eligible_errors: np.ndarray,
        nominal_coverages: list
    ) -> Dict[float, float]:
        """
        Raw (unnormalized) conformal calibration:
        Returns dictionary of raw quantiles {cov: q}.
        """
        if len(eligible_errors) == 0:
            return {cov: 0.0 for cov in nominal_coverages}
        window = eligible_errors[-self.window_size:]
        return self.compute_conformal_quantiles(window, nominal_coverages)

    def calibrate_scale_normalized(
        self,
        eligible_errors: np.ndarray,
        nominal_coverages: list
    ) -> Dict[float, float]:
        """
        Adaptive Method A: Scale-Normalized Conformal Calibration.
        Nonconformity score: S_r = |E_r| / sigma_r
        q_adaptive = q_norm * sigma_t
        Returns dictionary of adaptive quantiles {cov: q_adaptive}.
        """
        n = len(eligible_errors)
        if n == 0:
            return {cov: 0.0 for cov in nominal_coverages}

        current_sigma = self.estimate_residual_scale(eligible_errors)
        
        # Slicing the calibration window of size window_size
        window_errors = eligible_errors[-self.window_size:]
        
        # Vectorized rolling scale computation
        sigma_series = pd.Series(eligible_errors).rolling(self.scale_window, min_periods=1).mean().values
        norm_scores = window_errors / np.maximum(sigma_series[-len(window_errors):], self.epsilon)
        
        norm_quantiles = self.compute_conformal_quantiles(norm_scores, nominal_coverages)
        return {cov: norm_q * current_sigma for cov, norm_q in norm_quantiles.items()}

    def calibrate_volume_normalized(
        self,
        eligible_errors: np.ndarray,
        historical_volume: float,
        nominal_coverages: list
    ) -> Dict[float, float]:
        """
        Adaptive Method B: Volume-Normalized Conformal Calibration.
        Nonconformity score: S_r = |E_r| / max(volume, epsilon)
        q_adaptive = q_norm * volume_t
        Returns dictionary of adaptive quantiles {cov: q_adaptive}.
        """
        if len(eligible_errors) == 0:
            return {cov: 0.0 for cov in nominal_coverages}

        vol_scale = max(float(historical_volume), self.epsilon)
        window_errors = eligible_errors[-self.window_size:]
        norm_scores = window_errors / vol_scale
        
        norm_quantiles = self.compute_conformal_quantiles(norm_scores, nominal_coverages)
        return {cov: norm_q * vol_scale for cov, norm_q in norm_quantiles.items()}
