"""
local_forecast_conditional_conformal.py
---------------------------------------
Implements Locally Weighted Forecast-Conditional Conformal Prediction for multi-step
lead-time demand uncertainty calibration.
Replaces hard forecast-magnitude bins with a smooth local-neighborhood calibration mechanism.
Optimized with pure NumPy vectorized operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


def fast_rolling_mean_50(arr: np.ndarray, k: int = 50) -> np.ndarray:
    """
    Blazing fast pure NumPy rolling mean with window k.
    """
    n = len(arr)
    if n == 0:
        return np.array([])
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    if n < k:
        return cumsum[1:] / np.arange(1, n + 1)
    init = cumsum[1:k] / np.arange(1, k)
    rest = (cumsum[k:] - cumsum[:-k]) / float(k)
    return np.concatenate([init, rest])


class LocalForecastConditionalCalibrator:
    """
    Manages Locally Weighted Forecast-Conditional, Hard-Bin Forecast-Conditional,
    Scale-Normalized, and Raw Conformal Calibration.
    """

    def __init__(
        self,
        window_size: int = 100,
        scale_window: int = 50,
        min_regime_samples: int = 20,
        epsilon: float = 1e-4
    ):
        self.window_size = window_size
        self.scale_window = scale_window
        self.min_regime_samples = min_regime_samples
        self.epsilon = epsilon
        self.thresholds: Dict[int, Tuple[float, float]] = {}

    def fit_thresholds_from_development(self, res_df: pd.DataFrame, lead_times: List[int]) -> Dict[int, Tuple[float, float]]:
        """
        Calculates empirical 33.33rd and 66.67th percentiles of predicted_lead_demand
        strictly from development-period residuals (2013-2016).
        """
        thresholds = {}
        for L in lead_times:
            df_L = res_df[res_df["lead_time"] == L]
            forecasts = df_L["predicted_lead_demand"].values
            t_low = float(np.percentile(forecasts, 100.0 / 3.0))
            t_high = float(np.percentile(forecasts, 200.0 / 3.0))
            thresholds[L] = (t_low, t_high)
        self.thresholds = thresholds
        return thresholds

    def get_regime(self, forecast: float, t_low: float, t_high: float) -> str:
        """
        Assigns a forecast to Low, Medium, or High regime based on frozen thresholds.
        """
        if forecast <= t_low:
            return "Low"
        elif forecast <= t_high:
            return "Medium"
        else:
            return "High"

    def compute_conformal_quantiles(self, scores: np.ndarray, nominal_coverages: List[float]) -> Dict[float, float]:
        """
        Standard finite-sample conformal quantile: index = ceil((n + 1) * (1 - alpha))
        """
        n = len(scores)
        quantiles = {}
        if n == 0:
            return {cov: 0.0 for cov in nominal_coverages}
        if n < self.min_regime_samples:
            for cov in nominal_coverages:
                quantiles[cov] = float(np.quantile(scores, cov))
            return quantiles

        sorted_scores = np.sort(scores)
        for cov in nominal_coverages:
            alpha = 1.0 - cov
            idx_1indexed = int(np.ceil((n + 1) * (1.0 - alpha)))
            idx_0indexed = min(max(idx_1indexed - 1, 0), n - 1)
            quantiles[cov] = float(sorted_scores[idx_0indexed])
        return quantiles

    def calibrate_raw(
        self,
        eligible_errors: np.ndarray,
        nominal_coverages: List[float]
    ) -> Dict[float, float]:
        """
        Method A: Raw Conformal (S = |E|)
        """
        if len(eligible_errors) == 0:
            return {cov: 0.0 for cov in nominal_coverages}
        window = eligible_errors[-self.window_size:]
        return self.compute_conformal_quantiles(window, nominal_coverages)

    def calibrate_scale_normalized(
        self,
        eligible_errors: np.ndarray,
        nominal_coverages: List[float]
    ) -> Dict[float, float]:
        """
        Method B: Scale-Normalized Conformal (S = |E| / sigma)
        """
        n = len(eligible_errors)
        if n == 0:
            return {cov: 0.0 for cov in nominal_coverages}

        recent = eligible_errors[-self.scale_window:]
        current_sigma = max(float(np.mean(recent)), self.epsilon)
        
        window_errors = eligible_errors[-self.window_size:]
        sigmas = fast_rolling_mean_50(eligible_errors, self.scale_window)
        norm_scores = window_errors / np.maximum(sigmas[-len(window_errors):], self.epsilon)
        
        norm_quantiles = self.compute_conformal_quantiles(norm_scores, nominal_coverages)
        return {cov: norm_q * current_sigma for cov, norm_q in norm_quantiles.items()}

    def calibrate_hard_conditional(
        self,
        eligible_errors: np.ndarray,
        eligible_forecasts: np.ndarray,
        current_forecast: float,
        lead_time: int,
        nominal_coverages: List[float]
    ) -> Tuple[Dict[float, float], bool, str]:
        """
        Method C: EXP-016 Hard-Bin Forecast-Conditional Conformal.
        """
        n = len(eligible_errors)
        if n == 0:
            return {cov: 0.0 for cov in nominal_coverages}, True, "Unknown"

        t_low, t_high = self.thresholds.get(lead_time, (37.93, 56.82))
        current_regime = self.get_regime(current_forecast, t_low, t_high)
        
        w_errors = eligible_errors[-self.window_size:]
        w_forecasts = eligible_forecasts[-self.window_size:]
        w_scores = w_errors / np.maximum(w_forecasts, self.epsilon)
        
        regime_mask = np.array([self.get_regime(f, t_low, t_high) == current_regime for f in w_forecasts])
        regime_scores = w_scores[regime_mask]
        
        is_fallback = False
        if len(regime_scores) < self.min_regime_samples:
            selected_scores = w_scores
            is_fallback = True
        else:
            selected_scores = regime_scores
            
        norm_quantiles = self.compute_conformal_quantiles(selected_scores, nominal_coverages)
        f_scale = max(current_forecast, self.epsilon)
        margins = {cov: q_val * f_scale for cov, q_val in norm_quantiles.items()}
        
        return margins, is_fallback, current_regime

    def calibrate_locally_weighted(
        self,
        eligible_errors: np.ndarray,
        eligible_forecasts: np.ndarray,
        current_forecast: float,
        k_neighbors: int,
        nominal_coverages: List[float]
    ) -> Tuple[Dict[float, float], Dict[str, float]]:
        """
        Method D: EXP-017 Locally Weighted Forecast-Conditional Conformal.
        Selects the K nearest historical forecast observations within window W=100.
        Returns (margins_dict, diagnostics_dict).
        """
        n = len(eligible_errors)
        if n == 0:
            margins = {cov: 0.0 for cov in nominal_coverages}
            diag = {
                "min_dist": 0.0, "med_dist": 0.0, "max_dist": 0.0,
                "mean_neighbor_f": 0.0, "std_neighbor_f": 0.0,
                "neighbor_count": 0, "is_partial_k": True
            }
            return margins, diag

        w_errors = eligible_errors[-self.window_size:]
        w_forecasts = eligible_forecasts[-self.window_size:]
        w_scores = w_errors / np.maximum(w_forecasts, self.epsilon)
        
        # Calculate absolute forecast distance
        distances = np.abs(w_forecasts - current_forecast)
        
        # Sort by distance and select K nearest
        actual_k = min(k_neighbors, len(w_forecasts))
        nearest_indices = np.argsort(distances)[:actual_k]
        
        selected_scores = w_scores[nearest_indices]
        selected_distances = distances[nearest_indices]
        selected_forecasts = w_forecasts[nearest_indices]
        
        norm_quantiles = self.compute_conformal_quantiles(selected_scores, nominal_coverages)
        f_scale = max(current_forecast, self.epsilon)
        margins = {cov: q_val * f_scale for cov, q_val in norm_quantiles.items()}
        
        diag = {
            "min_dist": float(np.min(selected_distances)),
            "med_dist": float(np.median(selected_distances)),
            "max_dist": float(np.max(selected_distances)),
            "mean_neighbor_f": float(np.mean(selected_forecasts)),
            "std_neighbor_f": float(np.std(selected_forecasts)),
            "neighbor_count": int(actual_k),
            "is_partial_k": bool(actual_k < k_neighbors)
        }
        
        return margins, diag
