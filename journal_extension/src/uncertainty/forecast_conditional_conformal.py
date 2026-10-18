"""
forecast_conditional_conformal.py
---------------------------------
Implements Forecast-Conditional Conformal Prediction for multi-step lead-time uncertainty.
Conditions nonconformity scores on predicted demand magnitude using frozen development-period bins.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


class ForecastConditionalCalibrator:
    """
    Manages Forecast-Conditional, Scale-Normalized, and Raw Conformal Calibration.
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
        Assigns a forecast to Low, Medium, or High regime.
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
        if n < self.min_regime_samples:
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

    def calibrate_raw(
        self,
        eligible_errors: np.ndarray,
        nominal_coverages: List[float]
    ) -> Dict[float, float]:
        """
        Raw (unnormalized) conformal calibration: S = |E|
        Returns dictionary of raw quantiles {cov: q}.
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
        EXP-015 Scale-Normalized Conformal Calibration: S = |E| / sigma
        Returns dictionary of adaptive quantiles {cov: q_adaptive}.
        """
        n = len(eligible_errors)
        if n == 0:
            return {cov: 0.0 for cov in nominal_coverages}

        recent = eligible_errors[-self.scale_window:]
        current_sigma = max(float(np.mean(recent)), self.epsilon)
        
        window_errors = eligible_errors[-self.window_size:]
        sigmas = pd.Series(eligible_errors).rolling(self.scale_window, min_periods=1).mean().values
        norm_scores = window_errors / np.maximum(sigmas[-len(window_errors):], self.epsilon)
        
        norm_quantiles = self.compute_conformal_quantiles(norm_scores, nominal_coverages)
        return {cov: norm_q * current_sigma for cov, norm_q in norm_quantiles.items()}

    def calibrate_forecast_conditional(
        self,
        eligible_errors: np.ndarray,
        eligible_forecasts: np.ndarray,
        current_forecast: float,
        lead_time: int,
        nominal_coverages: List[float]
    ) -> Tuple[Dict[float, float], bool, str]:
        """
        EXP-016 Forecast-Conditional Conformal Calibration.
        S_r = |E_r| / max(F_r, epsilon)
        Returns (margins_dict, is_fallback_used, current_regime).
        """
        n = len(eligible_errors)
        if n == 0:
            return {cov: 0.0 for cov in nominal_coverages}, True, "Unknown"

        t_low, t_high = self.thresholds.get(lead_time, (20.0, 50.0))
        current_regime = self.get_regime(current_forecast, t_low, t_high)
        
        # Take the most recent W=100 eligible observations
        w_errors = eligible_errors[-self.window_size:]
        w_forecasts = eligible_forecasts[-self.window_size:]
        
        # Calculate nonconformity scores for the window
        w_scores = w_errors / np.maximum(w_forecasts, self.epsilon)
        
        # Filter to scores belonging to current forecast-magnitude regime
        regime_mask = np.array([self.get_regime(f, t_low, t_high) == current_regime for f in w_forecasts])
        regime_scores = w_scores[regime_mask]
        
        is_fallback = False
        if len(regime_scores) < self.min_regime_samples:
            # Deterministic Fallback: use all eligible scores in the W=100 window
            selected_scores = w_scores
            is_fallback = True
        else:
            selected_scores = regime_scores
            
        norm_quantiles = self.compute_conformal_quantiles(selected_scores, nominal_coverages)
        
        # Convert normalized quantile back to demand units: Margin = q * F_t
        f_scale = max(current_forecast, self.epsilon)
        margins = {cov: q_val * f_scale for cov, q_val in norm_quantiles.items()}
        
        return margins, is_fallback, current_regime
