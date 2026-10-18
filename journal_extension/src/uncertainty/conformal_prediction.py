"""
conformal_prediction.py
-----------------------
Implements Rolling-Origin Conformal Calibration for retail sales forecasting,
supporting multi-step cumulative lead-time demand uncertainty calibration.
"""

import numpy as np
from typing import List, Tuple, Optional

class ConformalCalibrator:
    """
    Manages rolling conformal prediction intervals for lead-time demand.
    """

    def __init__(self, window_size: int = 100, min_observations: int = 20):
        self.window_size = window_size
        self.min_observations = min_observations

    def compute_quantile(self, absolute_residuals: np.ndarray, alpha: float) -> float:
        """
        Computes the conformal quantile using the standard finite-sample correction.
        Index = ceil((n + 1) * (1 - alpha))
        """
        n = len(absolute_residuals)
        if n < self.min_observations:
            raise ValueError(f"Insufficient calibration residuals: {n} (minimum required is {self.min_observations})")
        
        sorted_residuals = np.sort(absolute_residuals)
        
        # Finite-sample index calculation (1-indexed index in sorted array)
        idx_1indexed = int(np.ceil((n + 1) * (1 - alpha)))
        
        # Clip index to ensure it falls within the array bounds (0-indexed)
        idx_0indexed = min(idx_1indexed - 1, n - 1)
        idx_0indexed = max(idx_0indexed, 0)
        
        return float(sorted_residuals[idx_0indexed])

    def get_rolling_residuals(self, 
                              historical_residuals: np.ndarray, 
                              current_index: int) -> np.ndarray:
        """
        Extracts the sliding window of residuals prior to the current prediction origin.
        """
        start = max(0, current_index - self.window_size)
        return historical_residuals[start:current_index]

    def calibrate_lead_time_demand(self,
                                  historical_actuals: np.ndarray,
                                  historical_preds: np.ndarray,
                                  lead_time: int) -> np.ndarray:
        """
        Computes absolute lead-time prediction residuals over the historical set.
        For lead-time L, the cumulative error is:
        E_t = | sum_{k=1}^L actual_{t+k} - sum_{k=1}^L pred_{t+k} |
        """
        n = len(historical_actuals)
        if n < lead_time:
            return np.array([])
            
        lead_time_errors = []
        for t in range(n - lead_time + 1):
            act_sum = np.sum(historical_actuals[t : t + lead_time])
            pred_sum = np.sum(historical_preds[t : t + lead_time])
            lead_time_errors.append(abs(act_sum - pred_sum))
            
        return np.array(lead_time_errors)
