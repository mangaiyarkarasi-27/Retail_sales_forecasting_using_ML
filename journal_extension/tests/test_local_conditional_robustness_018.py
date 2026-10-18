"""
test_local_conditional_robustness_018.py
-----------------------------------------
Unit tests for EXP-018: Robustness and Neighborhood-Size Audit of Locally Weighted
Forecast-Conditional Conformal Prediction.
"""

import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from journal_extension.src.uncertainty.local_forecast_conditional_conformal import (
    LocalForecastConditionalCalibrator
)


class TestLocalConditionalRobustness018(unittest.TestCase):

    def test_k_sweep_quantiles_and_margins(self):
        """
        Verifies that LocalForecastConditionalCalibrator produces valid margins
        across all 8 candidate K values: {10, 20, 30, 40, 50, 60, 75, 100}.
        """
        calibrator = LocalForecastConditionalCalibrator(window_size=100)
        np.random.seed(42)
        
        eligible_forecasts = np.random.uniform(20.0, 100.0, size=100)
        eligible_errors = np.random.normal(loc=0.0, scale=10.0, size=100)
        current_forecast = 50.0
        nominal_coverages = [0.80, 0.90, 0.95]
        k_candidates = [10, 20, 30, 40, 50, 60, 75, 100]

        for k in k_candidates:
            margins, diag = calibrator.calibrate_locally_weighted(
                eligible_errors=eligible_errors,
                eligible_forecasts=eligible_forecasts,
                current_forecast=current_forecast,
                k_neighbors=k,
                nominal_coverages=nominal_coverages
            )
            
            self.assertEqual(len(margins), 3)
            self.assertLessEqual(margins[0.80], margins[0.90])
            self.assertLessEqual(margins[0.90], margins[0.95])
            self.assertEqual(diag["neighbor_count"], k)
            self.assertFalse(diag["is_partial_k"])
            self.assertLessEqual(diag["min_dist"], diag["med_dist"])
            self.assertLessEqual(diag["med_dist"], diag["max_dist"])

    def test_k_100_unconditioned_benchmark(self):
        """
        Verifies that K=100 uses all 100 available window observations,
        serving as the exact unconditioned forecast-normalized benchmark.
        """
        calibrator = LocalForecastConditionalCalibrator(window_size=100)
        np.random.seed(123)
        
        eligible_forecasts = np.linspace(10.0, 110.0, 100)
        eligible_errors = np.abs(np.random.normal(0, 5, 100))
        current_forecast = 60.0
        
        margins_k100, diag_k100 = calibrator.calibrate_locally_weighted(
            eligible_errors=eligible_errors,
            eligible_forecasts=eligible_forecasts,
            current_forecast=current_forecast,
            k_neighbors=100,
            nominal_coverages=[0.90]
        )
        
        self.assertEqual(diag_k100["neighbor_count"], 100)
        self.assertFalse(diag_k100["is_partial_k"])
        self.assertAlmostEqual(diag_k100["max_dist"], 50.0, delta=0.5)

    def test_partial_k_startup_handling(self):
        """
        Verifies that when fewer than K eligible observations exist (e.g. early origins),
        all available observations are used and is_partial_k is set to True.
        """
        calibrator = LocalForecastConditionalCalibrator(window_size=100)
        eligible_forecasts = np.array([30.0, 40.0, 50.0])
        eligible_errors = np.array([2.0, 4.0, 1.0])
        current_forecast = 45.0
        
        margins, diag = calibrator.calibrate_locally_weighted(
            eligible_errors=eligible_errors,
            eligible_forecasts=eligible_forecasts,
            current_forecast=current_forecast,
            k_neighbors=50,
            nominal_coverages=[0.90]
        )
        
        self.assertEqual(diag["neighbor_count"], 3)
        self.assertTrue(diag["is_partial_k"])
        self.assertGreater(margins[0.90], 0.0)

    def test_distance_metric_ordering(self):
        """
        Verifies that historical observations are ordered strictly by absolute forecast distance.
        """
        calibrator = LocalForecastConditionalCalibrator(window_size=100)
        eligible_forecasts = np.array([10.0, 50.0, 80.0, 52.0, 48.0])
        eligible_errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        current_forecast = 50.0
        
        margins, diag = calibrator.calibrate_locally_weighted(
            eligible_errors=eligible_errors,
            eligible_forecasts=eligible_forecasts,
            current_forecast=current_forecast,
            k_neighbors=3,
            nominal_coverages=[0.90]
        )
        
        self.assertEqual(diag["neighbor_count"], 3)
        self.assertEqual(diag["min_dist"], 0.0)
        self.assertEqual(diag["max_dist"], 2.0)
        expected_mean = (50.0 + 52.0 + 48.0) / 3.0
        self.assertAlmostEqual(diag["mean_neighbor_f"], expected_mean, delta=0.01)

    def test_predefined_k_selection_rule(self):
        """
        Verifies the EXP-018 predefined K selection rule:
        Primary: lowest Winkler score.
        Tie-breaker: if relative diff <= 1%, choose larger K.
        """
        results_df = pd.DataFrame([
            {"k": 20, "winkler": 2100.0, "cov_error": 0.64},
            {"k": 50, "winkler": 2050.0, "cov_error": 0.63},
            {"k": 100, "winkler": 2080.0, "cov_error": 0.65},
        ])
        best_winkler = results_df["winkler"].min()
        close_mask = (results_df["winkler"] - best_winkler) / best_winkler <= 0.01
        candidates = results_df[close_mask]
        selected_k = int(candidates.sort_values(by="k", ascending=False).iloc[0]["k"])
        self.assertEqual(selected_k, 50)

        # Tie-breaker scenario: K=50 (winkler 2050) vs K=60 (winkler 2055, diff = 0.24%)
        results_df2 = pd.DataFrame([
            {"k": 50, "winkler": 2050.0, "cov_error": 0.63},
            {"k": 60, "winkler": 2055.0, "cov_error": 0.63},
        ])
        best_winkler2 = results_df2["winkler"].min()
        close_mask2 = (results_df2["winkler"] - best_winkler2) / best_winkler2 <= 0.01
        candidates2 = results_df2[close_mask2]
        selected_k2 = int(candidates2.sort_values(by="k", ascending=False).iloc[0]["k"])
        self.assertEqual(selected_k2, 60)


if __name__ == "__main__":
    unittest.main()
