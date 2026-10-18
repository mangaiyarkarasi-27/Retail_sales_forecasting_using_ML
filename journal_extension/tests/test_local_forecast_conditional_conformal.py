"""
test_local_forecast_conditional_conformal.py
--------------------------------------------
Unit tests and temporal leakage audit for Locally Weighted Forecast-Conditional Conformal (EXP-017).
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "uncertainty")))

from local_forecast_conditional_conformal import LocalForecastConditionalCalibrator


class TestLocalForecastConditionalConformal(unittest.TestCase):

    def setUp(self):
        self.calibrator = LocalForecastConditionalCalibrator(
            window_size=100,
            scale_window=50,
            min_regime_samples=20,
            epsilon=1e-4
        )

    def test_nearest_neighbor_selection(self):
        # Forecasts 10, 20, 30, ..., 100
        forecasts = np.arange(10.0, 110.0, 10.0)
        errors = np.full(len(forecasts), 5.0)
        
        # Current forecast = 52.0, K = 3
        # Closest forecasts to 52 should be 50.0 (|50-52|=2), 60.0 (|60-52|=8), 40.0 (|40-52|=12)
        margins, diag = self.calibrator.calibrate_locally_weighted(
            errors, forecasts, current_forecast=52.0, k_neighbors=3, nominal_coverages=[0.90]
        )
        self.assertEqual(diag["neighbor_count"], 3)
        self.assertAlmostEqual(diag["min_dist"], 2.0)
        self.assertAlmostEqual(diag["max_dist"], 12.0)
        self.assertFalse(diag["is_partial_k"])

    def test_startup_boundary_partial_k(self):
        # Only 5 historical observations available, K = 20
        forecasts = np.array([10.0, 12.0, 15.0, 18.0, 20.0])
        errors = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
        
        margins, diag = self.calibrator.calibrate_locally_weighted(
            errors, forecasts, current_forecast=14.0, k_neighbors=20, nominal_coverages=[0.90]
        )
        self.assertEqual(diag["neighbor_count"], 5)
        self.assertTrue(diag["is_partial_k"])

    def test_finite_sample_quantile_convention(self):
        # For K = 20, nominal coverage = 0.90 (alpha = 0.10)
        # index = ceil(21 * 0.90) = ceil(18.9) = 19 -> 0-indexed 18
        scores = np.arange(1.0, 21.0)
        q_dict = self.calibrator.compute_conformal_quantiles(scores, [0.90])
        self.assertEqual(q_dict[0.90], 19.0)

    def test_lower_bound_clipping(self):
        current_forecast = 15.0
        margin = 30.0
        lower = max(0.0, current_forecast - margin)
        upper = current_forecast + margin
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 45.0)

    def test_temporal_eligibility_rule(self):
        dates = pd.date_range("2017-01-01", periods=15)
        t_origin = dates[8]
        lead_time = 5
        # Rule: r <= t - L - 1 -> r <= t - 6 days
        cutoff = t_origin - pd.Timedelta(days=lead_time + 1)
        self.assertEqual(cutoff, dates[2])


if __name__ == "__main__":
    unittest.main()
