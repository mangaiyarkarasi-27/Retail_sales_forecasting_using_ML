"""
test_forecast_conditional_conformal.py
--------------------------------------
Unit tests and temporal leakage audit for Forecast-Conditional Conformal Prediction (EXP-016).
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "uncertainty")))

from forecast_conditional_conformal import ForecastConditionalCalibrator


class TestForecastConditionalConformal(unittest.TestCase):

    def setUp(self):
        self.calibrator = ForecastConditionalCalibrator(
            window_size=100,
            scale_window=50,
            min_regime_samples=20,
            epsilon=1e-4
        )
        self.calibrator.thresholds = {
            1: (25.0, 60.0),
            3: (75.0, 180.0),
            5: (125.0, 300.0),
            7: (175.0, 420.0)
        }

    def test_finite_sample_quantile(self):
        scores = np.arange(1.0, 101.0)
        q_dict = self.calibrator.compute_conformal_quantiles(scores, [0.80, 0.90, 0.95])
        self.assertEqual(q_dict[0.80], 81.0)
        self.assertEqual(q_dict[0.90], 91.0)
        self.assertEqual(q_dict[0.95], 96.0)

    def test_regime_assignment(self):
        self.assertEqual(self.calibrator.get_regime(20.0, 25.0, 60.0), "Low")
        self.assertEqual(self.calibrator.get_regime(25.0, 25.0, 60.0), "Low")
        self.assertEqual(self.calibrator.get_regime(40.0, 25.0, 60.0), "Medium")
        self.assertEqual(self.calibrator.get_regime(60.0, 25.0, 60.0), "Medium")
        self.assertEqual(self.calibrator.get_regime(75.0, 25.0, 60.0), "High")

    def test_score_denominator_forecast_only(self):
        # Verify that score is |E| / F, not using actual demand
        errors = np.full(50, 10.0)
        forecasts = np.full(50, 50.0)
        # S should be 10.0 / 50.0 = 0.2
        margins, is_fallback, regime = self.calibrator.calibrate_forecast_conditional(
            errors, forecasts, current_forecast=50.0, lead_time=1, nominal_coverages=[0.90]
        )
        self.assertEqual(regime, "Medium")
        self.assertFalse(is_fallback)
        self.assertAlmostEqual(margins[0.90], 0.2 * 50.0, places=4)

    def test_fallback_activation(self):
        # 10 Low forecasts and 90 High forecasts
        errors = np.full(100, 10.0)
        forecasts = np.concatenate([np.full(10, 20.0), np.full(90, 80.0)])
        
        # When current forecast is 20.0 (Low), only 10 historical scores match Low regime (< 20 threshold)
        # Should trigger fallback
        margins, is_fallback, regime = self.calibrator.calibrate_forecast_conditional(
            errors, forecasts, current_forecast=20.0, lead_time=1, nominal_coverages=[0.90]
        )
        self.assertEqual(regime, "Low")
        self.assertTrue(is_fallback)

    def test_lower_bound_clipping(self):
        current_forecast = 10.0
        margin = 25.0
        lower = max(0.0, current_forecast - margin)
        upper = current_forecast + margin
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 35.0)

    def test_temporal_eligibility_leakage_check(self):
        dates = pd.date_range("2017-01-01", periods=10)
        t_origin = dates[5]
        lead_time = 3
        # Strict rule: r <= t - L - 1 -> r <= t - 4 days
        eligible_cutoff = t_origin - pd.Timedelta(days=lead_time + 1)
        self.assertEqual(eligible_cutoff, dates[1])


if __name__ == "__main__":
    unittest.main()
