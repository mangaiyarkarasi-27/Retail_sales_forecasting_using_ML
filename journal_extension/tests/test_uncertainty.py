"""
test_uncertainty.py
-------------------
Unit tests for the uncertainty quantification and conformal prediction modules.
Verifies out-of-sample properties, rolling-window behavior, and leakage prevention.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "uncertainty")))

from prediction_intervals import PredictionInterval
from conformal_prediction import ConformalCalibrator
from uncertainty_metrics import (
    calculate_picp,
    calculate_miw,
    calculate_interval_score,
    calculate_coverage_error
)


class TestUncertainty(unittest.TestCase):
    
    def setUp(self):
        self.calibrator = ConformalCalibrator(window_size=10, min_observations=5)

    def test_prediction_interval_construction(self):
        pi = PredictionInterval(point_forecast=10.0, margin=3.0, clip_zero=True)
        self.assertEqual(pi.lower, 7.0)
        self.assertEqual(pi.upper, 13.0)
        self.assertEqual(pi.width, 6.0)
        self.assertTrue(pi.contains(10.0))
        self.assertFalse(pi.contains(6.0))
        
        pi_clipped = PredictionInterval(point_forecast=5.0, margin=8.0, clip_zero=True)
        self.assertEqual(pi_clipped.lower, 0.0)
        self.assertEqual(pi_clipped.upper, 13.0)

    def test_quantile_calculation(self):
        residuals = np.array([5.0, 2.0, 9.0, 1.0, 3.0, 7.0, 4.0, 6.0, 8.0])
        q = self.calibrator.compute_quantile(residuals, alpha=0.1)
        self.assertEqual(q, 9.0)

        # Insufficient observations handling
        with self.assertRaises(ValueError):
            self.calibrator.compute_quantile(residuals[:4], alpha=0.1)

    def test_rolling_residuals(self):
        residuals = np.arange(20, dtype=float)
        subset = self.calibrator.get_rolling_residuals(residuals, current_index=15)
        self.assertEqual(len(subset), 10)
        self.assertEqual(subset[0], 5.0)
        self.assertEqual(subset[-1], 14.0)

    def test_lead_time_aggregation(self):
        actuals = np.array([10.0, 20.0, 30.0, 40.0])
        preds = np.array([8.0, 19.0, 31.0, 38.0])
        errors = self.calibrator.calibrate_lead_time_demand(actuals, preds, lead_time=2)
        np.testing.assert_array_equal(errors, np.array([3.0, 0.0, 1.0]))

    def test_metrics(self):
        actuals = np.array([10.0, 20.0, 30.0])
        lowers = np.array([8.0, 18.0, 25.0])
        uppers = np.array([12.0, 22.0, 28.0])

        picp = calculate_picp(actuals, lowers, uppers)
        self.assertAlmostEqual(picp, 2.0 / 3.0)

        miw = calculate_miw(lowers, uppers)
        self.assertAlmostEqual(miw, 11.0 / 3.0)

        score = calculate_interval_score(actuals, lowers, uppers, alpha=0.2)
        self.assertAlmostEqual(score, 10.33333333333)

    def test_no_leakage_and_oos_verification(self):
        # 1. Verification: Prediction at t cannot use y[t] (current-target isolation)
        # We ensure that when sliding window extracts calibration residuals for index `t`,
        # it only slices indices up to `t-1`.
        residuals = np.arange(100, dtype=float)
        current_idx = 50
        calib_subset = self.calibrator.get_rolling_residuals(residuals, current_index=current_idx)
        
        # Ensure the current target index is NOT in the calibration residuals
        self.assertTrue(current_idx not in calib_subset)
        self.assertEqual(len(calib_subset), 10) # Window size is 10
        self.assertEqual(calib_subset[-1], 49.0) # Last index is t-1 (49)
        
        # 2. Verification: Calibration residuals contain only previous residuals (chronological ordering)
        self.assertTrue(all(val < current_idx for val in calib_subset))
        
        # 3. Verification: No future indices enter the calibration window
        self.assertTrue(all(val < 50 for val in calib_subset))

    def test_insufficient_calibration_handling(self):
        residuals = np.array([1.5, 2.3, 0.8])
        # Quantile estimation on 3 elements should fail if min_observations is 5
        with self.assertRaises(ValueError):
            self.calibrator.compute_quantile(residuals, alpha=0.1)


if __name__ == "__main__":
    unittest.main()
