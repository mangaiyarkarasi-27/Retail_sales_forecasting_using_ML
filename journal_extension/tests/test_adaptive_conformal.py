"""
test_adaptive_conformal.py
--------------------------
Unit tests for Scale-Normalized and Volume-Normalized Conformal Calibration.
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "uncertainty")))

from adaptive_conformal import AdaptiveConformalCalibrator


class TestAdaptiveConformal(unittest.TestCase):

    def setUp(self):
        self.calibrator = AdaptiveConformalCalibrator(
            window_size=100,
            scale_window=50,
            min_observations=20,
            epsilon=1e-4
        )

    def test_finite_sample_quantile(self):
        # 100 observations: 1 to 100
        scores = np.arange(1.0, 101.0)
        # For alpha = 0.10 (90% coverage), index = ceil(101 * 0.9) = 91 -> 0-indexed 90 -> score 91.0
        q_90 = self.calibrator.compute_conformal_quantile(scores, alpha=0.10)
        self.assertEqual(q_90, 91.0)

        # For alpha = 0.05 (95% coverage), index = ceil(101 * 0.95) = 96 -> 0-indexed 95 -> score 96.0
        q_95 = self.calibrator.compute_conformal_quantile(scores, alpha=0.05)
        self.assertEqual(q_95, 96.0)

        # For alpha = 0.20 (80% coverage), index = ceil(101 * 0.80) = 81 -> 0-indexed 80 -> score 81.0
        q_80 = self.calibrator.compute_conformal_quantile(scores, alpha=0.20)
        self.assertEqual(q_80, 81.0)

    def test_scale_estimation(self):
        # Constant error of 10.0
        errors = np.full(100, 10.0)
        sigma = self.calibrator.estimate_residual_scale(errors)
        self.assertEqual(sigma, 10.0)

        # Varying errors with last 50 having mean 20.0
        errors_varied = np.concatenate([np.full(50, 5.0), np.full(50, 20.0)])
        sigma_recent = self.calibrator.estimate_residual_scale(errors_varied)
        self.assertEqual(sigma_recent, 20.0)

    def test_scale_normalized_calibration(self):
        # Scaled errors
        errors = np.linspace(10.0, 30.0, 120)
        q_adaptive, q_norm, sigma = self.calibrator.calibrate_scale_normalized(errors, nominal_coverage=0.90)
        self.assertGreater(q_adaptive, 0.0)
        self.assertGreater(q_norm, 0.0)
        self.assertGreater(sigma, 0.0)
        # Assert q_adaptive is proportional to current scale
        self.assertAlmostEqual(q_adaptive, q_norm * sigma, places=4)

    def test_volume_normalized_calibration(self):
        errors = np.linspace(5.0, 25.0, 100)
        volume = 50.0
        q_adaptive, q_norm, vol_scale = self.calibrator.calibrate_volume_normalized(errors, volume, nominal_coverage=0.90)
        self.assertEqual(vol_scale, 50.0)
        self.assertAlmostEqual(q_adaptive, q_norm * volume, places=4)

    def test_lower_bound_clipping(self):
        pred_val = 15.0
        q_adaptive = 25.0
        lower = max(0.0, pred_val - q_adaptive)
        upper = pred_val + q_adaptive
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 40.0)


if __name__ == "__main__":
    unittest.main()
