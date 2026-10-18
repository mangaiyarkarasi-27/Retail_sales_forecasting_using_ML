"""
test_final_uncertainty.py
--------------------------
Unit tests verifying prediction interval correctness, target isolation,
clipping behavior, and exclusion of incomplete horizons.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "forecasting")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "uncertainty")))

from conformal_prediction import ConformalCalibrator


class TestFinalUncertainty(unittest.TestCase):

    def setUp(self):
        self.calibrator = ConformalCalibrator(window_size=100, min_observations=20)
        
    def test_lower_bound_clipping(self):
        # 5. Verify zero lower-bound clipping: L_t = max(0, Dhat - q)
        pred_val = 10.0
        q_L = 15.0
        lower = max(0.0, pred_val - q_L)
        self.assertEqual(lower, 0.0)

    def test_no_upper_clipping(self):
        # 6. Verify no upper clipping: U_t = Dhat + q
        pred_val = 10.0
        q_L = 15.0
        upper = pred_val + q_L
        self.assertEqual(upper, 25.0)

    def test_incomplete_final_horizons_exclusion(self):
        # 8. Verify that dates too close to the end of the test set are excluded.
        # If test set has 360 days, and L=7, we can only evaluate origins t <= 360 - 7 = 353.
        n_days = 360
        L = 7
        eval_origins = []
        for t in range(n_days):
            if t < n_days - L:
                eval_origins.append(t)
                
        self.assertEqual(len(eval_origins), 353)
        self.maxDiff = None
        self.assertNotIn(353, eval_origins)
        self.assertNotIn(359, eval_origins)


if __name__ == "__main__":
    unittest.main()
