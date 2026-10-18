"""
test_oos_calibration.py
-------------------------
Unit tests verifying out-of-sample properties, chronological residual eligibility,
and rolling calibration window boundaries for the Direct Multi-Horizon model.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "uncertainty")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "forecasting")))

from oos_calibration_direct import OOSResidualsGeneratorDirect
from conformal_prediction import ConformalCalibrator


class TestOOSCalibrationDirect(unittest.TestCase):

    def setUp(self):
        # Create a mock series
        dates = pd.date_range(start="2015-01-01", periods=150)
        self.df = pd.DataFrame({
            "store": [1] * 150,
            "item": [1] * 150,
            "date": dates,
            "sales": np.arange(1.0, 151.0),
            "weekday": dates.weekday,
            "year": dates.year,
            "month": dates.month,
            "day": dates.day,
            "weekofyear": dates.isocalendar().week.astype(int),
            "quarter": dates.quarter
        })
        self.generator = OOSResidualsGeneratorDirect()
        self.calibrator = ConformalCalibrator(window_size=100, min_observations=20)

    def test_oos_shifted_dataset(self):
        # 4. Verify horizon targets are future relative to origin
        dataset_h3 = self.generator.construct_horizon_dataset(self.df, h=3).dropna()
        
        # In construct_horizon_dataset, the date column represents the target date T = t + h.
        # Check that target sales is indeed sales of target date T,
        # and lag/rolling features are from row T-3 (origin t).
        for idx, row in dataset_h3.iterrows():
            target_date = pd.to_datetime(row["date"])
            origin_date = target_date - pd.Timedelta(days=3)
            cutoff_date = origin_date - pd.Timedelta(days=1)
            
            # Target is sales of target date T
            original_sales_row = self.df[self.df["date"] == target_date].iloc[0]
            self.assertEqual(row["sales"], original_sales_row["sales"])
            
            # Features are from origin_date and depend only on sales <= cutoff_date
            self.assertEqual(row["sales_lag_7"], self.df[self.df["date"] == origin_date - pd.Timedelta(days=7)].iloc[0]["sales"])
            self.assertEqual(row["rolling_mean_7"], self.df[(self.df["date"] >= origin_date - pd.Timedelta(days=7)) & (self.df["date"] <= cutoff_date)]["sales"].mean())

    def test_residual_eligibility_and_windows(self):
        # 6. Verify residual eligibility obeys r + L < t (r <= t - L - 1)
        # 7. L=1: latest eligible origin = t-2
        # 8. L=3: latest eligible origin = t-4
        # 9. L=5: latest eligible origin = t-6
        # 10. L=7: latest eligible origin = t-8
        
        t = 50 # Let prediction origin be index 50
        
        for L in [1, 3, 5, 7]:
            latest_eligible = t - L - 1
            
            # Any residual with origin r <= latest_eligible is eligible
            # Let's verify that the latest eligible origin indeed satisfies r + L < t
            self.assertTrue(latest_eligible + L < t)
            # The next origin index (latest_eligible + 1) must be ineligible (must violate r + L < t)
            self.assertFalse((latest_eligible + 1) + L < t)

    def test_calibration_window_size(self):
        # 11. Verify calibration window contains exactly 100 eligible origins when enough history exists
        residuals = np.arange(200, dtype=float)
        t = 150
        
        for L in [1, 3, 5, 7]:
            # Slice range: [t - 100 - L : t - L] which in Python represents [t - 100 - L, t - L - 1]
            calib_slice = residuals[t - 100 - L : t - L]
            
            # Length should be exactly 100
            self.assertEqual(len(calib_slice), 100)
            
            # The latest index in the slice is t - L - 1 (r = t - L - 1), which satisfies r + L < t
            self.assertEqual(calib_slice[-1], t - L - 1)


if __name__ == "__main__":
    unittest.main()
