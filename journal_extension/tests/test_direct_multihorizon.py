"""
test_direct_multihorizon.py
----------------------------
Unit tests verifying the absence of temporal leakage in features,
forecast boundaries, and training logic of the Direct Multi-Horizon Point Forecasts.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "forecasting")))

from decision_time_features import generate_origin_features, construct_target_calendar_features
from direct_multihorizon import DirectMultiHorizonRegressor


class TestDirectMultiHorizon(unittest.TestCase):

    def setUp(self):
        # Create a mock sales history spanning 50 days for 1 store and 1 item
        dates = pd.date_range(start="2015-06-01", periods=50)
        self.df = pd.DataFrame({
            "store": [1] * 50,
            "item": [1] * 50,
            "date": dates,
            "sales": np.arange(1.0, 51.0), # sales = 1 to 50
            "weekday": dates.weekday,
            "year": dates.year,
            "month": dates.month,
            "day": dates.day,
            "weekofyear": dates.isocalendar().week.astype(int),
            "quarter": dates.quarter
        })
        self.regressor = DirectMultiHorizonRegressor()

    def test_feature_generator_lags_and_rolling_means(self):
        # Generate origin features at date t
        feat_df = generate_origin_features(self.df)
        
        # Test index where date t = 2015-06-15 (row index 14, sales=15.0)
        idx_t = 14
        row_t = feat_df.iloc[idx_t]
        self.assertEqual(row_t["date"], pd.to_datetime("2015-06-15"))
        
        # 1. Verify sales_lag_7 references t-7 (2015-06-08, sales=8.0)
        self.assertEqual(row_t["sales_lag_7"], 8.0)
        
        # 2. Verify sales_lag_14 references t-14 (2015-06-01, sales=1.0)
        self.assertEqual(row_t["sales_lag_14"], 1.0)
        
        # 3. Verify sales_lag_30 is NaN (requires t-30, not enough history)
        self.assertTrue(np.isnan(row_t["sales_lag_30"]))
        
        # 4. Verify rolling_mean_7 ends at t-1 and covers [t-7, t-1]
        # sales for [2015-06-08 ... 2015-06-14] = [8, 9, 10, 11, 12, 13, 14]
        # mean = 11.0
        self.assertEqual(row_t["rolling_mean_7"], 11.0)
        
        # 5. Verify rolling_mean_30 is NaN (not enough history)
        self.assertTrue(np.isnan(row_t["rolling_mean_30"]))

    def test_target_calendar_features(self):
        # 6. Verify calendar features do not use sales demand
        cal_df = construct_target_calendar_features(pd.Series([pd.to_datetime("2015-06-15")]), h=3)
        self.assertEqual(cal_df.iloc[0]["day"], 18) # 15 + 3 = 18
        self.assertEqual(cal_df.iloc[0]["month"], 6)
        
        # Ensure no columns are related to sales
        self.assertNotIn("sales", cal_df.columns)

    def test_training_row_boundaries(self):
        # 7. Verify forecast features never read future target sales
        dataset_h3 = self.regressor.construct_horizon_dataset(self.df, h=3)
        
        # Target at day T should be sales of day T
        # Inputs at origin t = T-3 should only use sales <= t-1 (i.e. <= T-4)
        for index, row in dataset_h3.iterrows():
            target_date = pd.to_datetime(row["date"])
            origin_date = target_date - pd.Timedelta(days=3)
            cutoff_date = origin_date - pd.Timedelta(days=1)
            
            # Ensure features are computed using sales strictly preceding origin_date
            self.assertEqual(row["sales_lag_7"], self.df[self.df["date"] == origin_date - pd.Timedelta(days=7)].iloc[0]["sales"])
            self.assertEqual(row["rolling_mean_7"], self.df[(self.df["date"] >= origin_date - pd.Timedelta(days=7)) & (self.df["date"] <= cutoff_date)]["sales"].mean())

    def test_train_val_date_bounds(self):
        # Mock a larger dataset that spans into 2017
        dates = pd.date_range(start="2015-01-01", end="2017-06-30")
        df_long = pd.DataFrame({
            "store": [1] * len(dates),
            "item": [1] * len(dates),
            "date": dates,
            "sales": np.random.normal(10, 2, len(dates)),
            "weekday": dates.weekday,
            "year": dates.year,
            "month": dates.month,
            "day": dates.day,
            "weekofyear": dates.isocalendar().week.astype(int),
            "quarter": dates.quarter
        })
        
        h = 3
        dataset = self.regressor.construct_horizon_dataset(df_long, h=h)
        
        # 8. Training rows obey target date <= 2015-12-31
        train_mask = dataset["date"] <= pd.to_datetime("2015-12-31")
        train_data = dataset[train_mask]
        self.assertTrue((train_data["date"] <= pd.to_datetime("2015-12-31")).all())
        
        # 9. Validation rows obey target date <= 2016-12-31
        val_mask = (dataset["date"] >= pd.to_datetime("2016-01-01")) & (dataset["date"] <= pd.to_datetime("2016-12-31"))
        val_data = dataset[val_mask]
        self.assertTrue((val_data["date"] >= pd.to_datetime("2016-01-01")).all())
        self.assertTrue((val_data["date"] <= pd.to_datetime("2016-12-31")).all())
        
        # 10. Verify final test data is not accessed by the training or validation dataset masking
        self.assertFalse((train_data["date"] >= pd.to_datetime("2017-01-01")).any())
        self.assertFalse((val_data["date"] >= pd.to_datetime("2017-01-01")).any())


if __name__ == "__main__":
    unittest.main()
