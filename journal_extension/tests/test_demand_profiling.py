"""
Unit tests for demand profiling and regime detection modules.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "demand_profiling")))

from demand_profile import DemandProfiler
from regime_detection import DemandRegimeDetector


class TestDemandProfiling(unittest.TestCase):

    def setUp(self):
        # Create a simple synthetic dataframe for testing
        # 2 store-item series, 10 days
        dates = pd.date_range(start="2020-01-01", periods=10)
        
        # Series 1: store=1, item=1 (smooth positive sales)
        s1 = pd.DataFrame({
            "date": dates,
            "store": 1,
            "item": 1,
            "sales": [10, 12, 11, 13, 10, 12, 11, 13, 10, 12]
        })
        
        # Series 2: store=1, item=2 (sparse/zero sales)
        s2 = pd.DataFrame({
            "date": dates,
            "store": 1,
            "item": 2,
            "sales": [0, 5, 0, 5, 0, 5, 0, 5, 0, 5]
        })
        
        self.df = pd.concat([s1, s2], ignore_index=True)
        self.profiler = DemandProfiler()
        self.detector = DemandRegimeDetector()
        
        # Update config thresholds temporarily for testing
        self.profiler.config["min_observations"] = 5
        self.detector.config["min_observations"] = 5

    def test_no_mutation(self):
        """Verify the input DataFrame is not mutated."""
        df_copy = self.df.copy()
        _ = self.profiler.build_profiles(self.df)
        pd.testing.assert_frame_equal(self.df, df_copy)

    def test_grouping_and_ordering(self):
        """Verify output has correct groups and is deterministic."""
        res = self.profiler.build_profiles(self.df)
        self.assertEqual(len(res), 2)
        self.assertListEqual(list(res["store"]), [1, 1])
        self.assertListEqual(list(res["item"]), [1, 2])

    def test_metrics_calculation(self):
        """Verify the metrics calculated are mathematically correct."""
        profiles = self.profiler.build_profiles(self.df)
        
        # Store 1, Item 1
        p1 = profiles[profiles["item"] == 1].iloc[0]
        self.assertEqual(p1["mean_sales"], 11.4)
        self.assertEqual(p1["zero_demand_proportion"], 0.0)
        self.assertEqual(p1["avg_demand_interval"], 1.0)
        
        # Store 1, Item 2
        p2 = profiles[profiles["item"] == 2].iloc[0]
        self.assertEqual(p2["mean_sales"], 2.5)
        self.assertEqual(p2["zero_demand_proportion"], 0.5)
        self.assertEqual(p2["avg_demand_interval"], 2.0)

    def test_insufficient_observations(self):
        """Verify handling of series with too few observations."""
        self.profiler.config["min_observations"] = 100
        profiles = self.profiler.build_profiles(self.df)
        p1 = profiles[profiles["item"] == 1].iloc[0]
        self.assertEqual(p1["sbc_category"], "Insufficient Data")

    def test_regime_classification(self):
        """Verify regime classification maps correctly based on thresholds."""
        # SBC thresholds: ADI=1.32, CV_sq=0.49
        profiles = self.profiler.build_profiles(self.df)
        regimes = self.detector.detect_regimes(profiles)
        
        # Store 1, Item 1 (smooth: ADI=1.0 <= 1.32, CV_sq ~ 0.01 <= 0.49)
        r1 = regimes[regimes["item"] == 1].iloc[0]
        self.assertEqual(r1["sbc_category"], "Smooth")
        
        # Store 1, Item 2 (Lumpy: ADI=2.0 > 1.32, CV_sq=1.0 > 0.49)
        r2 = regimes[regimes["item"] == 2].iloc[0]
        self.assertEqual(r2["sbc_category"], "Lumpy")


if __name__ == "__main__":
    unittest.main()
