"""
selection_features.py
---------------------
Helper module to extract and align demand-profile features
for the adaptive model router.
"""

import os
import pandas as pd
import numpy as np
from typing import List

class ProfileFeatureExtractor:
    """
    Manages loading and alignment of demand profile features.
    """
    def __init__(self, profiles_path: str = "journal_extension/results/model_selection/demand_profiles.csv"):
        self.profiles_path = os.path.normpath(profiles_path)
        self.profiles_df = self._load_profiles()
        
        # Router-safe feature list
        self.feature_cols = [
            "mean_sales",
            "median_sales",
            "std_sales",
            "cv_sales",
            "cv_sq_sales",
            "zero_demand_proportion",
            "avg_nonzero_demand",
            "avg_demand_interval",
            "lag1_autocorr",
            "lag7_autocorr",
            "trend_strength_r2",
            "trend_slope",
            "lag365_autocorr",
            "demand_change_ratio"
        ]

    def _load_profiles(self) -> pd.DataFrame:
        if not os.path.exists(self.profiles_path):
            raise FileNotFoundError(f"Profiles CSV not found at {self.profiles_path}")
        return pd.read_csv(self.profiles_path)

    def get_features_for_series(self, store: int, item: int) -> np.ndarray:
        """
        Retrieves the profile feature vector for a specific store-item pair.
        """
        row = self.profiles_df[
            (self.profiles_df["store"] == store) & (self.profiles_df["item"] == item)
        ]
        if row.empty:
            raise KeyError(f"No profile found for store {store}, item {item}")
        return row[self.feature_cols].values[0]

    def get_features_dataframe(self) -> pd.DataFrame:
        """
        Returns the safe feature columns merged with store and item identifiers.
        """
        return self.profiles_df[["store", "item"] + self.feature_cols].copy()
