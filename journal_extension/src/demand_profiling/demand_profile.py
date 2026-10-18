"""
demand_profile.py
-----------------
This module implements the Demand Profiling logic for the journal extension.
It profiles each store-item demand time series using metrics computed strictly 
from the training partition, preventing data leakage.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DemandProfiler:
    """
    Constructs demand profiles for each store-item series using historical training data.
    """

    def __init__(self, config_path: str = "journal_extension/configs/journal_config.yaml"):
        self.config_path = os.path.normpath(config_path)
        self.config = self._load_config()
        self.features_df = None

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f).get("demand_profiling", {})
        return {
            "min_observations": 100,
            "adi_threshold": 1.32,
            "cv_sq_threshold": 0.49,
            "trend_threshold": 0.20,
            "seasonality_threshold": 0.30
        }

    def compute_series_profile(self, series: pd.Series, dates: pd.Series) -> Dict[str, Any]:
        """
        Computes demand characteristics for a single time series.
        """
        n_obs = len(series)
        if n_obs < self.config.get("min_observations", 100):
            return {
                "mean_sales": series.mean() if n_obs > 0 else 0.0,
                "median_sales": np.nan,
                "std_sales": np.nan,
                "cv_sales": np.nan,
                "cv_sq_sales": np.nan,
                "zero_demand_proportion": np.nan,
                "avg_nonzero_demand": np.nan,
                "avg_demand_interval": np.nan,
                "lag1_autocorr": np.nan,
                "lag7_autocorr": np.nan,
                "trend_strength_r2": np.nan,
                "trend_slope": np.nan,
                "lag365_autocorr": np.nan,
                "demand_change_ratio": np.nan,
                "sbc_category": "Insufficient Data"
            }

        sales_vals = series.values
        non_zero = sales_vals[sales_vals > 0]
        n_nonzero = len(non_zero)
        n_zero = n_obs - n_nonzero

        # 1. Basic Stats
        mean_sales = float(np.mean(sales_vals))
        median_sales = float(np.median(sales_vals))
        std_sales = float(np.std(sales_vals))
        cv_sales = std_sales / mean_sales if mean_sales > 0 else 0.0
        cv_sq_sales = cv_sales ** 2

        # 2. Intermittency/Sparsity
        zero_demand_prop = n_zero / n_obs
        avg_nonzero_demand = float(np.mean(non_zero)) if n_nonzero > 0 else 0.0

        # Calculate Average Demand Interval (ADI)
        # Time indices where sales > 0
        nonzero_indices = np.where(sales_vals > 0)[0]
        if len(nonzero_indices) > 1:
            intervals = np.diff(nonzero_indices)
            avg_demand_interval = float(np.mean(intervals))
        else:
            avg_demand_interval = 1.0

        # 3. Autocorrelation
        # Lag-1 Autocorrelation
        lag1_autocorr = 0.0
        if std_sales > 0:
            lag1_autocorr = float(pd.Series(sales_vals).autocorr(lag=1))
            if np.isnan(lag1_autocorr):
                lag1_autocorr = 0.0

        # Lag-7 Autocorrelation (Weekly cycles)
        lag7_autocorr = 0.0
        if std_sales > 0 and n_obs > 7:
            lag7_autocorr = float(pd.Series(sales_vals).autocorr(lag=7))
            if np.isnan(lag7_autocorr):
                lag7_autocorr = 0.0

        # 4. Trend Strength
        # We calculate trend strength using R2 of sales regressed on time index
        t = np.arange(n_obs)
        if len(t) > 1 and std_sales > 0:
            slope, intercept = np.polyfit(t, sales_vals, 1)
            pred = slope * t + intercept
            ss_tot = np.sum((sales_vals - mean_sales) ** 2)
            ss_res = np.sum((sales_vals - pred) ** 2)
            trend_strength = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        else:
            trend_strength = 0.0
            slope = 0.0

        # 5. Seasonality Strength (Lag-365 Autocorrelation, annual seasonality)
        lag365_autocorr = 0.0
        if std_sales > 0 and n_obs > 365:
            lag365_autocorr = float(pd.Series(sales_vals).autocorr(lag=365))
            if np.isnan(lag365_autocorr):
                lag365_autocorr = 0.0

        # 6. Recent vs Historical Demand Change
        # Mean of last 30 days compared to previous historical mean
        recent_mean = float(np.mean(sales_vals[-30:])) if n_obs >= 30 else mean_sales
        historical_mean = float(np.mean(sales_vals[:-30])) if n_obs > 30 else mean_sales
        demand_change_ratio = recent_mean / (historical_mean + 1e-5)

        # SBC Categorization (Syntetos-Boylan Classification)
        adi_thresh = self.config.get("adi_threshold", 1.32)
        cv_sq_thresh = self.config.get("cv_sq_threshold", 0.49)

        if avg_demand_interval <= adi_thresh:
            if cv_sq_sales <= cv_sq_thresh:
                sbc_category = "Smooth"
            else:
                sbc_category = "Erratic"
        else:
            if cv_sq_sales <= cv_sq_thresh:
                sbc_category = "Intermittent"
            else:
                sbc_category = "Lumpy"

        return {
            "mean_sales": round(mean_sales, 4),
            "median_sales": round(median_sales, 4),
            "std_sales": round(std_sales, 4),
            "cv_sales": round(cv_sales, 4),
            "cv_sq_sales": round(cv_sq_sales, 4),
            "zero_demand_proportion": round(zero_demand_prop, 6),
            "avg_nonzero_demand": round(avg_nonzero_demand, 4),
            "avg_demand_interval": round(avg_demand_interval, 4),
            "lag1_autocorr": round(lag1_autocorr, 4),
            "lag7_autocorr": round(lag7_autocorr, 4),
            "trend_strength_r2": round(trend_strength, 4),
            "trend_slope": round(slope, 6),
            "lag365_autocorr": round(lag365_autocorr, 4),
            "demand_change_ratio": round(demand_change_ratio, 4),
            "sbc_category": sbc_category
        }

    def build_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds profiles for all store-item combinations present in the DataFrame.
        """
        logger.info("Computing demand profiles for all store-item groups...")
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
        df_copy = df_copy.sort_values(by="date").reset_index(drop=True)

        profiles = []
        groups = df_copy.groupby(["store", "item"])

        for (store, item), group in groups:
            profile = self.compute_series_profile(group["sales"], group["date"])
            profile["store"] = int(store)
            profile["item"] = int(item)
            profiles.append(profile)

        self.features_df = pd.DataFrame(profiles)
        
        # Sort features columns to place store and item at front
        cols = ["store", "item", "sbc_category", "mean_sales", "median_sales", "std_sales", 
                "cv_sales", "cv_sq_sales", "zero_demand_proportion", "avg_nonzero_demand", 
                "avg_demand_interval", "lag1_autocorr", "lag7_autocorr", "trend_strength_r2", 
                "trend_slope", "lag365_autocorr", "demand_change_ratio"]
        self.features_df = self.features_df[cols]

        return self.features_df
