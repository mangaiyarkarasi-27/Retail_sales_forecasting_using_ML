"""
regime_detection.py
-------------------
This module classifies store-item sales series into multi-attribute demand regimes
based on metrics computed strictly from the training dataset.
"""

import os
import logging
import yaml
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DemandRegimeDetector:
    """
    Categorizes series using multi-attribute demand profiles without forcing 
    non-intermittent series into arbitrary classes.
    """

    def __init__(self, config_path: str = "journal_extension/configs/journal_config.yaml"):
        self.config_path = os.path.normpath(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f).get("demand_profiling", {})
        return {
            "adi_threshold": 1.32,
            "cv_sq_threshold": 0.49,
            "trend_threshold": 0.20,
            "seasonality_threshold": 0.30
        }

    def detect_regimes(self, profiles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes numerical profile features and maps each store-item combination 
        to an array of active demand descriptors and a primary routing label.
        """
        logger.info("Detecting demand regimes from series profiles...")
        df = profiles_df.copy()

        trend_thresh = self.config.get("trend_threshold", 0.20)
        sea_thresh = self.config.get("seasonality_threshold", 0.30)
        cv_sq_thresh = self.config.get("cv_sq_threshold", 0.49)

        trend_labels = []
        season_labels = []
        volatility_labels = []
        intermittent_labels = []
        combined_attributes = []
        primary_routing_label = []

        for _, row in df.iterrows():
            attrs = []
            
            # 1. Trend Attribution
            if row["trend_strength_r2"] >= trend_thresh:
                t_label = "Strong Trend"
                attrs.append("trending")
            else:
                t_label = "Stable Trend"
                
            # 2. Seasonality Attribution
            if row["lag365_autocorr"] >= sea_thresh:
                s_label = "Strong Seasonality"
                attrs.append("seasonal")
            else:
                s_label = "Weak Seasonality"
                
            # 3. Volatility/SBC Attribution
            if row["cv_sq_sales"] > cv_sq_thresh:
                v_label = "High Volatility"
                attrs.append("volatile")
            else:
                v_label = "Low/Medium Volatility"
                
            # 4. Intermittency Attribution
            if row["avg_demand_interval"] > self.config.get("adi_threshold", 1.32):
                i_label = "Intermittent"
                attrs.append("sparse")
            else:
                i_label = "Continuous"
            
            if not attrs:
                attrs.append("smooth_stable")

            trend_labels.append(t_label)
            season_labels.append(s_label)
            volatility_labels.append(v_label)
            intermittent_labels.append(i_label)
            combined_attributes.append(", ".join(attrs))

            # Primary routing label logic: deterministic hierarchy
            # Highly Seasonal > Highly Trending > Volatile > Smooth/Stable
            if "seasonal" in attrs:
                primary = "Seasonal_Expert"
            elif "trending" in attrs:
                primary = "Trend_Expert"
            elif "volatile" in attrs or "sparse" in attrs:
                primary = "Robust_Baseline_Expert"
            else:
                primary = "Standard_ML_Expert"
            primary_routing_label.append(primary)

        df["trend_regime"] = trend_labels
        df["seasonality_regime"] = season_labels
        df["volatility_regime"] = volatility_labels
        df["sparsity_regime"] = intermittent_labels
        df["demand_attributes"] = combined_attributes
        df["primary_routing_regime"] = primary_routing_label

        return df
