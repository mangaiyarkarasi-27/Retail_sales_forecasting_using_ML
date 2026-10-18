"""
forecast_vector.py
------------------
Assembles the leakage-free forecast vector [ŷ_t+1, ..., ŷ_t+7] at origin t
using the 7 global Direct Multi-Horizon models.
"""

import os
import joblib
import pandas as pd
import numpy as np
from decision_time_features import generate_origin_features, construct_target_calendar_features


class ForecastVectorOrchestrator:
    """
    Orchestrates the predictions of the 7 global XGBoost models to produce
    leakage-free future forecast vectors.
    """

    def __init__(self, models_dir: str = "journal_extension/results/forecasting_models"):
        self.models_dir = os.path.normpath(models_dir)
        self.models = {}
        self.feature_names = [
            "store", "item", "year", "month", "day", "weekday", "weekofyear", "quarter",
            "is_weekend", "month_start", "month_end",
            "sales_lag_7", "sales_lag_14", "sales_lag_30",
            "rolling_mean_7", "rolling_mean_30"
        ]
        self._load_models()

    def _load_models(self):
        for h in range(1, 8):
            model_path = os.path.join(self.models_dir, f"xgb_direct_h{h}.joblib")
            if os.path.exists(model_path):
                self.models[h] = joblib.load(model_path)

    def predict_horizon_batch(self, df: pd.DataFrame, h: int) -> np.ndarray:
        """
        Generates predictions for horizon h for all dates in df acting as origin t.
        Uses features based on sales <= t-1.
        """
        if h not in self.models:
            raise ValueError(f"Model for horizon h={h} is not loaded.")
            
        # 1. Base features calculated at row date (acting as origin date t)
        base_df = generate_origin_features(df)
        
        # 2. Add calendar features for target date t+h
        cal_features = construct_target_calendar_features(df["date"], h)
        
        # Assemble feature matrix
        X = pd.DataFrame(index=df.index)
        X["store"] = df["store"]
        X["item"] = df["item"]
        
        for col in cal_features.columns:
            X[col] = cal_features[col]
            
        for col in ["sales_lag_7", "sales_lag_14", "sales_lag_30", "rolling_mean_7", "rolling_mean_30"]:
            X[col] = base_df[col]
            
        # Handle NaN values (fill with training mean or zero if any, though there shouldn't be any
        # once the history is sufficient)
        X = X.fillna(0)
        
        return self.models[h].predict(X[self.feature_names])

    def generate_lead_time_forecasts(self, df: pd.DataFrame, max_horizon: int = 7) -> pd.DataFrame:
        """
        Generates daily point forecasts for horizons 1 to max_horizon.
        Adds columns pred_sales_h1 ... pred_sales_h7 to the input dataframe.
        """
        res_df = df.copy()
        for h in range(1, max_horizon + 1):
            res_df[f"pred_sales_h{h}"] = self.predict_horizon_batch(df, h)
        return res_df
