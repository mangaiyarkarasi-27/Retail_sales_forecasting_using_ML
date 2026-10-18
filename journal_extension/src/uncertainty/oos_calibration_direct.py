"""
oos_calibration_direct.py
-------------------------
Implements out-of-sample (OOS) residual generation for the 7 Direct Multi-Horizon
models on the 2013-2016 training partition using a 5-fold TimeSeriesSplit.
Saves residuals under results/uncertainty/oos_direct/.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# Add local package import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting")))

from decision_time_features import generate_origin_features, construct_target_calendar_features

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OOSResidualsGeneratorDirect:
    """
    Generates leakage-safe out-of-sample forecasting residuals using
    5-fold TimeSeriesSplit on the training partition.
    """

    def __init__(self, data_path: str = "data/processed_train.csv"):
        self.data_path = os.path.normpath(data_path)
        self.xgb_params = {
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 8,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1
        }
        self.feature_names = [
            "store", "item", "year", "month", "day", "weekday", "weekofyear", "quarter",
            "is_weekend", "month_start", "month_end",
            "sales_lag_7", "sales_lag_14", "sales_lag_30",
            "rolling_mean_7", "rolling_mean_30"
        ]

    def construct_horizon_dataset(self, df: pd.DataFrame, h: int) -> pd.DataFrame:
        """
        Builds the shifted dataset for training model M_h.
        """
        # Base features calculated at row date (acting as origin date t)
        base_df = generate_origin_features(df)
        
        # Shift base features by h days
        shifted_features = base_df.groupby(["store", "item"])[
            ["sales_lag_7", "sales_lag_14", "sales_lag_30", "rolling_mean_7", "rolling_mean_30"]
        ].shift(h)
        
        dataset = pd.DataFrame(index=df.index)
        dataset["date"] = df["date"]
        dataset["store"] = df["store"]
        dataset["item"] = df["item"]
        dataset["sales"] = df["sales"]
        
        cal_features = construct_target_calendar_features(df["date"], 0)
        for col in cal_features.columns:
            dataset[col] = cal_features[col]
            
        for col in shifted_features.columns:
            dataset[col] = shifted_features[col]
            
        return dataset

    def generate_oos_forecasts(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs TimeSeriesSplit cross-validation folds and predicts on validation sets
        to collect OOS forecasts for horizons 1..7.
        """
        logger.info(f"Loading processed training set from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
        
        # Chronological Split (first 80% is the training partition)
        df_sorted = df.sort_values(by="date").reset_index(drop=True)
        split_idx = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
        
        # 5-fold TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        # We will collect OOS predictions in validation splits
        # Create columns for OOS predictions for horizons 1..7
        for h in range(1, 8):
            train_df[f"pred_sales_h{h}"] = np.nan
            
        logger.info("Starting TimeSeriesSplit cross-validation training...")
        
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_df)):
            logger.info(f"--- Processing OOS Fold {fold + 1}/5 ---")
            
            # Sub-train and sub-val dataframes
            sub_train = train_df.iloc[tr_idx].copy().reset_index(drop=True)
            sub_val = train_df.iloc[val_idx].copy().reset_index(drop=True)
            
            # Predict for each horizon h using model trained on sub_train
            for h in range(1, 8):
                logger.info(f"Fold {fold+1} Horizon h={h} training...")
                
                # 1. Build training set for M_h
                train_ds = self.construct_horizon_dataset(sub_train, h).dropna()
                X_tr = train_ds[self.feature_names]
                y_tr = train_ds["sales"]
                
                # 2. Fit Model
                model = XGBRegressor(**self.xgb_params)
                model.fit(X_tr, y_tr)
                
                # 3. Build features at validation origins to predict t+h
                # Validation origins are dates in sub_val.
                # Base features are computed at origin t (sales <= t-1).
                val_base = generate_origin_features(sub_val)
                val_cal = construct_target_calendar_features(sub_val["date"], h)
                
                X_val = pd.DataFrame(index=sub_val.index)
                X_val["store"] = sub_val["store"]
                X_val["item"] = sub_val["item"]
                for col in val_cal.columns:
                    X_val[col] = val_cal[col]
                for col in ["sales_lag_7", "sales_lag_14", "sales_lag_30", "rolling_mean_7", "rolling_mean_30"]:
                    X_val[col] = val_base[col]
                
                X_val = X_val.fillna(0)
                
                # Make predictions
                preds = model.predict(X_val[self.feature_names])
                
                # Store back in train_df
                train_df.loc[val_idx, f"pred_sales_h{h}"] = preds
                
        # Drop rows where OOS predictions are NaN (the first fold training partition has no OOS preds)
        oos_df = train_df.dropna(subset=["pred_sales_h1"]).reset_index(drop=True)
        return oos_df
