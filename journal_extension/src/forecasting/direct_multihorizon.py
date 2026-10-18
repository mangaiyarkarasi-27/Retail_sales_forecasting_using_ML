"""
direct_multihorizon.py
----------------------
Implements the training, serialization, and validation of the 7 global
Direct Multi-Horizon XGBoost models.
"""

import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from decision_time_features import generate_origin_features, construct_target_calendar_features

logger = logging.getLogger(__name__)


class DirectMultiHorizonRegressor:
    """
    Manages the 7 global horizon-specific models (M_1 ... M_7).
    """

    def __init__(self, models_dir: str = "journal_extension/results/forecasting_models"):
        self.models_dir = os.path.normpath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        self.models = {}
        
        # Frozen XGBoost Hyperparameters (from verified conference configuration)
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
        Constructs the training/validation dataset for horizon h by shifting the features
        by h steps per store-item group, and aligning them with the target sales.
        """
        # Ensure sorting
        df = df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
        
        # Base features calculated at row date (which will serve as origin date t)
        base_df = generate_origin_features(df)
        
        # To align features from origin t with target date t+h:
        # We shift the base features by h days within each group.
        shifted_features = base_df.groupby(["store", "item"])[
            ["sales_lag_7", "sales_lag_14", "sales_lag_30", "rolling_mean_7", "rolling_mean_30"]
        ].shift(h)
        
        # Target details (aligned to row date, which acts as the target date t+h)
        dataset = pd.DataFrame(index=df.index)
        dataset["date"] = df["date"]
        dataset["store"] = df["store"]
        dataset["item"] = df["item"]
        dataset["sales"] = df["sales"]
        
        # Add target calendar features (which are known in advance)
        cal_features = construct_target_calendar_features(df["date"], 0)
        for col in cal_features.columns:
            dataset[col] = cal_features[col]
            
        # Add shifted base features
        for col in shifted_features.columns:
            dataset[col] = shifted_features[col]
            
        # Drop rows with NaNs (which occur at the beginning of each group due to shifting)
        dataset = dataset.dropna().reset_index(drop=True)
        return dataset

    def train_horizon_model(self, df: pd.DataFrame, h: int) -> Dict[str, float]:
        """
        Trains and validates model M_h using the 2013-2015 train and 2016 validation splits.
        """
        logger.info(f"--- Training Model M_h={h} (predicts y_t+{h} from origin t) ---")
        
        # 1. Construct dataset for horizon h
        dataset = self.construct_horizon_dataset(df, h)
        
        # 2. Chronological Splits (Target date based)
        train_mask = dataset["date"] <= pd.to_datetime("2015-12-31")
        val_mask = (dataset["date"] >= pd.to_datetime("2016-01-01")) & (dataset["date"] <= pd.to_datetime("2016-12-31"))
        
        train_data = dataset[train_mask]
        val_data = dataset[val_mask]
        
        logger.info(f"Train Rows: {len(train_data)} (Targets <= 2015-12-31)")
        logger.info(f"Val Rows: {len(val_data)} (Targets in 2016)")
        
        X_train = train_data[self.feature_names]
        y_train = train_data["sales"]
        X_val = val_data[self.feature_names]
        y_val = val_data["sales"]
        
        # 3. Train Model
        model = XGBRegressor(**self.xgb_params)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info(f"Model M_h={h} trained in {training_time:.2f} seconds.")
        
        # Save model
        model_path = os.path.join(self.models_dir, f"xgb_direct_h{h}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model weights saved to {model_path}")
        self.models[h] = model
        
        # 4. Predict and evaluate
        start_pred = time.time()
        preds = model.predict(X_val)
        prediction_time = time.time() - start_pred
        
        mae = np.mean(np.abs(y_val - preds))
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        logger.info(f"Horizon h={h} Validation MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        
        return {
            "horizon": h,
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "training_time": round(training_time, 4),
            "prediction_time": round(prediction_time, 4)
        }
