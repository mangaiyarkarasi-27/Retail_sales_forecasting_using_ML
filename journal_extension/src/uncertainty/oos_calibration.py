"""
oos_calibration.py
------------------
Generates leakage-safe out-of-sample (OOS) predictions and residuals
on the training partition using a 5-fold TimeSeriesSplit.
Saves residuals to oos_calibration_residuals.csv.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_oos_residuals():
    logger.info("Initializing leakage-safe Out-of-Sample Residual Generator...")
    
    # 1. Load preprocessed training dataset
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed training set not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Sort and add lag features (matching the conference configuration)
    df = df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
    df["sales_lag_7"] = df.groupby(["store", "item"])["sales"].shift(7)
    df["sales_lag_14"] = df.groupby(["store", "item"])["sales"].shift(14)
    df["sales_lag_30"] = df.groupby(["store", "item"])["sales"].shift(30)
    
    df["rolling_mean_7"] = df.groupby(["store", "item"])["sales"].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    df["rolling_mean_30"] = df.groupby(["store", "item"])["sales"].transform(
        lambda x: x.shift(1).rolling(window=30).mean()
    )
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["month_start"] = df["date"].dt.is_month_start.astype(int)
    df["month_end"] = df["date"].dt.is_month_end.astype(int)
    
    df = df.dropna().reset_index(drop=True)
    
    features = [
        "store", "item", "year", "month", "day", "weekday", "weekofyear", "quarter",
        "is_weekend", "month_start", "month_end",
        "sales_lag_7", "sales_lag_14", "sales_lag_30",
        "rolling_mean_7", "rolling_mean_30"
    ]
    target = "sales"
    
    # 2. Chronological Split (80% training partition, 20% test partition)
    df = df.sort_values(by="date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    # Pre-allocate predicted sales column
    train_df["pred_sales"] = np.nan
    test_df["pred_sales"] = np.nan
    
    # 3. 5-Fold TimeSeriesSplit to generate OOS predictions on Train Partition
    tscv = TimeSeriesSplit(n_splits=5)
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_df)):
        logger.info(f"OOS Calibration Fold {fold + 1}/5...")
        X_tr = train_df.iloc[tr_idx][features]
        y_tr = train_df.iloc[tr_idx][target]
        X_val = train_df.iloc[val_idx][features]
        
        xgb = XGBRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=8, 
            subsample=1.0, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        train_df.loc[val_idx, "pred_sales"] = xgb.predict(X_val)
        
    # Drop rows in train_df that do not have OOS predictions (first fold training data)
    train_df = train_df.dropna(subset=["pred_sales"]).reset_index(drop=True)
    
    # 4. Generate predictions on Test Partition (XGBoost fit on full train partition)
    logger.info("Generating predictions on untouched test partition...")
    final_xgb = XGBRegressor(
        n_estimators=50, learning_rate=0.1, max_depth=8, 
        subsample=1.0, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    final_xgb.fit(train_df[features], train_df[target])
    test_df["pred_sales"] = final_xgb.predict(test_df[features])
    
    # 5. Combine and calculate cumulative lead-time demand residuals
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
    
    lead_times = [1, 3, 5, 7]
    residual_records = []
    
    logger.info("Computing cumulative lead-time demand residuals...")
    
    for L in lead_times:
        for (s, i), group in full_df.groupby(["store", "item"]):
            # Group is sorted by date
            sales_arr = group["sales"].values
            preds_arr = group["pred_sales"].values
            dates_arr = group["date"].values
            
            n = len(group)
            for t in range(n - L):
                # Cumulative lead-time actual and forecast demand starting tomorrow (t+1) to (t+L)
                actual_lt_demand = np.sum(sales_arr[t + 1 : t + L + 1])
                predicted_lt_demand = np.sum(preds_arr[t + 1 : t + L + 1])
                absolute_error = abs(actual_lt_demand - predicted_lt_demand)
                
                residual_records.append({
                    "store": int(s),
                    "item": int(i),
                    "forecast_origin": dates_arr[t],
                    "lead_time": L,
                    "actual_lead_demand": round(float(actual_lt_demand), 4),
                    "predicted_lead_demand": round(float(predicted_lt_demand), 4),
                    "absolute_error": round(float(absolute_error), 4)
                })
                
    res_df = pd.DataFrame(residual_records)
    results_dir = "journal_extension/results/uncertainty"
    os.makedirs(results_dir, exist_ok=True)
    res_df.to_csv(os.path.join(results_dir, "oos_calibration_residuals.csv"), index=False)
    logger.info(f"OOS residuals saved successfully to {os.path.join(results_dir, 'oos_calibration_residuals.csv')}.")


if __name__ == "__main__":
    generate_oos_residuals()
