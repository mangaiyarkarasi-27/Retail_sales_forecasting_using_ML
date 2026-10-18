"""
run_direct_multihorizon_validation.py
-------------------------------------
Executes chronological training and validation of the Direct Multi-Horizon Point Forecasting layer.
Evaluates MAE/RMSE on the 2016 validation period and serializes model weights.
"""

import os
import time
import logging
import pandas as pd
import numpy as np

# Add src directories to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting")))

from direct_multihorizon import DirectMultiHorizonRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Direct Multi-Horizon Training Layer...")
    
    # 1. Load preprocessed training dataset
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed training set not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Sort chronologically by store, item, date
    df = df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
    
    # 2. Initialize Regressor
    regressor = DirectMultiHorizonRegressor()
    
    # 3. Train and Validate each model M_h
    validation_records = []
    
    for h in range(1, 8):
        metrics = regressor.train_horizon_model(df, h)
        validation_records.append(metrics)
        
    val_df = pd.DataFrame(validation_records)
    
    # Save validation results
    forecasting_results_dir = "journal_extension/results/forecasting"
    os.makedirs(forecasting_results_dir, exist_ok=True)
    val_df.to_csv(os.path.join(forecasting_results_dir, "direct_multihorizon_validation.csv"), index=False)
    
    # 4. Compute Aggregate Metrics (weighted by active rows, but splits are balanced so simple mean is appropriate)
    mean_mae = val_df["MAE"].mean()
    mean_rmse = val_df["RMSE"].mean()
    
    logger.info("=========================================")
    logger.info("Aggregate Validation Summary (2016 period):")
    logger.info(f"Aggregate MAE: {mean_mae:.4f}")
    logger.info(f"Aggregate RMSE: {mean_rmse:.4f}")
    logger.info("=========================================")
    
    summary_df = pd.DataFrame([{
        "aggregate_validation_mae": round(mean_mae, 4),
        "aggregate_validation_rmse": round(mean_rmse, 4),
        "number_of_trained_models": 7,
        "features_used": ", ".join(regressor.feature_names),
        "target_boundary": "Validation target dates strictly <= 2016-12-31"
    }])
    summary_df.to_csv(os.path.join(forecasting_results_dir, "direct_multihorizon_summary.csv"), index=False)
    logger.info("Trained models and validation reports saved successfully.")


if __name__ == "__main__":
    main()
