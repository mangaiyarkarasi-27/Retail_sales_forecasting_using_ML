"""
evaluate_model.py
-----------------
This module evaluates the performance of the trained sales forecasting model.
It calculates key regression metrics and visualizes the prediction performance.

Metrics computed:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R2 Coefficient)
- Mean Absolute Percentage Error (MAPE)

Outputs:
- Save evaluation metrics to results/metrics/evaluation_metrics.json.
- Save Actual vs. Predicted comparison plot to results/plots/actual_vs_predicted.png.
"""

import os
import json
import logging
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics.
    
    Parameters:
    -----------
    y_true : pd.Series
        Ground truth sales values.
    y_pred : np.ndarray
        Predicted sales values.
        
    Returns:
    --------
    Dict[str, float]
        A dictionary containing MAE, RMSE, R2, and MAPE metrics.
    """
    logger.info("Computing evaluation metrics...")
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero in MAPE calculation
    y_true_safe = np.where(y_true == 0, 1e-5, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    metrics = {
        "MAE": float(round(mae, 4)),
        "RMSE": float(round(rmse, 4)),
        "R2": float(round(r2, 4)),
        "MAPE_pct": float(round(mape, 4))
    }
    
    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics

def save_metrics_to_json(metrics: Dict[str, float], save_path: str = "results/metrics/evaluation_metrics.json") -> None:
    """
    Saves metrics dictionary to a JSON file.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Dictionary of computed performance metrics.
    save_path : str
        Target JSON filepath.
    """
    logger.info(f"Saving metrics to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info("Metrics saved successfully.")

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, save_path: str = "results/plots/actual_vs_predicted.png") -> None:
    """
    Generates a scatter plot comparing actual versus predicted sales values.
    
    Parameters:
    -----------
    y_true : pd.Series
        Ground truth sales values.
    y_pred : np.ndarray
        Predicted sales values.
    save_path : str
        File path where prediction scatter plot should be saved.
    """
    logger.info("Plotting actual vs predicted sales...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color="#8c564b")
    
    # 45-degree reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Perfect Forecast")
    
    plt.title("Actual vs. Predicted Sales")
    plt.xlabel("Actual Sales (Units)")
    plt.ylabel("Predicted Sales (Units)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Actual vs Predicted plot saved to {save_path}")

def run_evaluation_pipeline(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Executes prediction, computes metrics, plots differences and exports values.
    
    Parameters:
    -----------
    model : Any
        Trained model instance.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test targets.
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of evaluated performance metrics.
    """
    logger.info("Executing evaluation pipeline...")
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    save_metrics_to_json(metrics)
    plot_actual_vs_predicted(y_test, y_pred)
    
    logger.info("Evaluation pipeline complete.")
    return metrics

if __name__ == "__main__":
    from preprocessing import load_data, clean_data, engineer_features, prepare_and_split_data
    from train_model import load_trained_model
    
    try:
        raw_df = load_data("data/train.csv")
        cleaned_df = clean_data(raw_df)
        feat_df = engineer_features(cleaned_df)
        _, X_te, _, y_te = prepare_and_split_data(feat_df)
        
        model_instance = load_trained_model()
        run_evaluation_pipeline(model_instance, X_te, y_te)
        print("Evaluation module validated successfully!")
    except FileNotFoundError as e:
        logger.info(f"Dry-run notice: {e}")
