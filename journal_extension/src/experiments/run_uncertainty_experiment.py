"""
run_uncertainty_experiment.py
----------------------------
Executes the Rolling-Origin Conformal Uncertainty Quantification experiment on the retail dataset.
Evaluates empirical coverage, mean interval width, and interval scores across lead times and nominal coverages.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Add src directories to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uncertainty")))

from conformal_prediction import ConformalCalibrator
from uncertainty_metrics import (
    calculate_picp,
    calculate_miw,
    calculate_interval_score,
    calculate_coverage_error
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Conformal Uncertainty Experiment...")
    
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
    
    logger.info("Training Point Forecast Model (XGBoost)...")
    xgb = XGBRegressor(
        n_estimators=50, learning_rate=0.1, max_depth=8, 
        subsample=1.0, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    xgb.fit(train_df[features], train_df[target])
    
    # Generate predictions on both train and test partitions
    train_df["pred_sales"] = xgb.predict(train_df[features])
    test_df["pred_sales"] = xgb.predict(test_df[features])
    
    # Combined dataframe to allow rolling window residual computation across split boundary
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
    
    # 3. Parameters
    window_size = 100
    lead_times = [1, 3, 5, 7]
    nominal_coverages = [0.80, 0.90, 0.95]
    
    store_item_pairs = full_df[["store", "item"]].drop_duplicates().values
    
    per_series_results = []
    
    # Calibrator instance
    calibrator = ConformalCalibrator(window_size=window_size, min_observations=20)
    
    logger.info("Evaluating rolling conformal uncertainty across all series...")
    
    for L in lead_times:
        logger.info(f"Running uncertainty calibration for Lead Time L={L}...")
        for alpha_nominal in nominal_coverages:
            alpha = 1.0 - alpha_nominal
            
            for s, i in store_item_pairs:
                # Get series data
                series_data = full_df[(full_df["store"] == s) & (full_df["item"] == i)].copy().reset_index(drop=True)
                
                # Check actual lead-time demand D_{t, L}
                # Rolling sum of next L days actual demand
                series_data["actual_lt_demand"] = series_data["sales"].rolling(L).sum().shift(-L)
                series_data["pred_lt_demand"] = series_data["pred_sales"].rolling(L).sum().shift(-L)
                
                series_data = series_data.dropna().reset_index(drop=True)
                if len(series_data) < window_size + L:
                    continue
                    
                # Calculate daily absolute lead-time prediction error
                series_data["lt_abs_err"] = (series_data["actual_lt_demand"] - series_data["pred_lt_demand"]).abs()
                
                # Out-of-Sample Test set filter (keep dates belonging to test partition)
                test_dates = test_df["date"].unique()
                test_mask = series_data["date"].isin(test_dates)
                
                # We will slide through the test set indices to generate prediction intervals
                test_indices = series_data[test_mask].index.values
                
                if len(test_indices) == 0:
                    continue
                    
                lowers = []
                uppers = []
                actuals = []
                widths = []
                
                # Pre-convert to numpy arrays for 100x faster slicing
                lt_abs_err_arr = series_data["lt_abs_err"].values
                pred_lt_demand_arr = series_data["pred_lt_demand"].values
                actual_lt_demand_arr = series_data["actual_lt_demand"].values
                
                for idx in test_indices:
                    # Calibration set of the last 100 days preceding prediction date
                    # Residuals are shift(1) to avoid leakage
                    calib_errors = lt_abs_err_arr[idx - window_size : idx]
                    
                    # Conformal Quantile
                    q_L = calibrator.compute_quantile(calib_errors, alpha=alpha)
                    
                    pred_val = pred_lt_demand_arr[idx]
                    actual_val = actual_lt_demand_arr[idx]
                    
                    lower = max(0.0, pred_val - q_L)
                    upper = pred_val + q_L
                    
                    lowers.append(lower)
                    uppers.append(upper)
                    actuals.append(actual_val)
                    widths.append(upper - lower)
                    
                lowers = np.array(lowers)
                uppers = np.array(uppers)
                actuals = np.array(actuals)
                
                # Compute metrics
                picp = calculate_picp(actuals, lowers, uppers)
                miw = calculate_miw(lowers, uppers)
                score = calculate_interval_score(actuals, lowers, uppers, alpha)
                cov_err = calculate_coverage_error(picp, alpha_nominal)
                
                per_series_results.append({
                    "store": int(s),
                    "item": int(i),
                    "lead_time": L,
                    "nominal_coverage": alpha_nominal,
                    "empirical_coverage": round(picp, 4),
                    "mean_interval_width": round(miw, 4),
                    "interval_score": round(score, 4),
                    "coverage_error": round(cov_err, 4),
                    "calibration_sample_count": len(test_indices)
                })
                
    per_series_df = pd.DataFrame(per_series_results)
    
    # 4. Save per-series results
    results_dir = "journal_extension/results/uncertainty"
    os.makedirs(results_dir, exist_ok=True)
    per_series_df.to_csv(os.path.join(results_dir, "per_series_uncertainty_results.csv"), index=False)
    logger.info("Saved per-series uncertainty results.")
    
    # 5. Compute and Save Aggregate Summary
    summary_records = []
    for (lt, cov), group in per_series_df.groupby(["lead_time", "nominal_coverage"]):
        summary_records.append({
            "lead_time": lt,
            "nominal_coverage": cov,
            "empirical_coverage": round(group["empirical_coverage"].mean(), 4),
            "coverage_error": round(group["coverage_error"].mean(), 4),
            "mean_interval_width": round(group["mean_interval_width"].mean(), 4),
            "interval_score": round(group["interval_score"].mean(), 4),
            "number_of_evaluated_series": len(group)
        })
        
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(results_dir, "uncertainty_summary.csv"), index=False)
    logger.info("Saved uncertainty summary.")
    
    # 6. Generate Research Figures
    logger.info("Generating uncertainty calibration plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot A: Nominal vs Empirical Coverage
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=summary_df, 
        x="nominal_coverage", 
        y="empirical_coverage", 
        hue="lead_time", 
        style="lead_time", 
        markers=True, 
        dashes=False,
        palette="tab10"
    )
    plt.plot([0.80, 0.95], [0.80, 0.95], color="red", linestyle="--", label="Ideal Calibration")
    plt.title("Nominal vs. Empirical Coverage Across Lead Times", pad=15)
    plt.xlabel("Nominal Coverage Level")
    plt.ylabel("Empirical Coverage Level")
    plt.legend(title="Lead Time (Days)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "nominal_vs_empirical_coverage.png"), dpi=300)
    plt.close()
    
    # Plot B: Mean Interval Width by Lead Time
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=summary_df, 
        x="lead_time", 
        y="mean_interval_width", 
        hue="nominal_coverage",
        palette="Blues"
    )
    plt.title("Mean Interval Width by Lead Time and Coverage Level", pad=15)
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Mean Interval Width (Units)")
    plt.legend(title="Nominal Coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mean_interval_width.png"), dpi=300)
    plt.close()
    
    # Merge with Demand Profiles for Group Slicing
    profiles_path = "journal_extension/results/model_selection/demand_profiles.csv"
    if os.path.exists(profiles_path):
        profiles_df = pd.read_csv(profiles_path)
        merged = per_series_df.merge(
            profiles_df[["store", "item", "mean_sales", "cv_sales", "lag365_autocorr"]],
            on=["store", "item"],
            how="left"
        )
        
        # Plot C: Interval Width vs Demand Volume (for L=3, Nominal=90%)
        subset_plot = merged[(merged["lead_time"] == 3) & (merged["nominal_coverage"] == 0.90)]
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=subset_plot, x="mean_sales", y="mean_interval_width", alpha=0.7)
        plt.title("Mean Interval Width vs. Demand Volume (L=3, 90% Coverage)", pad=15)
        plt.xlabel("Mean Historical Sales (Units/Day)")
        plt.ylabel("Mean Prediction Interval Width (Units)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "interval_width_vs_volume.png"), dpi=300)
        plt.close()
        
    logger.info("Conformal uncertainty quantification experiment completed successfully.")


if __name__ == "__main__":
    main()
