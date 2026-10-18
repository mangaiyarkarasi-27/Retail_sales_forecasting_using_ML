"""
run_uncertainty_experiment_corrected.py
--------------------------------------
Runs the corrected uncertainty experiment using pre-computed out-of-sample (OOS) calibration residuals.
Computes metrics, evaluates performance on the untouched test partition, saves results
to results/uncertainty/corrected/, and creates a comparison CSV against the original EXP-005 results.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    logger.info("Initializing Corrected Conformal Uncertainty Experiment (OOS residuals)...")
    
    # 1. Load pre-computed OOS residuals
    res_path = "journal_extension/results/uncertainty/oos_calibration_residuals.csv"
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"OOS calibration residuals not found at {res_path}")
        
    logger.info("Loading OOS residuals CSV (this may take a few seconds)...")
    res_df = pd.read_csv(res_path)
    res_df["forecast_origin"] = pd.to_datetime(res_df["forecast_origin"])
    
    # 2. Parameters
    window_size = 100
    lead_times = [1, 3, 5, 7]
    nominal_coverages = [0.80, 0.90, 0.95]
    
    calibrator = ConformalCalibrator(window_size=window_size, min_observations=20)
    
    # Identify test dates (2017-01-06 onwards)
    test_start_date = pd.to_datetime("2017-01-06")
    
    # Group by store, item, lead_time for fast iteration
    per_series_results = []
    
    logger.info("Evaluating rolling conformal intervals using OOS residuals...")
    
    for L in lead_times:
        logger.info(f"Processing Lead Time L={L}...")
        lt_df = res_df[res_df["lead_time"] == L].copy().sort_values(by=["store", "item", "forecast_origin"]).reset_index(drop=True)
        
        for alpha_nominal in nominal_coverages:
            alpha = 1.0 - alpha_nominal
            
            for (s, i), group in lt_df.groupby(["store", "item"]):
                # Reset index to allow fast integer slicing
                group = group.reset_index(drop=True)
                
                # Get the indices that belong to the test partition
                test_mask = group["forecast_origin"] >= test_start_date
                test_indices = group[test_mask].index.values
                
                if len(test_indices) == 0:
                    continue
                    
                errors_arr = group["absolute_error"].values
                preds_arr = group["predicted_lead_demand"].values
                actuals_arr = group["actual_lead_demand"].values
                
                lowers = []
                uppers = []
                actuals_eval = []
                
                for idx in test_indices:
                    # Calibration set: W elements preceding current origin
                    calib_errors = errors_arr[idx - window_size : idx]
                    
                    # Conformal Quantile
                    q_L = calibrator.compute_quantile(calib_errors, alpha=alpha)
                    
                    pred_val = preds_arr[idx]
                    actual_val = actuals_arr[idx]
                    
                    lower = max(0.0, pred_val - q_L)
                    upper = pred_val + q_L
                    
                    lowers.append(lower)
                    uppers.append(upper)
                    actuals_eval.append(actual_val)
                    
                lowers = np.array(lowers)
                uppers = np.array(uppers)
                actuals_eval = np.array(actuals_eval)
                
                # Compute metrics
                picp = calculate_picp(actuals_eval, lowers, uppers)
                miw = calculate_miw(lowers, uppers)
                score = calculate_interval_score(actuals_eval, lowers, uppers, alpha)
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
    
    # 3. Save Corrected Results
    corrected_dir = "journal_extension/results/uncertainty/corrected"
    os.makedirs(corrected_dir, exist_ok=True)
    per_series_df.to_csv(os.path.join(corrected_dir, "per_series_uncertainty_results.csv"), index=False)
    
    # Save corrected summary
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
    summary_df.to_csv(os.path.join(corrected_dir, "uncertainty_summary.csv"), index=False)
    logger.info("Saved corrected summary CSV.")
    
    # 4. Compare EXP-005 (Original) vs. Corrected
    logger.info("Comparing original EXP-005 vs. Corrected OOS calibration results...")
    orig_summary_path = "journal_extension/results/uncertainty/original_exp005/uncertainty_summary.csv"
    if not os.path.exists(orig_summary_path):
        raise FileNotFoundError(f"Original summary CSV not found at {orig_summary_path}")
        
    orig_summary = pd.read_csv(orig_summary_path)
    
    comparison_records = []
    for index, row in summary_df.iterrows():
        lt = row["lead_time"]
        cov = row["nominal_coverage"]
        
        orig_row = orig_summary[(orig_summary["lead_time"] == lt) & (orig_summary["nominal_coverage"] == cov)].iloc[0]
        
        comparison_records.append({
            "lead_time": int(lt),
            "nominal_coverage": cov,
            "original_coverage": orig_row["empirical_coverage"],
            "corrected_coverage": row["empirical_coverage"],
            "original_miw": orig_row["mean_interval_width"],
            "corrected_miw": row["mean_interval_width"],
            "original_interval_score": orig_row["interval_score"],
            "corrected_interval_score": row["interval_score"]
        })
        
    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv("journal_extension/results/uncertainty/calibration_correction_comparison.csv", index=False)
    logger.info("Saved comparison CSV to journal_extension/results/uncertainty/calibration_correction_comparison.csv.")
    
    # 5. Generate corrected research plots
    logger.info("Generating corrected uncertainty calibration plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot A: Nominal vs Empirical Coverage (Corrected)
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
    plt.title("Corrected Nominal vs. Empirical Coverage Across Lead Times", pad=15)
    plt.xlabel("Nominal Coverage Level")
    plt.ylabel("Empirical Coverage Level")
    plt.legend(title="Lead Time (Days)")
    plt.tight_layout()
    plt.savefig(os.path.join(corrected_dir, "nominal_vs_empirical_coverage.png"), dpi=300)
    plt.close()
    
    # Plot B: Mean Interval Width by Lead Time (Corrected)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=summary_df, 
        x="lead_time", 
        y="mean_interval_width", 
        hue="nominal_coverage",
        palette="Blues"
    )
    plt.title("Corrected Mean Interval Width by Lead Time and Coverage Level", pad=15)
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Mean Interval Width (Units)")
    plt.legend(title="Nominal Coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(corrected_dir, "mean_interval_width.png"), dpi=300)
    plt.close()
    
    logger.info("Corrected uncertainty quantification experiment completed successfully.")


if __name__ == "__main__":
    main()
