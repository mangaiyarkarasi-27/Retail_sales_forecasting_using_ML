"""
run_final_uncertainty_evaluation.py
------------------------------------
Executes the final out-of-sample uncertainty evaluation on the 2017 test period.
Highly optimized:
- Pre-groups residuals by store-item.
- Swaps nominal coverage loop to the innermost position to sort rolling window only once.
- Uses vector-based date arithmetic.
Saves metrics, summary files, slices analysis, and plots to results/uncertainty/final_2017/.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add local package import
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uncertainty")))

from forecast_vector import ForecastVectorOrchestrator
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
    logger.info("Initializing Final 2017 Uncertainty Evaluation...")
    
    # 1. Load Data
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed training data not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
    
    # 2. Chronological Split (Training 80% [2013-2016], Test 20% [2017])
    df_sorted = df.sort_values(by="date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)
    
    # 3. Load Forecasting Orchestrator
    orchestrator = ForecastVectorOrchestrator(models_dir="journal_extension/results/forecasting_models")
    
    # 4. Generate Point Forecasts for the test partition (2017)
    logger.info("Generating leakage-free point forecasts for the 2017 test period...")
    test_forecasts = orchestrator.generate_lead_time_forecasts(test_df)
    
    # 5. Load OOS residuals (EXP-012)
    residuals_path = "journal_extension/results/uncertainty/oos_direct/lead_time_oos_residuals.csv"
    if not os.path.exists(residuals_path):
        raise FileNotFoundError(f"OOS calibration residuals not found at {residuals_path}")
    res_df = pd.read_csv(residuals_path)
    res_df["forecast_origin"] = pd.to_datetime(res_df["forecast_origin"])
    
    # 6. Training-derived demand profiles (Volume & Seasonality computed on train_df)
    logger.info("Computing training-derived demand profiles...")
    profiles = []
    for (s, i), group in train_df.groupby(["store", "item"]):
        mean_vol = group["sales"].mean()
        acf_365 = 0.0
        if len(group) > 365:
            sales_series = group["sales"]
            acf_365 = sales_series.autocorr(lag=365)
            if np.isnan(acf_365):
                acf_365 = 0.0
        
        if mean_vol < 20:
            vol_cat = "Low"
        elif mean_vol <= 50:
            vol_cat = "Medium"
        else:
            vol_cat = "High"
            
        if acf_365 < 0.20:
            seas_cat = "Weak"
        elif acf_365 < 0.40:
            seas_cat = "Moderate"
        else:
            seas_cat = "Strong"
            
        profiles.append({
            "store": s,
            "item": i,
            "mean_volume": round(mean_vol, 4),
            "seasonality_strength": round(acf_365, 4),
            "volume_category": vol_cat,
            "seasonality_category": seas_cat
        })
    profiles_df = pd.DataFrame(profiles)
    
    # 7. Conformal Calibration & Evaluation Loops
    lead_times = [1, 3, 5, 7]
    nominal_coverages = [0.80, 0.90, 0.95]
    window_size = 100
    
    per_series_results = []
    
    logger.info("Running rolling conformal calibration and metric evaluations...")
    
    # Group test_forecasts by store, item once
    test_groups = {name: grp.sort_values(by="date").reset_index(drop=True) for name, grp in test_forecasts.groupby(["store", "item"])}
    
    for L in lead_times:
        logger.info(f"Evaluating Lead Time L={L}...")
        
        # Filter OOS residuals for lead time L and group once by store, item
        res_L = res_df[res_df["lead_time"] == L].copy().sort_values(by=["store", "item", "forecast_origin"]).reset_index(drop=True)
        res_groups = {name: grp.reset_index(drop=True) for name, grp in res_L.groupby(["store", "item"])}
        
        for (s, i), group in test_groups.items():
            if (s, i) not in res_groups:
                continue
            group_res = res_groups[(s, i)]
            res_origins = group_res["forecast_origin"].values
            res_errors = group_res["absolute_lead_error"].values
            
            # Align test group
            dates = group["date"].values
            sales = group["sales"].values
            
            # Pre-calculate cutoff dates array for numpy speed
            cutoff_dates = dates - np.timedelta64(L + 1, 'D')
            
            # Predicted cumulative lead-time demand array
            pred_lt_sum = np.zeros(len(group))
            for h in range(1, L + 1):
                pred_lt_sum += group[f"pred_sales_h{h}"].values
                
            # Initialize buffers for each nominal coverage
            buffers = {cov: {"lowers": [], "uppers": [], "actuals": []} for cov in nominal_coverages}
            
            # Rolling loop across test origins t
            n_days = len(group)
            for t in range(n_days - L):
                pred_val = pred_lt_sum[t]
                actual_val = np.sum(sales[t + 1 : t + L + 1])
                
                # Conformal calibration window: W=100 eligible OOS residuals satisfying r <= t - L - 1
                cutoff_date = cutoff_dates[t]
                
                # Optimized binary search
                eligible_count = np.searchsorted(res_origins, cutoff_date, side='right')
                
                if eligible_count < 20:
                    # Fallback
                    for cov in nominal_coverages:
                        if eligible_count > 0:
                            q_L = np.quantile(res_errors[:eligible_count], cov)
                        else:
                            q_L = 0.0
                        lower = max(0.0, pred_val - q_L)
                        upper = pred_val + q_L
                        buffers[cov]["lowers"].append(lower)
                        buffers[cov]["uppers"].append(upper)
                        buffers[cov]["actuals"].append(actual_val)
                else:
                    # Take the 100 most recent eligible residuals and sort once
                    window_errors = res_errors[max(0, eligible_count - window_size) : eligible_count]
                    sorted_window = np.sort(window_errors)
                    n_w = len(sorted_window)
                    
                    # Calculate quantiles for all coverages from the same sorted array
                    for cov in nominal_coverages:
                        alpha = 1.0 - cov
                        idx_1indexed = int(np.ceil((n_w + 1) * (1 - alpha)))
                        idx_0indexed = min(idx_1indexed - 1, n_w - 1)
                        idx_0indexed = max(idx_0indexed, 0)
                        q_L = sorted_window[idx_0indexed]
                        
                        lower = max(0.0, pred_val - q_L)
                        upper = pred_val + q_L
                        buffers[cov]["lowers"].append(lower)
                        buffers[cov]["uppers"].append(upper)
                        buffers[cov]["actuals"].append(actual_val)
                        
            # Compute and store metrics for each coverage level
            for cov in nominal_coverages:
                lowers = np.array(buffers[cov]["lowers"])
                uppers = np.array(buffers[cov]["uppers"])
                actuals = np.array(buffers[cov]["actuals"])
                
                alpha = 1.0 - cov
                picp = calculate_picp(actuals, lowers, uppers)
                miw = calculate_miw(lowers, uppers)
                score = calculate_interval_score(actuals, lowers, uppers, alpha)
                cov_err = calculate_coverage_error(picp, cov)
                
                per_series_results.append({
                    "store": int(s),
                    "item": int(i),
                    "lead_time": L,
                    "nominal_coverage": cov,
                    "empirical_coverage": round(picp, 4),
                    "coverage_error": round(cov_err, 4),
                    "mean_interval_width": round(miw, 4),
                    "interval_score": round(score, 4),
                    "evaluated_origins": len(actuals)
                })
                
    per_series_df = pd.DataFrame(per_series_results)
    
    # 8. Save Outputs
    output_dir = "journal_extension/results/uncertainty/final_2017"
    os.makedirs(output_dir, exist_ok=True)
    per_series_df.to_csv(os.path.join(output_dir, "per_series_uncertainty_results.csv"), index=False)
    
    # A. Aggregate summary
    summary_records = []
    for (L, cov), group in per_series_df.groupby(["lead_time", "nominal_coverage"]):
        summary_records.append({
            "lead_time": L,
            "nominal_coverage": cov,
            "empirical_coverage": round(group["empirical_coverage"].mean(), 4),
            "coverage_error": round(group["coverage_error"].mean(), 4),
            "mean_interval_width": round(group["mean_interval_width"].mean(), 4),
            "interval_score": round(group["interval_score"].mean(), 4),
            "evaluated_origins": int(group["evaluated_origins"].iloc[0])
        })
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(output_dir, "uncertainty_summary.csv"), index=False)
    
    # B. Slice-based summaries
    merged_df = per_series_df.merge(profiles_df, on=["store", "item"])
    
    # Volume slice summary
    vol_records = []
    for (L, cov, vol), group in merged_df.groupby(["lead_time", "nominal_coverage", "volume_category"]):
        vol_records.append({
            "lead_time": L,
            "nominal_coverage": cov,
            "volume_category": vol,
            "empirical_coverage": round(group["empirical_coverage"].mean(), 4),
            "coverage_error": round(group["coverage_error"].mean(), 4),
            "mean_interval_width": round(group["mean_interval_width"].mean(), 4),
            "interval_score": round(group["interval_score"].mean(), 4)
        })
    pd.DataFrame(vol_records).to_csv(os.path.join(output_dir, "coverage_by_volume.csv"), index=False)
    
    # Seasonality slice summary
    seas_records = []
    for (L, cov, seas), group in merged_df.groupby(["lead_time", "nominal_coverage", "seasonality_category"]):
        seas_records.append({
            "lead_time": L,
            "nominal_coverage": cov,
            "seasonality_category": seas,
            "empirical_coverage": round(group["empirical_coverage"].mean(), 4),
            "coverage_error": round(group["coverage_error"].mean(), 4),
            "mean_interval_width": round(group["mean_interval_width"].mean(), 4),
            "interval_score": round(group["interval_score"].mean(), 4)
        })
    pd.DataFrame(seas_records).to_csv(os.path.join(output_dir, "coverage_by_seasonality.csv"), index=False)
    
    # 9. Generate Diagnostics & Plots
    logger.info("Generating evaluation diagnostics & plots...")
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
    plt.title("EXP-013 Final Nominal vs. Empirical Coverage on 2017 Test period", pad=15)
    plt.xlabel("Nominal Coverage Level")
    plt.ylabel("Empirical Coverage Level")
    plt.legend(title="Lead Time (Days)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nominal_vs_empirical_coverage.png"), dpi=300)
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
    plt.title("EXP-013 Mean Interval Width by Lead Time and Coverage Level", pad=15)
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Mean Interval Width (Units)")
    plt.legend(title="Nominal Coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "interval_width_by_lead_time.png"), dpi=300)
    plt.close()
    
    logger.info("EXP-013 Final Uncertainty evaluation executed successfully.")


if __name__ == "__main__":
    main()
