"""
run_uncertainty_diagnostics.py
------------------------------
Executes EXP-014: Comprehensive diagnosis of the final 2017 uncertainty coverage failure.
Optimized with vectorized pandas/numpy routines for sub-minute execution.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directories to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uncertainty")))

from forecast_vector import ForecastVectorOrchestrator
from conformal_prediction import ConformalCalibrator
from uncertainty_metrics import calculate_picp, calculate_miw, calculate_interval_score, calculate_coverage_error

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing EXP-014: Uncertainty Failure Diagnosis...")
    output_dir = "journal_extension/results/uncertainty/diagnostics_2017"
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Load Data
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)

    # 80% train / 20% test chronological split
    df_sorted = df.sort_values(by="date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)

    # Load direct multihorizon forecasting orchestrator
    orchestrator = ForecastVectorOrchestrator(models_dir="journal_extension/results/forecasting_models")
    logger.info("Generating 2017 direct point forecasts...")
    test_forecasts = orchestrator.generate_lead_time_forecasts(test_df)
    test_forecasts["date"] = pd.to_datetime(test_forecasts["date"])
    test_forecasts = test_forecasts.sort_values(by=["store", "item", "date"]).reset_index(drop=True)

    # Load EXP-012 OOS calibration residuals
    residuals_path = "journal_extension/results/uncertainty/oos_direct/lead_time_oos_residuals.csv"
    if not os.path.exists(residuals_path):
        raise FileNotFoundError(f"Residuals not found at {residuals_path}")
    res_df = pd.read_csv(residuals_path)
    res_df["forecast_origin"] = pd.to_datetime(res_df["forecast_origin"])

    # Compute training-derived demand profiles
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
        
        vol_cat = "Low" if mean_vol < 20 else ("Medium" if mean_vol <= 50 else "High")
        seas_cat = "Weak" if acf_365 < 0.20 else ("Moderate" if acf_365 < 0.40 else "Strong")
            
        profiles.append({
            "store": s,
            "item": i,
            "mean_volume": round(mean_vol, 4),
            "seasonality_strength": round(acf_365, 4),
            "volume_category": vol_cat,
            "seasonality_category": seas_cat
        })
    profiles_df = pd.DataFrame(profiles)

    # =========================================================================
    # TASK 2: Vectorized Point Forecast Bias Across Horizons (h=1..7)
    # =========================================================================
    logger.info("TASK 2: Computing point forecast bias across horizons...")
    horizon_bias_records = []
    
    # Store daily error series in test_forecasts
    for h in range(1, 8):
        test_forecasts[f"actual_h{h}"] = test_forecasts.groupby(["store", "item"])["sales"].shift(-h)
        test_forecasts[f"error_h{h}"] = test_forecasts[f"actual_h{h}"] - test_forecasts[f"pred_sales_h{h}"]
        test_forecasts[f"abs_error_h{h}"] = np.abs(test_forecasts[f"error_h{h}"])

        valid_mask = test_forecasts[f"actual_h{h}"].notna()
        acts = test_forecasts.loc[valid_mask, f"actual_h{h}"].values
        preds = test_forecasts.loc[valid_mask, f"pred_sales_h{h}"].values
        errs = test_forecasts.loc[valid_mask, f"error_h{h}"].values

        mean_act = np.mean(acts)
        mean_prd = np.mean(preds)
        mean_err = np.mean(errs)
        median_err = np.median(errs)
        mae = np.mean(np.abs(errs))
        rmse = np.sqrt(np.mean(errs ** 2))
        bias_ratio = mean_err / mean_act if mean_act != 0 else 0.0
        mape = np.mean(np.abs(errs) / np.maximum(acts, 1.0)) * 100.0

        horizon_bias_records.append({
            "horizon": h,
            "mean_actual_demand": round(float(mean_act), 4),
            "mean_prediction": round(float(mean_prd), 4),
            "mean_error": round(float(mean_err), 4),
            "median_error": round(float(median_err), 4),
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "bias_ratio": round(float(bias_ratio), 4),
            "MAPE_percent": round(float(mape), 4),
            "total_observations": int(len(errs))
        })

    horizon_bias_df = pd.DataFrame(horizon_bias_records)
    horizon_bias_df.to_csv(os.path.join(output_dir, "forecast_bias_by_horizon.csv"), index=False)
    logger.info(f"Forecast Bias by Horizon:\n{horizon_bias_df[['horizon', 'mean_actual_demand', 'mean_prediction', 'mean_error', 'bias_ratio', 'MAE']]}")

    # Plot Forecast Bias by Horizon
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(data=horizon_bias_df, x="horizon", y="mean_error", color="coral")
    plt.title("Mean Forecast Error by Horizon (2017)", pad=10)
    plt.xlabel("Forecast Horizon h (Days)")
    plt.ylabel("Mean Signed Error (Actual - Pred)")
    plt.axhline(0, color="black", linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    sns.barplot(data=horizon_bias_df, x="horizon", y="bias_ratio", color="steelblue")
    plt.title("Bias Ratio (Mean Error / Mean Demand)", pad=10)
    plt.xlabel("Forecast Horizon h (Days)")
    plt.ylabel("Bias Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forecast_bias_by_horizon.png"), dpi=300)
    plt.close()

    # =========================================================================
    # TASK 3: Bias Over Time (Quarters in 2017)
    # =========================================================================
    logger.info("TASK 3: Computing forecast bias by quarter...")
    test_forecasts["quarter"] = test_forecasts["date"].dt.to_period("Q").astype(str).str.replace(r"\d{4}", "", regex=True)
    # Ensure Q1..Q4 names
    test_forecasts["quarter"] = "Q" + test_forecasts["date"].dt.quarter.astype(str)
    
    quarter_bias_records = []
    for q_name in ["Q1", "Q2", "Q3", "Q4"]:
        q_mask = test_forecasts["quarter"] == q_name
        for h in range(1, 8):
            valid_mask = q_mask & test_forecasts[f"actual_h{h}"].notna()
            act_vals = test_forecasts.loc[valid_mask, f"actual_h{h}"].values
            prd_vals = test_forecasts.loc[valid_mask, f"pred_sales_h{h}"].values
            err_vals = test_forecasts.loc[valid_mask, f"error_h{h}"].values
            
            mean_act = np.mean(act_vals)
            mean_prd = np.mean(prd_vals)
            mean_err = np.mean(err_vals)
            median_err = np.median(err_vals)
            mae = np.mean(np.abs(err_vals))
            rmse = np.sqrt(np.mean(err_vals ** 2))
            bias_ratio = mean_err / mean_act if mean_act != 0 else 0.0

            quarter_bias_records.append({
                "quarter": q_name,
                "horizon": h,
                "mean_actual_demand": round(float(mean_act), 4),
                "mean_prediction": round(float(mean_prd), 4),
                "mean_error": round(float(mean_err), 4),
                "median_error": round(float(median_err), 4),
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "bias_ratio": round(float(bias_ratio), 4),
                "total_observations": int(len(err_vals))
            })

    quarter_bias_df = pd.DataFrame(quarter_bias_records)
    quarter_bias_df.to_csv(os.path.join(output_dir, "forecast_bias_by_quarter.csv"), index=False)

    # Plot Forecast Bias Over 2017 Quarters
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=quarter_bias_df, x="quarter", y="mean_error", hue="horizon", marker="o", palette="viridis")
    plt.title("Forecast Bias (Mean Signed Error) Across 2017 Quarters", pad=12)
    plt.xlabel("2017 Quarter")
    plt.ylabel("Mean Signed Error (Actual - Pred)")
    plt.axhline(0, color="red", linestyle="--", alpha=0.6, label="Zero Bias")
    plt.legend(title="Horizon h (Days)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forecast_bias_over_2017.png"), dpi=300)
    plt.close()

    # =========================================================================
    # TASK 4, 5, 7, 8: Conformal Evaluation & Diagnostics for Lead Times
    # =========================================================================
    logger.info("TASK 4, 5, 7, 8: Evaluating lead times, residuals, quantiles, and violations...")
    lead_times = [1, 3, 5, 7]
    nominal_coverages = [0.80, 0.90, 0.95]
    window_size = 100

    test_groups = {name: grp.sort_values(by="date").reset_index(drop=True) for name, grp in test_forecasts.groupby(["store", "item"])}
    
    cal_vs_test_records = []
    quantile_vs_actual_records = []
    violation_records = []
    lead_time_eval_dfs = {}

    for L in lead_times:
        logger.info(f"Processing Lead Time L={L}...")
        res_L = res_df[res_df["lead_time"] == L].copy().sort_values(by=["store", "item", "forecast_origin"]).reset_index(drop=True)
        res_groups = {name: grp.reset_index(drop=True) for name, grp in res_L.groupby(["store", "item"])}

        eval_records_L = []

        for (s, i), group in test_groups.items():
            if (s, i) not in res_groups:
                continue
            group_res = res_groups[(s, i)]
            res_origins = group_res["forecast_origin"].values
            res_errors = group_res["absolute_lead_error"].values

            dates = group["date"].values
            sales = group["sales"].values
            cutoff_dates = dates - np.timedelta64(L + 1, 'D')

            pred_lt_sum = np.zeros(len(group))
            for h in range(1, L + 1):
                pred_lt_sum += group[f"pred_sales_h{h}"].values

            n_days = len(group)
            for t in range(n_days - L):
                pred_val = pred_lt_sum[t]
                actual_val = np.sum(sales[t + 1 : t + L + 1])
                signed_err = actual_val - pred_val
                abs_err = abs(signed_err)

                cutoff_date = cutoff_dates[t]
                eligible_count = np.searchsorted(res_origins, cutoff_date, side='right')

                if eligible_count < 20:
                    q_dict = {cov: (float(np.quantile(res_errors[:eligible_count], cov)) if eligible_count > 0 else 0.0) for cov in nominal_coverages}
                else:
                    window_errors = res_errors[max(0, eligible_count - window_size) : eligible_count]
                    sorted_window = np.sort(window_errors)
                    n_w = len(sorted_window)
                    q_dict = {}
                    for cov in nominal_coverages:
                        alpha = 1.0 - cov
                        idx_1indexed = int(np.ceil((n_w + 1) * (1 - alpha)))
                        idx_0indexed = min(max(idx_1indexed - 1, 0), n_w - 1)
                        q_dict[cov] = float(sorted_window[idx_0indexed])

                eval_records_L.append({
                    "store": s,
                    "item": i,
                    "origin": dates[t],
                    "lead_time": L,
                    "pred_lead_demand": pred_val,
                    "actual_lead_demand": actual_val,
                    "signed_error": signed_err,
                    "abs_error": abs_err,
                    "q_80": q_dict[0.80],
                    "q_90": q_dict[0.90],
                    "q_95": q_dict[0.95],
                })

        eval_df_L = pd.DataFrame(eval_records_L)
        lead_time_eval_dfs[L] = eval_df_L

        # --- Task 4: Calibration vs Test Residual Stats ---
        cal_signed = res_L["signed_lead_error"].values
        cal_abs = res_L["absolute_lead_error"].values
        test_signed = eval_df_L["signed_error"].values
        test_abs = eval_df_L["abs_error"].values

        cal_mean = np.mean(cal_signed)
        test_mean = np.mean(test_signed)
        cal_std = np.std(cal_signed)
        test_std = np.std(test_signed)
        cal_mae = np.mean(cal_abs)
        test_mae = np.mean(test_abs)
        cal_rmse = np.sqrt(np.mean(cal_signed ** 2))
        test_rmse = np.sqrt(np.mean(test_signed ** 2))
        cal_p90 = np.percentile(cal_abs, 90)
        test_p90 = np.percentile(test_abs, 90)
        cal_p95 = np.percentile(cal_abs, 95)
        test_p95 = np.percentile(test_abs, 95)

        cal_vs_test_records.append({
            "lead_time": L,
            "cal_mean_signed": round(float(cal_mean), 4),
            "test_mean_signed": round(float(test_mean), 4),
            "ratio_mean_signed": round(float(test_mean / cal_mean) if cal_mean != 0 else np.nan, 4),
            "cal_median_signed": round(float(np.median(cal_signed)), 4),
            "test_median_signed": round(float(np.median(test_signed)), 4),
            "cal_std": round(float(cal_std), 4),
            "test_std": round(float(test_std), 4),
            "ratio_std": round(float(test_std / cal_std), 4),
            "cal_MAE": round(float(cal_mae), 4),
            "test_MAE": round(float(test_mae), 4),
            "ratio_MAE": round(float(test_mae / cal_mae), 4),
            "cal_RMSE": round(float(cal_rmse), 4),
            "test_RMSE": round(float(test_rmse), 4),
            "ratio_RMSE": round(float(test_rmse / cal_rmse), 4),
            "cal_P90_abs": round(float(cal_p90), 4),
            "test_P90_abs": round(float(test_p90), 4),
            "ratio_P90": round(float(test_p90 / cal_p90), 4),
            "cal_P95_abs": round(float(cal_p95), 4),
            "test_P95_abs": round(float(test_p95), 4),
            "ratio_P95": round(float(test_p95 / cal_p95), 4),
            "cal_count": int(len(cal_signed)),
            "test_count": int(len(test_signed))
        })

        # --- Task 7 & 8: Quantile vs Error & Violations ---
        actual_errors = eval_df_L["abs_error"].values
        preds = eval_df_L["pred_lead_demand"].values
        actuals = eval_df_L["actual_lead_demand"].values
        n_total = len(actuals)

        for cov in nominal_coverages:
            q_col = f"q_{int(cov * 100)}"
            q_vals = eval_df_L[q_col].values

            # Quantile stats
            under_est_rate = np.mean(q_vals < actual_errors)
            quantile_vs_actual_records.append({
                "lead_time": L,
                "nominal_coverage": cov,
                "mean_q": round(float(np.mean(q_vals)), 4),
                "median_q": round(float(np.median(q_vals)), 4),
                "P90_q": round(float(np.percentile(q_vals, 90)), 4),
                "P95_q": round(float(np.percentile(q_vals, 95)), 4),
                "mean_actual_error": round(float(np.mean(actual_errors)), 4),
                "median_actual_error": round(float(np.median(actual_errors)), 4),
                "P90_actual_error": round(float(np.percentile(actual_errors, 90)), 4),
                "P95_actual_error": round(float(np.percentile(actual_errors, 95)), 4),
                "underestimation_rate_percent": round(float(under_est_rate * 100.0), 4),
                "coverage_target_gap": round(float((under_est_rate - (1.0 - cov)) * 100.0), 4)
            })

            # Interval Violations
            lowers = np.maximum(0.0, preds - q_vals)
            uppers = preds + q_vals

            lower_viols = np.sum(actuals < lowers)
            upper_viols = np.sum(actuals > uppers)
            covered = np.sum((actuals >= lowers) & (actuals <= uppers))

            lower_rate = lower_viols / n_total * 100.0
            upper_rate = upper_viols / n_total * 100.0
            total_viol_rate = (lower_viols + upper_viols) / n_total * 100.0
            empirical_cov = covered / n_total * 100.0

            violation_records.append({
                "lead_time": L,
                "nominal_coverage": cov,
                "empirical_coverage_percent": round(float(empirical_cov), 4),
                "lower_violation_rate_percent": round(float(lower_rate), 4),
                "upper_violation_rate_percent": round(float(upper_rate), 4),
                "total_violation_rate_percent": round(float(total_viol_rate), 4),
                "lower_violation_count": int(lower_viols),
                "upper_violation_count": int(upper_viols),
                "total_observations": int(n_total)
            })

    # Save Task 4 Table & Plot
    cal_vs_test_df = pd.DataFrame(cal_vs_test_records)
    cal_vs_test_df.to_csv(os.path.join(output_dir, "calibration_vs_test_residuals.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    x = np.arange(len(lead_times))
    w = 0.35
    plt.bar(x - w/2, cal_vs_test_df["cal_MAE"], width=w, label="Calibration OOS (2013-16)", color="gray")
    plt.bar(x + w/2, cal_vs_test_df["test_MAE"], width=w, label="2017 Test Set", color="darkred")
    plt.xticks(x, [f"L={L}" for L in lead_times])
    plt.title("MAE: Calibration OOS vs. 2017 Test", pad=10)
    plt.xlabel("Lead Time L (Days)")
    plt.ylabel("Mean Absolute Error")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(x - w/2, cal_vs_test_df["cal_mean_signed"], width=w, label="Calibration OOS (2013-16)", color="gray")
    plt.bar(x + w/2, cal_vs_test_df["test_mean_signed"], width=w, label="2017 Test Set", color="darkred")
    plt.xticks(x, [f"L={L}" for L in lead_times])
    plt.title("Mean Signed Error (Bias): Calib vs. 2017", pad=10)
    plt.xlabel("Lead Time L (Days)")
    plt.ylabel("Mean Signed Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_vs_test_residual_scale.png"), dpi=300)
    plt.close()

    # Save Task 7 Table & Plot
    quantile_vs_actual_df = pd.DataFrame(quantile_vs_actual_records)
    quantile_vs_actual_df.to_csv(os.path.join(output_dir, "quantile_vs_actual_error.csv"), index=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=quantile_vs_actual_df, x="lead_time", y="underestimation_rate_percent", hue="nominal_coverage", palette="Reds")
    plt.title("Proportion of 2017 Days Where Conformal q < Actual Error", pad=12)
    plt.xlabel("Lead Time L (Days)")
    plt.ylabel("Underestimation Rate (%)")
    plt.axhline(20, color="gray", linestyle=":", label="Nominal 80% Non-Coverage (20%)")
    plt.axhline(10, color="gray", linestyle="--", label="Nominal 90% Non-Coverage (10%)")
    plt.axhline(5, color="gray", linestyle="-", label="Nominal 95% Non-Coverage (5%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantile_vs_actual_error.png"), dpi=300)
    plt.close()

    # Save Task 8 Table & Plot
    violation_df = pd.DataFrame(violation_records)
    violation_df.to_csv(os.path.join(output_dir, "interval_violation_breakdown.csv"), index=False)

    plt.figure(figsize=(10, 5))
    melted_viols = pd.melt(
        violation_df, 
        id_vars=["lead_time", "nominal_coverage"], 
        value_vars=["lower_violation_rate_percent", "upper_violation_rate_percent"],
        var_name="violation_type", 
        value_name="violation_rate"
    )
    melted_viols["violation_type"] = melted_viols["violation_type"].map({
        "lower_violation_rate_percent": "Lower Bound (Demand < Lower)",
        "upper_violation_rate_percent": "Upper Bound (Demand > Upper)"
    })
    
    sns.barplot(
        data=melted_viols[melted_viols["nominal_coverage"] == 0.90], 
        x="lead_time", 
        y="violation_rate", 
        hue="violation_type",
        palette=["skyblue", "salmon"]
    )
    plt.title("Breakdown of 90% Conformal Interval Violations (2017)", pad=12)
    plt.xlabel("Lead Time L (Days)")
    plt.ylabel("Violation Rate (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lower_vs_upper_violations.png"), dpi=300)
    plt.close()

    # =========================================================================
    # TASK 6: Residual Drift by Training Year (2013..2016)
    # =========================================================================
    logger.info("TASK 6: Computing historical residual drift across training years...")
    res_df["year"] = res_df["forecast_origin"].dt.year
    year_drift_records = []
    
    for yr in sorted(res_df["year"].unique()):
        yr_subset = res_df[res_df["year"] == yr]
        for L in lead_times:
            yr_L = yr_subset[yr_subset["lead_time"] == L]
            if len(yr_L) == 0:
                continue
            
            s_err = yr_L["signed_lead_error"].values
            a_err = yr_L["absolute_lead_error"].values
            
            year_drift_records.append({
                "year": yr,
                "lead_time": L,
                "mean_signed_error": round(float(np.mean(s_err)), 4),
                "median_signed_error": round(float(np.median(s_err)), 4),
                "MAE": round(float(np.mean(a_err)), 4),
                "RMSE": round(float(np.sqrt(np.mean(s_err ** 2))), 4),
                "median_absolute_error": round(float(np.median(a_err)), 4),
                "std_deviation": round(float(np.std(s_err)), 4),
                "P90_absolute_error": round(float(np.percentile(a_err, 90)), 4),
                "P95_absolute_error": round(float(np.percentile(a_err, 95)), 4),
                "residual_count": int(len(yr_L))
            })

    year_drift_df = pd.DataFrame(year_drift_records)
    year_drift_df.to_csv(os.path.join(output_dir, "residual_drift_by_year.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=year_drift_df, x="year", y="mean_signed_error", hue="lead_time", marker="o", palette="tab10")
    plt.title("Historical Mean Signed Error by Year (OOS)", pad=10)
    plt.xlabel("Calendar Year")
    plt.ylabel("Mean Signed Error")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    sns.lineplot(data=year_drift_df, x="year", y="MAE", hue="lead_time", marker="o", palette="tab10")
    plt.title("Historical MAE by Year (OOS)", pad=10)
    plt.xlabel("Calendar Year")
    plt.ylabel("Mean Absolute Error")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residual_drift_by_year.png"), dpi=300)
    plt.close()

    # =========================================================================
    # TASK 9 & 10: Demand Volume & Seasonality Diagnostics
    # =========================================================================
    logger.info("TASK 9 & 10: Generating volume and seasonality diagnostics...")
    vol_diag_records = []
    seas_diag_records = []

    for L in lead_times:
        eval_df_L = lead_time_eval_dfs[L].merge(profiles_df, on=["store", "item"])
        
        for cov in nominal_coverages:
            q_col = f"q_{int(cov * 100)}"
            eval_df_L["lower"] = np.maximum(0.0, eval_df_L["pred_lead_demand"] - eval_df_L[q_col])
            eval_df_L["upper"] = eval_df_L["pred_lead_demand"] + eval_df_L[q_col]
            eval_df_L["covered"] = (eval_df_L["actual_lead_demand"] >= eval_df_L["lower"]) & (eval_df_L["actual_lead_demand"] <= eval_df_L["upper"])
            eval_df_L["width"] = eval_df_L["upper"] - eval_df_L["lower"]
            
            # Volume breakdown
            for vol, grp in eval_df_L.groupby("volume_category"):
                mean_act = float(grp["actual_lead_demand"].mean())
                mean_prd = float(grp["pred_lead_demand"].mean())
                mean_bias = float((grp["actual_lead_demand"] - grp["pred_lead_demand"]).mean())
                bias_rat = mean_bias / mean_act if mean_act != 0 else 0.0
                
                vol_diag_records.append({
                    "lead_time": L,
                    "nominal_coverage": cov,
                    "volume_category": vol,
                    "mean_actual_demand": round(mean_act, 4),
                    "mean_prediction": round(mean_prd, 4),
                    "mean_signed_bias": round(mean_bias, 4),
                    "bias_ratio": round(bias_rat, 4),
                    "empirical_coverage": round(float(grp["covered"].mean()), 4),
                    "mean_interval_width": round(float(grp["width"].mean()), 4),
                    "mean_absolute_error": round(float(grp["abs_error"].mean()), 4)
                })
                
            # Seasonality breakdown
            for seas, grp in eval_df_L.groupby("seasonality_category"):
                mean_act = float(grp["actual_lead_demand"].mean())
                mean_prd = float(grp["pred_lead_demand"].mean())
                mean_bias = float((grp["actual_lead_demand"] - grp["pred_lead_demand"]).mean())
                bias_rat = mean_bias / mean_act if mean_act != 0 else 0.0

                seas_diag_records.append({
                    "lead_time": L,
                    "nominal_coverage": cov,
                    "seasonality_category": seas,
                    "mean_actual_demand": round(mean_act, 4),
                    "mean_prediction": round(mean_prd, 4),
                    "mean_signed_bias": round(mean_bias, 4),
                    "bias_ratio": round(bias_rat, 4),
                    "empirical_coverage": round(float(grp["covered"].mean()), 4),
                    "mean_interval_width": round(float(grp["width"].mean()), 4),
                    "mean_absolute_error": round(float(grp["abs_error"].mean()), 4)
                })

    vol_diag_df = pd.DataFrame(vol_diag_records)
    vol_diag_df.to_csv(os.path.join(output_dir, "coverage_by_volume_diagnostic.csv"), index=False)

    seas_diag_df = pd.DataFrame(seas_diag_records)
    seas_diag_df.to_csv(os.path.join(output_dir, "coverage_by_seasonality_diagnostic.csv"), index=False)

    # =========================================================================
    # TASK 11: Series-Level Coverage Distribution (Across 500 Series)
    # =========================================================================
    logger.info("TASK 11: Computing series-level coverage distribution...")
    per_series_path = "journal_extension/results/uncertainty/final_2017/per_series_uncertainty_results.csv"
    if not os.path.exists(per_series_path):
        raise FileNotFoundError(f"Per-series results not found at {per_series_path}")
    per_series_df = pd.read_csv(per_series_path)
    
    series_dist_records = []
    for (L, cov), grp in per_series_df.groupby(["lead_time", "nominal_coverage"]):
        cov_vals = grp["empirical_coverage"].values
        
        p_within_5 = np.mean(np.abs(cov_vals - cov) <= 0.05) * 100.0
        p_within_10 = np.mean(np.abs(cov_vals - cov) <= 0.10) * 100.0
        p_below_50 = np.mean(cov_vals < 0.50) * 100.0
        
        series_dist_records.append({
            "lead_time": L,
            "nominal_coverage": cov,
            "mean_coverage": round(float(np.mean(cov_vals)), 4),
            "median_coverage": round(float(np.median(cov_vals)), 4),
            "P25_coverage": round(float(np.percentile(cov_vals, 25)), 4),
            "P75_coverage": round(float(np.percentile(cov_vals, 75)), 4),
            "min_coverage": round(float(np.min(cov_vals)), 4),
            "max_coverage": round(float(np.max(cov_vals)), 4),
            "percent_series_within_5pct": round(float(p_within_5), 2),
            "percent_series_within_10pct": round(float(p_within_10), 2),
            "percent_series_below_50pct": round(float(p_below_50), 2),
            "total_series": len(cov_vals)
        })

    series_dist_df = pd.DataFrame(series_dist_records)
    series_dist_df.to_csv(os.path.join(output_dir, "series_coverage_distribution.csv"), index=False)

    logger.info("EXP-014: Diagnostic analysis completed successfully and all outputs serialized.")


if __name__ == "__main__":
    main()
