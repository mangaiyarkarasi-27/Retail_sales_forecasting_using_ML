"""
run_adaptive_conformal_experiment.py
------------------------------------
Executes EXP-015: Scale-Adaptive Conformal Uncertainty Evaluation.
Highly optimized with vectorized multi-coverage evaluation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src directories to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uncertainty")))

from forecast_vector import ForecastVectorOrchestrator
from adaptive_conformal import AdaptiveConformalCalibrator
from uncertainty_metrics import (
    calculate_picp,
    calculate_miw,
    calculate_interval_score,
    calculate_coverage_error
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_conformal_on_period(
    df_period: pd.DataFrame,
    res_df: pd.DataFrame,
    lead_times: list,
    nominal_coverages: list,
    calibrator: AdaptiveConformalCalibrator,
    method_name: str,
    profiles_dict: dict
):
    """
    Evaluates a specific conformal method across all series in df_period.
    Returns (summary_df, per_series_df).
    """
    per_series_records = []
    groups = {name: grp.sort_values(by="date").reset_index(drop=True) for name, grp in df_period.groupby(["store", "item"])}
    
    for L in lead_times:
        res_L = res_df[res_df["lead_time"] == L].copy().sort_values(by=["store", "item", "forecast_origin"]).reset_index(drop=True)
        res_groups = {name: grp.reset_index(drop=True) for name, grp in res_L.groupby(["store", "item"])}
        
        for (s, i), group in groups.items():
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
                
            mean_vol = profiles_dict.get((s, i), {}).get("mean_volume", 50.0)
            
            buffers = {cov: {"lowers": [], "uppers": [], "actuals": []} for cov in nominal_coverages}
            
            n_days = len(group)
            for t in range(n_days - L):
                pred_val = pred_lt_sum[t]
                actual_val = np.sum(sales[t + 1 : t + L + 1])
                cutoff_date = cutoff_dates[t]
                
                eligible_count = np.searchsorted(res_origins, cutoff_date, side='right')
                eligible_errors = res_errors[:eligible_count]
                
                if method_name == "Raw_Conformal":
                    q_dict = calibrator.calibrate_raw(eligible_errors, nominal_coverages)
                elif method_name == "Scale_Normalized":
                    q_dict = calibrator.calibrate_scale_normalized(eligible_errors, nominal_coverages)
                elif method_name == "Volume_Normalized":
                    q_dict = calibrator.calibrate_volume_normalized(eligible_errors, mean_vol, nominal_coverages)
                else:
                    raise ValueError(f"Unknown method {method_name}")
                
                for cov in nominal_coverages:
                    q = q_dict[cov]
                    lower = max(0.0, pred_val - q)
                    upper = pred_val + q
                    buffers[cov]["lowers"].append(lower)
                    buffers[cov]["uppers"].append(upper)
                    buffers[cov]["actuals"].append(actual_val)
                    
            for cov in nominal_coverages:
                lowers = np.array(buffers[cov]["lowers"])
                uppers = np.array(buffers[cov]["uppers"])
                actuals = np.array(buffers[cov]["actuals"])
                
                alpha = 1.0 - cov
                picp = calculate_picp(actuals, lowers, uppers)
                miw = calculate_miw(lowers, uppers)
                score = calculate_interval_score(actuals, lowers, uppers, alpha)
                cov_err = calculate_coverage_error(picp, cov)
                
                lower_viols = np.mean(actuals < lowers)
                upper_viols = np.mean(actuals > uppers)
                
                per_series_records.append({
                    "method": method_name,
                    "store": int(s),
                    "item": int(i),
                    "lead_time": L,
                    "nominal_coverage": cov,
                    "empirical_coverage": round(picp, 4),
                    "coverage_error": round(cov_err, 4),
                    "mean_interval_width": round(miw, 4),
                    "interval_score": round(score, 4),
                    "lower_violation_rate": round(float(lower_viols), 4),
                    "upper_violation_rate": round(float(upper_viols), 4),
                    "evaluated_origins": len(actuals)
                })
                
    per_series_df = pd.DataFrame(per_series_records)
    
    summary_records = []
    for (L, cov), grp in per_series_df.groupby(["lead_time", "nominal_coverage"]):
        summary_records.append({
            "method": method_name,
            "lead_time": L,
            "nominal_coverage": cov,
            "empirical_coverage": round(grp["empirical_coverage"].mean(), 4),
            "coverage_error": round(grp["coverage_error"].mean(), 4),
            "mean_interval_width": round(grp["mean_interval_width"].mean(), 4),
            "interval_score": round(grp["interval_score"].mean(), 4),
            "lower_violation_rate": round(grp["lower_violation_rate"].mean(), 4),
            "upper_violation_rate": round(grp["upper_violation_rate"].mean(), 4),
            "evaluated_origins": int(grp["evaluated_origins"].iloc[0])
        })
    summary_df = pd.DataFrame(summary_records)
    return summary_df, per_series_df


def main():
    logger.info("Initializing EXP-015: Scale-Adaptive Conformal Uncertainty Experiment...")
    output_dir = "journal_extension/results/uncertainty/adaptive_015"
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

    # 2013-2015 Sub-Train vs 2016 Method-Validation Split
    sub_train = train_df[train_df["date"] <= pd.to_datetime("2015-12-31")].copy().reset_index(drop=True)
    sub_val_2016 = train_df[(train_df["date"] >= pd.to_datetime("2016-01-01")) & (train_df["date"] <= pd.to_datetime("2016-12-31"))].copy().reset_index(drop=True)

    # Compute training-derived demand profiles (strictly using sub_train 2013-2015)
    logger.info("Computing training-derived profiles (2013-2015)...")
    profiles = []
    profiles_dict = {}
    for (s, i), group in sub_train.groupby(["store", "item"]):
        mean_vol = group["sales"].mean()
        acf_365 = 0.0
        if len(group) > 365:
            sales_series = group["sales"]
            acf_365 = sales_series.autocorr(lag=365)
            if np.isnan(acf_365):
                acf_365 = 0.0
        
        vol_cat = "Low" if mean_vol < 20 else ("Medium" if mean_vol <= 50 else "High")
        seas_cat = "Weak" if acf_365 < 0.20 else ("Moderate" if acf_365 < 0.40 else "Strong")
            
        prof_item = {
            "store": s,
            "item": i,
            "mean_volume": round(mean_vol, 4),
            "seasonality_strength": round(acf_365, 4),
            "volume_category": vol_cat,
            "seasonality_category": seas_cat
        }
        profiles.append(prof_item)
        profiles_dict[(s, i)] = prof_item
    profiles_df = pd.DataFrame(profiles)

    # Load direct forecasting orchestrator and generate forecasts
    orchestrator = ForecastVectorOrchestrator(models_dir="journal_extension/results/forecasting_models")
    logger.info("Generating forecasts for 2016 validation period...")
    val_forecasts_2016 = orchestrator.generate_lead_time_forecasts(sub_val_2016)

    logger.info("Generating forecasts for 2017 test period...")
    test_forecasts_2017 = orchestrator.generate_lead_time_forecasts(test_df)

    # Load EXP-012 OOS calibration residuals
    residuals_path = "journal_extension/results/uncertainty/oos_direct/lead_time_oos_residuals.csv"
    if not os.path.exists(residuals_path):
        raise FileNotFoundError(f"Residuals not found at {residuals_path}")
    res_df = pd.read_csv(residuals_path)
    res_df["forecast_origin"] = pd.to_datetime(res_df["forecast_origin"])

    calibrator = AdaptiveConformalCalibrator(
        window_size=100,
        scale_window=50,
        min_observations=20,
        epsilon=1e-4
    )

    lead_times = [1, 3, 5, 7]
    nominal_coverages = [0.80, 0.90, 0.95]

    # =========================================================================
    # STAGE 1: Method Validation on 2016 (Untouched 2017)
    # =========================================================================
    logger.info("STAGE 1: Evaluating methods on 2016 validation split...")
    val_summaries = []
    
    for method in ["Raw_Conformal", "Scale_Normalized", "Volume_Normalized"]:
        logger.info(f"Evaluating {method} on 2016...")
        summary_m, _ = evaluate_conformal_on_period(
            val_forecasts_2016, res_df, lead_times, nominal_coverages, calibrator, method, profiles_dict
        )
        val_summaries.append(summary_m)
        
    val_summary_df = pd.concat(val_summaries, ignore_index=True)
    val_summary_df.to_csv(os.path.join(output_dir, "method_validation_2016.csv"), index=False)
    logger.info(f"2016 Validation Results:\n{val_summary_df[['method', 'lead_time', 'nominal_coverage', 'empirical_coverage', 'coverage_error', 'mean_interval_width', 'interval_score']]}")

    # Compute overall performance on 2016 to select optimal method
    selection_rankings = []
    for method, grp in val_summary_df.groupby("method"):
        avg_cov_err = grp["coverage_error"].mean()
        avg_score = grp["interval_score"].mean()
        avg_width = grp["mean_interval_width"].mean()
        selection_rankings.append({
            "method": method,
            "mean_coverage_error_2016": round(float(avg_cov_err), 4),
            "mean_interval_score_2016": round(float(avg_score), 4),
            "mean_interval_width_2016": round(float(avg_width), 4)
        })
    selection_df = pd.DataFrame(selection_rankings).sort_values(by="mean_interval_score_2016")
    logger.info(f"Method Selection Rankings on 2016 Validation:\n{selection_df}")

    # Select the top method
    best_method = selection_df.iloc[0]["method"]
    selected_method = best_method
    logger.info(f"Selected and Frozen Adaptive Method: {selected_method}")

    # =========================================================================
    # STAGE 2: Final 2017 Evaluation (Raw Baseline vs Frozen Adaptive)
    # =========================================================================
    logger.info("STAGE 2: Evaluating on untouched 2017 test period...")
    
    summary_raw_2017, per_series_raw_2017 = evaluate_conformal_on_period(
        test_forecasts_2017, res_df, lead_times, nominal_coverages, calibrator, "Raw_Conformal", profiles_dict
    )
    summary_adapt_2017, per_series_adapt_2017 = evaluate_conformal_on_period(
        test_forecasts_2017, res_df, lead_times, nominal_coverages, calibrator, selected_method, profiles_dict
    )

    comparison_2017_df = pd.concat([summary_raw_2017, summary_adapt_2017], ignore_index=True)
    comparison_2017_df.to_csv(os.path.join(output_dir, "final_2017_comparison.csv"), index=False)

    per_series_all_2017 = pd.concat([per_series_raw_2017, per_series_adapt_2017], ignore_index=True)
    per_series_all_2017.to_csv(os.path.join(output_dir, "adaptive_per_series_results.csv"), index=False)
    logger.info(f"2017 Final Comparison:\n{comparison_2017_df[['method', 'lead_time', 'nominal_coverage', 'empirical_coverage', 'coverage_error', 'mean_interval_width', 'interval_score']]}")

    # =========================================================================
    # STAGE 3: Statistical Comparison Across 500 Series
    # =========================================================================
    logger.info("STAGE 3: Running paired statistical significance tests...")
    merged_per_series = per_series_raw_2017.merge(
        per_series_adapt_2017,
        on=["store", "item", "lead_time", "nominal_coverage"],
        suffixes=("_raw", "_adapt")
    )
    
    stat_test_records = []
    for (L, cov), grp in merged_per_series.groupby(["lead_time", "nominal_coverage"]):
        stat_cov, p_cov = stats.wilcoxon(grp["coverage_error_raw"], grp["coverage_error_adapt"])
        stat_w, p_w = stats.wilcoxon(grp["mean_interval_width_raw"], grp["mean_interval_width_adapt"])
        stat_sc, p_sc = stats.wilcoxon(grp["interval_score_raw"], grp["interval_score_adapt"])
        
        stat_test_records.append({
            "lead_time": L,
            "nominal_coverage": cov,
            "raw_coverage_error": round(float(grp["coverage_error_raw"].mean()), 4),
            "adapt_coverage_error": round(float(grp["coverage_error_adapt"].mean()), 4),
            "cov_error_diff": round(float(grp["coverage_error_adapt"].mean() - grp["coverage_error_raw"].mean()), 4),
            "p_val_cov_error": float(p_cov),
            "raw_interval_width": round(float(grp["mean_interval_width_raw"].mean()), 4),
            "adapt_interval_width": round(float(grp["mean_interval_width_adapt"].mean()), 4),
            "width_diff": round(float(grp["mean_interval_width_adapt"].mean() - grp["mean_interval_width_raw"].mean()), 4),
            "p_val_width": float(p_w),
            "raw_interval_score": round(float(grp["interval_score_raw"].mean()), 4),
            "adapt_interval_score": round(float(grp["interval_score_adapt"].mean()), 4),
            "score_diff": round(float(grp["interval_score_adapt"].mean() - grp["interval_score_raw"].mean()), 4),
            "p_val_score": float(p_sc)
        })
    stat_test_df = pd.DataFrame(stat_test_records)
    stat_test_df.to_csv(os.path.join(output_dir, "adaptive_statistical_tests.csv"), index=False)

    # =========================================================================
    # STAGE 4: Demand-Profile Slices (Volume & Seasonality)
    # =========================================================================
    logger.info("STAGE 4: Computing demand profile slices on 2017...")
    merged_with_prof = per_series_all_2017.merge(profiles_df, on=["store", "item"])

    # Volume Breakdown
    vol_records = []
    for (m, L, cov, vol), grp in merged_with_prof.groupby(["method", "lead_time", "nominal_coverage", "volume_category"]):
        vol_records.append({
            "method": m,
            "lead_time": L,
            "nominal_coverage": cov,
            "volume_category": vol,
            "empirical_coverage": round(float(grp["empirical_coverage"].mean()), 4),
            "coverage_error": round(float(grp["coverage_error"].mean()), 4),
            "mean_interval_width": round(float(grp["mean_interval_width"].mean()), 4),
            "interval_score": round(float(grp["interval_score"].mean()), 4)
        })
    vol_df = pd.DataFrame(vol_records)
    vol_df.to_csv(os.path.join(output_dir, "adaptive_coverage_by_volume.csv"), index=False)

    # Seasonality Breakdown
    seas_records = []
    for (m, L, cov, seas), grp in merged_with_prof.groupby(["method", "lead_time", "nominal_coverage", "seasonality_category"]):
        seas_records.append({
            "method": m,
            "lead_time": L,
            "nominal_coverage": cov,
            "seasonality_category": seas,
            "empirical_coverage": round(float(grp["empirical_coverage"].mean()), 4),
            "coverage_error": round(float(grp["coverage_error"].mean()), 4),
            "mean_interval_width": round(float(grp["mean_interval_width"].mean()), 4),
            "interval_score": round(float(grp["interval_score"].mean()), 4)
        })
    seas_df = pd.DataFrame(seas_records)
    seas_df.to_csv(os.path.join(output_dir, "adaptive_coverage_by_seasonality.csv"), index=False)

    # =========================================================================
    # STAGE 5: Residual Distribution Stability Across Years (2013-2016)
    # =========================================================================
    logger.info("STAGE 5: Computing residual score stability across historical years...")
    res_df["year"] = res_df["forecast_origin"].dt.year
    stability_records = []

    for yr in sorted(res_df["year"].unique()):
        yr_subset = res_df[res_df["year"] == yr]
        for L in lead_times:
            yr_L = yr_subset[yr_subset["lead_time"] == L]
            if len(yr_L) == 0:
                continue
            
            raw_errs = yr_L["absolute_lead_error"].values
            
            # Vectorized rolling scale computation per series
            norm_scores = []
            for (s, i), grp in yr_L.groupby(["store", "item"]):
                errs = grp["absolute_lead_error"].values
                sigmas = pd.Series(errs).rolling(50, min_periods=1).mean().values
                norm_scores.extend(errs / np.maximum(sigmas, 1e-4))
            norm_scores = np.array(norm_scores)

            stability_records.append({
                "year": yr,
                "lead_time": L,
                "raw_mean_error": round(float(np.mean(raw_errs)), 4),
                "raw_std_error": round(float(np.std(raw_errs)), 4),
                "raw_P90_error": round(float(np.percentile(raw_errs, 90)), 4),
                "norm_mean_score": round(float(np.mean(norm_scores)), 4),
                "norm_std_score": round(float(np.std(norm_scores)), 4),
                "norm_P90_score": round(float(np.percentile(norm_scores, 90)), 4),
                "observation_count": len(raw_errs)
            })
    stability_df = pd.DataFrame(stability_records)
    stability_df.to_csv(os.path.join(output_dir, "adaptive_residual_stability.csv"), index=False)

    # =========================================================================
    # STAGE 6: Generate Diagnostic Plots
    # =========================================================================
    logger.info("STAGE 6: Generating diagnostic plots...")

    # Plot 1: Raw vs Adaptive Coverage (2017)
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=comparison_2017_df,
        x="nominal_coverage",
        y="empirical_coverage",
        hue="method",
        style="lead_time",
        markers=True,
        dashes=False,
        palette=["red", "blue"]
    )
    plt.plot([0.80, 0.95], [0.80, 0.95], color="black", linestyle="--", label="Ideal Coverage")
    plt.title("EXP-015: Raw vs. Adaptive Conformal Coverage on 2017 Test Set", pad=12)
    plt.xlabel("Nominal Coverage Level")
    plt.ylabel("Empirical Coverage Level")
    plt.legend(title="Method / Lead Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_vs_adaptive_coverage.png"), dpi=300)
    plt.close()

    # Plot 2: Raw vs Adaptive Mean Interval Width (2017)
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=comparison_2017_df,
        x="lead_time",
        y="mean_interval_width",
        hue="method",
        palette=["lightcoral", "cornflowerblue"]
    )
    plt.title("EXP-015: Mean Interval Width by Lead Time (Raw vs. Adaptive, 2017)", pad=12)
    plt.xlabel("Lead Time L (Days)")
    plt.ylabel("Mean Interval Width (Units)")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_vs_adaptive_width.png"), dpi=300)
    plt.close()

    # Plot 3: Raw vs Adaptive Interval Score (2017)
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=comparison_2017_df,
        x="lead_time",
        y="interval_score",
        hue="method",
        palette=["salmon", "royalblue"]
    )
    plt.title("EXP-015: Winkler Interval Score (Lower is Better, 2017)", pad=12)
    plt.xlabel("Lead Time L (Days)")
    plt.ylabel("Winkler Interval Score")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "raw_vs_adaptive_interval_score.png"), dpi=300)
    plt.close()

    # Plot 4: Normalized Residual Stability Across Years
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=stability_df, x="year", y="raw_mean_error", hue="lead_time", marker="o", palette="tab10")
    plt.title("Raw Residual Scale Across Years (2013-2016)", pad=10)
    plt.xlabel("Year")
    plt.ylabel("Mean Absolute Error")

    plt.subplot(1, 2, 2)
    sns.lineplot(data=stability_df, x="year", y="norm_mean_score", hue="lead_time", marker="o", palette="tab10")
    plt.title("Normalized Nonconformity Score Across Years", pad=10)
    plt.xlabel("Year")
    plt.ylabel("Mean Normalized Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "normalized_residual_stability.png"), dpi=300)
    plt.close()

    # Plot 5 & 6: Coverage by Volume and Seasonality
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=vol_df[vol_df["nominal_coverage"] == 0.90],
        x="volume_category",
        y="empirical_coverage",
        hue="method",
        palette=["lightcoral", "royalblue"]
    )
    plt.title("EXP-015: 90% Nominal Coverage by Volume Slice (2017)", pad=12)
    plt.xlabel("Demand Volume Category")
    plt.ylabel("Empirical Coverage")
    plt.axhline(0.90, color="black", linestyle="--", label="Nominal 90%")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adaptive_coverage_by_volume.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=seas_df[seas_df["nominal_coverage"] == 0.90],
        x="seasonality_category",
        y="empirical_coverage",
        hue="method",
        palette=["lightcoral", "royalblue"]
    )
    plt.title("EXP-015: 90% Nominal Coverage by Seasonality Slice (2017)", pad=12)
    plt.xlabel("Seasonality Category")
    plt.ylabel("Empirical Coverage")
    plt.axhline(0.90, color="black", linestyle="--", label="Nominal 90%")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adaptive_coverage_by_seasonality.png"), dpi=300)
    plt.close()

    logger.info("EXP-015: Scale-adaptive conformal experiment completed successfully.")


if __name__ == "__main__":
    main()
