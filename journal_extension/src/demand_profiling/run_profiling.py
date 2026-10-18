"""
run_profiling.py
----------------
Runner script to perform demand profiling on the training partition
and save profiles, summaries, and diagnostic plots.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from demand_profile import DemandProfiler
from regime_detection import DemandRegimeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Demand Profiling Runner...")
    
    # 1. Load dataset
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed training set not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)
    
    # 2. Extract Training partition (first 80% chronologically) to prevent data leakage
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    logger.info(f"Loaded training partition with {len(train_df)} rows spanning {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}")
    
    # 3. Compute Profiles and Regimes
    profiler = DemandProfiler()
    detector = DemandRegimeDetector()
    
    raw_profiles = profiler.build_profiles(train_df)
    final_profiles = detector.detect_regimes(raw_profiles)
    
    # 4. Save results
    output_dir = "journal_extension/results/model_selection"
    os.makedirs(output_dir, exist_ok=True)
    
    profiles_file = os.path.join(output_dir, "demand_profiles.csv")
    final_profiles.to_csv(profiles_file, index=False)
    logger.info(f"Saved store-item profiles to {profiles_file}")
    
    # 5. Generate Summary Stats
    total_series = len(final_profiles)
    intermittent_count = (final_profiles["zero_demand_proportion"] > 0.0).sum()
    mean_cv = final_profiles["cv_sales"].mean()
    mean_cv_sq = final_profiles["cv_sq_sales"].mean()
    
    regime_counts = final_profiles["primary_routing_regime"].value_counts().to_dict()
    sbc_counts = final_profiles["sbc_category"].value_counts().to_dict()
    
    summary_data = {
        "Metric": [
            "total_store_item_combinations",
            "series_with_zero_demand",
            "mean_coefficient_of_variation",
            "mean_cv_squared",
            "sbc_smooth_count",
            "sbc_erratic_count",
            "sbc_intermittent_count",
            "sbc_lumpy_count",
            "regime_standard_ml_count",
            "regime_seasonal_expert_count",
            "regime_trend_expert_count",
            "regime_robust_baseline_count"
        ],
        "Value": [
            total_series,
            intermittent_count,
            round(mean_cv, 4),
            round(mean_cv_sq, 4),
            sbc_counts.get("Smooth", 0),
            sbc_counts.get("Erratic", 0),
            sbc_counts.get("Intermittent", 0),
            sbc_counts.get("Lumpy", 0),
            regime_counts.get("Standard_ML_Expert", 0),
            regime_counts.get("Seasonal_Expert", 0),
            regime_counts.get("Trend_Expert", 0),
            regime_counts.get("Robust_Baseline_Expert", 0)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "demand_profile_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved profile summary to {summary_file}")
    
    # 6. Visual Diagnostics
    logger.info("Generating diagnostic plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot A: Distribution of CV
    plt.figure(figsize=(8, 5))
    sns.histplot(final_profiles["cv_sales"], kde=True, color="#3b82f6")
    plt.title("Distribution of Coefficient of Variation (CV) Across Series", pad=15)
    plt.xlabel("Coefficient of Variation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_distribution.png"), dpi=300)
    plt.close()
    
    # Plot B: Trend vs Volatility Scatter
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=final_profiles, 
        x="cv_sq_sales", 
        y="trend_strength_r2", 
        hue="sbc_category", 
        style="primary_routing_regime",
        palette="Set1",
        alpha=0.8
    )
    plt.title("Series Trend Strength vs Volatility (CV²)", pad=15)
    plt.xlabel("Volatility (CV²)")
    plt.ylabel("Trend Strength (R²)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_vs_volatility_scatter.png"), dpi=300)
    plt.close()
    
    # Plot C: Primary Routing Regime Counts
    plt.figure(figsize=(8, 5))
    order = ["Standard_ML_Expert", "Seasonal_Expert", "Trend_Expert", "Robust_Baseline_Expert"]
    sns.countplot(
        data=final_profiles, 
        x="primary_routing_regime", 
        hue="primary_routing_regime", 
        order=[o for o in order if o in regime_counts],
        palette="viridis", 
        legend=False
    )
    plt.title("Dynamic Routing Regime Counts", pad=15)
    plt.xlabel("Primary Routing Expert")
    plt.ylabel("Number of Series")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regime_counts_bar.png"), dpi=300)
    plt.close()
    
    logger.info("Demand Profiling completed successfully.")


if __name__ == "__main__":
    main()
