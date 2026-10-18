"""
run_model_comparison.py
------------------------
Performs a data-driven validation comparison between XGBoost, Random Forest, 
and ARIMA experts on the chronological sub-validation split of the training partition.
Analyzes results against demand profiles to verify model specialization.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.ar_model import AutoReg

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Data-Driven Forecasting Expert Comparison...")
    
    # 1. Load preprocessed training dataset
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed training set not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # Add lag and rolling features per store-item group (identical to conference pipeline)
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
    
    # 2. Chronological Split of the Training Partition (80% sub-train, 20% sub-val)
    df = df.sort_values(by="date").reset_index(drop=True)
    train_split_idx = int(len(df) * 0.8)
    
    sub_train = df.iloc[:train_split_idx].copy().reset_index(drop=True)
    sub_val = df.iloc[train_split_idx:].copy().reset_index(drop=True)
    
    logger.info(f"Sub-Train size: {len(sub_train)} rows ({sub_train['date'].min().strftime('%Y-%m-%d')} to {sub_train['date'].max().strftime('%Y-%m-%d')})")
    logger.info(f"Sub-Val size: {len(sub_val)} rows ({sub_val['date'].min().strftime('%Y-%m-%d')} to {sub_val['date'].max().strftime('%Y-%m-%d')})")
    
    # 3. Train ML Experts
    logger.info("Training XGBoost Expert...")
    xgb = XGBRegressor(
        n_estimators=50, learning_rate=0.1, max_depth=8, 
        subsample=1.0, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    xgb.fit(sub_train[features], sub_train[target])
    sub_val["pred_xgb"] = xgb.predict(sub_val[features])
    
    logger.info("Training Random Forest Expert...")
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=15, min_samples_split=2, min_samples_leaf=1,
        random_state=42, n_jobs=-1
    )
    rf.fit(sub_train[features], sub_train[target])
    sub_val["pred_rf"] = rf.predict(sub_val[features])
    
    # 4. Train ARIMA (AutoReg) Expert per store-item
    logger.info("Running ARIMA baseline forecasts...")
    sub_val["pred_arima"] = 0.0
    store_item_pairs = sub_train[["store", "item"]].drop_duplicates().values
    
    for s, i in store_item_pairs:
        tr_sub = sub_train[(sub_train["store"] == s) & (sub_train["item"] == i)]
        val_sub = sub_val[(sub_val["store"] == s) & (sub_val["item"] == i)]
        if len(tr_sub) > 0 and len(val_sub) > 0:
            res = AutoReg(tr_sub["sales"].values, lags=7).fit()
            fc = res.predict(start=len(tr_sub), end=len(tr_sub) + len(val_sub) - 1)
            sub_val.loc[val_sub.index, "pred_arima"] = fc
            
    # 5. Evaluate Per-Series Metrics
    logger.info("Evaluating validation performance per store-item combination...")
    comparison_records = []
    
    for s, i in store_item_pairs:
        pair_val = sub_val[(sub_val["store"] == s) & (sub_val["item"] == i)]
        if len(pair_val) > 0:
            y_true = pair_val[target].values
            p_xgb = pair_val["pred_xgb"].values
            p_rf = pair_val["pred_rf"].values
            p_arima = pair_val["pred_arima"].values
            
            mae_xgb = float(np.mean(np.abs(y_true - p_xgb)))
            mae_rf = float(np.mean(np.abs(y_true - p_rf)))
            mae_arima = float(np.mean(np.abs(y_true - p_arima)))
            
            rmse_xgb = float(np.sqrt(np.mean((y_true - p_xgb)**2)))
            rmse_rf = float(np.sqrt(np.mean((y_true - p_rf)**2)))
            rmse_arima = float(np.sqrt(np.mean((y_true - p_arima)**2)))
            
            # Identify winning model among the three
            maes = {"XGBoost": mae_xgb, "Random_Forest": mae_rf, "ARIMA": mae_arima}
            best_model = min(maes, key=maes.get)
            sorted_maes = sorted(maes.values())
            margin = float(sorted_maes[1] - sorted_maes[0])
            
            comparison_records.append({
                "store": int(s),
                "item": int(i),
                "xgboost_mae": round(mae_xgb, 4),
                "random_forest_mae": round(mae_rf, 4),
                "arima_mae": round(mae_arima, 4),
                "xgboost_rmse": round(rmse_xgb, 4),
                "random_forest_rmse": round(rmse_rf, 4),
                "arima_rmse": round(rmse_arima, 4),
                "best_model": best_model,
                "margin": round(margin, 4)
            })
            
    comparison_df = pd.DataFrame(comparison_records)
    
    # 6. Merge with Demand Profiles for Analysis
    profiles_path = "journal_extension/results/model_selection/demand_profiles.csv"
    if not os.path.exists(profiles_path):
        raise FileNotFoundError(f"Demand profiles file not found at {profiles_path}")
    profiles_df = pd.read_csv(profiles_path)
    
    merged_df = comparison_df.merge(
        profiles_df[["store", "item", "mean_sales", "cv_sales", "trend_strength_r2", "lag365_autocorr"]],
        on=["store", "item"],
        how="left"
    )
    
    # Save winner analysis CSV
    output_dir = "journal_extension/results/model_selection"
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_df.to_csv(os.path.join(output_dir, "expert_comparison_per_series.csv"), index=False)
    merged_df.to_csv(os.path.join(output_dir, "model_winner_analysis.csv"), index=False)
    logger.info("Saved expert comparison and winner analysis CSV files.")
    
    # 7. Model Specialization Analysis by Seasonality Bins (Quantiles)
    logger.info("Analyzing model specialization by seasonality strength...")
    merged_df["seasonality_bin"] = pd.qcut(merged_df["lag365_autocorr"], q=3, labels=["Low", "Medium", "High"])
    
    summary_records = []
    for bin_name, group in merged_df.groupby("seasonality_bin", observed=True):
        win_counts = group["best_model"].value_counts().to_dict()
        summary_records.append({
            "Seasonality_Strength": bin_name,
            "Series_Count": len(group),
            "Mean_XGBoost_MAE": round(group["xgboost_mae"].mean(), 4),
            "Mean_RF_MAE": round(group["random_forest_mae"].mean(), 4),
            "Mean_ARIMA_MAE": round(group["arima_mae"].mean(), 4),
            "XGBoost_Wins": win_counts.get("XGBoost", 0),
            "Random_Forest_Wins": win_counts.get("Random_Forest", 0),
            "ARIMA_Wins": win_counts.get("ARIMA", 0)
        })
        
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(output_dir, "model_specialization_summary.csv"), index=False)
    logger.info("Saved model specialization summary CSV.")
    
    # 8. Paired Statistical Wilcoxon Signed-Rank Test
    logger.info("Running Wilcoxon signed-rank tests...")
    # XGBoost vs RF
    stat_rf, p_rf = wilcoxon(merged_df["xgboost_mae"], merged_df["random_forest_mae"])
    # XGBoost vs ARIMA
    stat_arima, p_arima = wilcoxon(merged_df["xgboost_mae"], merged_df["arima_mae"])
    
    logger.info(f"Wilcoxon: XGBoost vs RF -> p-val: {p_rf:.6e}")
    logger.info(f"Wilcoxon: XGBoost vs ARIMA -> p-val: {p_arima:.6e}")
    
    # 9. Generate Research Figures
    logger.info("Generating expert comparison research plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot A: MAE Distributions
    plt.figure(figsize=(9, 5))
    melted = merged_df.melt(
        id_vars=["store", "item"], 
        value_vars=["xgboost_mae", "random_forest_mae", "arima_mae"],
        var_name="Model", value_name="MAE"
    )
    melted["Model"] = melted["Model"].map({
        "xgboost_mae": "XGBoost Expert", 
        "random_forest_mae": "Random Forest Expert", 
        "arima_mae": "ARIMA Expert"
    })
    sns.boxplot(data=melted, x="Model", y="MAE", hue="Model", palette="Set2", legend=False)
    plt.title("Validation Mean Absolute Error (MAE) Distribution Across 500 Series", pad=15)
    plt.ylabel("MAE (Units)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_mae_boxplots.png"), dpi=300)
    plt.close()
    
    # Plot B: Winner Distribution by Seasonality
    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=merged_df, 
        x="seasonality_bin", 
        hue="best_model", 
        palette="viridis"
    )
    plt.title("Winning Forecasting Expert by Seasonality Strength Bins", pad=15)
    plt.xlabel("Seasonality Strength Bin (Lag-365 Autocorrelation)")
    plt.ylabel("Number of Series Won")
    plt.legend(title="Winning Expert")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_winners_by_seasonality.png"), dpi=300)
    plt.close()
    
    logger.info("Expert model comparison experiment completed successfully.")


if __name__ == "__main__":
    main()
