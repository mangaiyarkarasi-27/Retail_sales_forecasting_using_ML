"""
run_model_routing.py
--------------------
Performs out-of-fold routing label generation via 5-fold TimeSeriesSplit,
trains the ModelRouter classifier, conducts input ablation and robustness studies,
and evaluates performance on the untouched chronological test partition.
"""

import os
import logging
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Add src directories to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "adaptive_selection")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "forecasting_experts")))

from model_router import ModelRouter
from selection_features import ProfileFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Adaptive Model Router Experiment...")
    
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
    
    # 2. Chronological Split (80% training partition, 20% untouched test partition)
    df = df.sort_values(by="date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    # 3. Load Demand Profiles metadata (calculated strictly from train partition)
    extractor = ProfileFeatureExtractor()
    profiles_df = extractor.get_features_dataframe()
    
    # 4. Out-of-Fold (OOF) Label Generation using 5-Fold TimeSeriesSplit on Train Partition
    logger.info("Executing 5-Fold TimeSeriesSplit to generate Out-of-Fold routing labels...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    oof_records = []
    
    # Store-item combinations
    store_item_pairs = train_df[["store", "item"]].drop_duplicates().values
    
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(train_df)):
        logger.info(f"OOF Fold {fold + 1}/5: Training experts...")
        X_tr = train_df.iloc[tr_idx][features]
        y_tr = train_df.iloc[tr_idx][target]
        X_val = train_df.iloc[val_idx][features]
        y_val = train_df.iloc[val_idx][target]
        
        # Train XGBoost
        xgb = XGBRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=8, 
            subsample=1.0, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        val_preds_xgb = xgb.predict(X_val)
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=15, min_samples_split=2, min_samples_leaf=1,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        val_preds_rf = rf.predict(X_val)
        
        # Attach predictions to validation subset
        val_subset = train_df.iloc[val_idx].copy()
        val_subset["pred_xgb"] = val_preds_xgb
        val_subset["pred_rf"] = val_preds_rf
        
        # Compute errors per store-item pair for this fold
        for s, i in store_item_pairs:
            pair_val = val_subset[(val_subset["store"] == s) & (val_subset["item"] == i)]
            if len(pair_val) > 0:
                y_true = pair_val[target].values
                p_xgb = pair_val["pred_xgb"].values
                p_rf = pair_val["pred_rf"].values
                
                mae_xgb = float(np.mean(np.abs(y_true - p_xgb)))
                mae_rf = float(np.mean(np.abs(y_true - p_rf)))
                
                # Winner: 1 = XGBoost, 0 = Random Forest
                winner = 1 if mae_xgb <= mae_rf else 0
                
                oof_records.append({
                    "store": int(s),
                    "item": int(i),
                    "fold": fold + 1,
                    "xgboost_mae": round(mae_xgb, 4),
                    "random_forest_mae": round(mae_rf, 4),
                    "mae_diff": round(mae_xgb - mae_rf, 4),
                    "winner": winner
                })
                
    oof_df = pd.DataFrame(oof_records)
    output_dir = "journal_extension/results/model_selection"
    os.makedirs(output_dir, exist_ok=True)
    oof_df.to_csv(os.path.join(output_dir, "oof_router_training_data.csv"), index=False)
    logger.info("Saved Out-of-Fold training data.")
    
    # 5. Train the Meta-Router on OOF Data
    # Merge OOF data with static profiles
    oof_merged = oof_df.merge(profiles_df, on=["store", "item"], how="left")
    
    router = ModelRouter()
    # Fit the router using OOF labels
    router.fit(oof_merged, oof_merged["winner"])
    router.save_router("results/models/model_router.pkl")
    
    # Save Feature Importance
    importances = router.get_feature_importances()
    imp_df = pd.DataFrame({
        "Feature": list(importances.keys()),
        "Importance": list(importances.values())
    }).sort_values(by="Importance", ascending=False)
    os.makedirs("journal_extension/results/explainability", exist_ok=True)
    imp_df.to_csv("journal_extension/results/explainability/router_feature_importance.csv", index=False)
    logger.info("Saved router feature importances.")
    
    # 6. Evaluate Router Performance on the sub-validation split (Experiment 2 results)
    logger.info("Evaluating router validation performance...")
    # Load validation comparison results from Experiment 2
    comp_file = os.path.join(output_dir, "model_winner_analysis.csv")
    if not os.path.exists(comp_file):
        raise FileNotFoundError(f"Model winner analysis file not found at {comp_file}")
    val_comp_df = pd.read_csv(comp_file)
    
    # Predict routing on validation set
    val_comp_df["router_best"] = router.predict_best_expert(val_comp_df)
    val_comp_df["oracle_best"] = val_comp_df["best_model"]
    
    # Check correctness
    val_comp_df["routing_correct"] = (val_comp_df["router_best"] == val_comp_df["oracle_best"]).astype(int)
    
    # Map model choice to selected MAE
    selected_mae = []
    selected_rmse = []
    regret = []
    for _, row in val_comp_df.iterrows():
        choice = row["router_best"]
        oracle = row["oracle_best"]
        
        m_err = row["xgboost_mae"] if choice == "XGBoost" else row["random_forest_mae"]
        r_err = row["xgboost_rmse"] if choice == "XGBoost" else row["random_forest_rmse"]
        o_err = row["xgboost_mae"] if oracle == "XGBoost" else row["random_forest_mae"]
        
        selected_mae.append(m_err)
        selected_rmse.append(r_err)
        regret.append(m_err - o_err)
        
    val_comp_df["selected_mae"] = selected_mae
    val_comp_df["selected_rmse"] = selected_rmse
    val_comp_df["regret"] = regret
    
    # Save per-series routing results
    val_comp_df.to_csv(os.path.join(output_dir, "adaptive_routing_per_series.csv"), index=False)
    
    # Calculate Routing Metrics
    # True labels: 1 for XGBoost, 0 for RF
    y_true_route = (val_comp_df["oracle_best"] == "XGBoost").astype(int)
    y_pred_route = (val_comp_df["router_best"] == "XGBoost").astype(int)
    
    route_acc = accuracy_score(y_true_route, y_pred_route)
    route_bal = balanced_accuracy_score(y_true_route, y_pred_route)
    route_f1 = f1_score(y_true_route, y_pred_route, average="macro")
    
    avg_regret = np.mean(regret)
    mean_sel_mae = np.mean(selected_mae)
    mean_sel_rmse = np.mean(selected_rmse)
    
    # Baseline comparison (Always XGBoost)
    mean_xgb_mae = val_comp_df["xgboost_mae"].mean()
    mean_xgb_rmse = val_comp_df["xgboost_rmse"].mean()
    mae_improvement = ((mean_xgb_mae - mean_sel_mae) / mean_xgb_mae) * 100
    
    # Save Summary CSV
    summary_data = {
        "Metric": [
            "routed_to_xgboost", "routed_to_rf", "routing_accuracy", "balanced_accuracy",
            "macro_f1", "average_regret", "average_selected_model_mae", 
            "average_selected_model_rmse", "percentage_mae_improvement"
        ],
        "Value": [
            (val_comp_df["router_best"] == "XGBoost").sum(),
            (val_comp_df["router_best"] == "Random_Forest").sum(),
            round(route_acc, 4),
            round(route_bal, 4),
            round(route_f1, 4),
            round(avg_regret, 4),
            round(mean_sel_mae, 4),
            round(mean_sel_rmse, 4),
            round(mae_improvement, 4)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "adaptive_routing_summary.csv"), index=False)
    logger.info("Saved adaptive routing summary.")
    
    # 7. Wilcoxon Paired Statistical Comparison (Always-XGBoost vs Adaptive Router)
    stat_val, p_val = wilcoxon(val_comp_df["xgboost_mae"], val_comp_df["selected_mae"])
    logger.info(f"Wilcoxon: Always-XGBoost vs Adaptive Router -> p-val: {p_val:.6e}")
    
    # 8. Input Ablation Study
    logger.info("Running router input ablation study...")
    ablation_results = []
    ablation_sets = {
        "A_Seasonality_Only": ["lag365_autocorr"],
        "B_Volume_Seasonality": ["mean_sales", "lag365_autocorr"],
        "C_Volume_Volatility_Seasonality": ["mean_sales", "cv_sales", "lag365_autocorr"],
        "D_Full_Demand_Profile": ["mean_sales", "cv_sales", "trend_strength_r2", "lag365_autocorr"]
    }
    
    for set_name, cols in ablation_sets.items():
        abl_clf = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight="balanced")
        abl_clf.fit(oof_merged[cols], oof_merged["winner"])
        
        # Predict on validation comparison
        preds_val = abl_clf.predict(val_comp_df[cols])
        y_pred_abl = ["XGBoost" if p == 1 else "Random_Forest" for p in preds_val]
        
        # Calculate selected MAE
        sel_maes = []
        for idx, row in val_comp_df.iterrows():
            choice = y_pred_abl[idx]
            sel_maes.append(row["xgboost_mae"] if choice == "XGBoost" else row["random_forest_mae"])
            
        mean_abl_mae = np.mean(sel_maes)
        acc = accuracy_score(val_comp_df["oracle_best"] == "XGBoost", preds_val)
        
        ablation_results.append({
            "Features_Set": set_name,
            "Accuracy": round(acc, 4),
            "Mean_Forecasting_MAE": round(mean_abl_mae, 4)
        })
        
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(os.path.join(output_dir, "router_ablation_results.csv"), index=False)
    logger.info("Saved ablation study results.")
    
    # 9. Evaluate Robustness Across Folds
    logger.info("Evaluating router robustness across validation folds...")
    # Calculate how often the winner switches between folds for the same store-item
    switches = oof_df.groupby(["store", "item"])["winner"].nunique()
    stable_count = (switches == 1).sum()
    logger.info(f"Number of series with completely stable winners across all folds: {stable_count}/500 ({stable_count/500*100:.1f}%)")
    
    # 10. Final Test Evaluation (Touch the Test partition ONCE)
    logger.info("Running FINAL evaluation on untouched test partition...")
    # Train final models on the COMPLETE training partition (898,000 rows minus test split)
    # Fit XGBoost on full train partition
    final_xgb = XGBRegressor(
        n_estimators=50, learning_rate=0.1, max_depth=8, 
        subsample=1.0, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    final_xgb.fit(train_df[features], train_df[target])
    test_preds_xgb = final_xgb.predict(test_df[features])
    
    # Fit Random Forest on full train partition
    final_rf = RandomForestRegressor(
        n_estimators=50, max_depth=15, min_samples_split=2, min_samples_leaf=1,
        random_state=42, n_jobs=-1
    )
    final_rf.fit(train_df[features], train_df[target])
    test_preds_rf = final_rf.predict(test_df[features])
    
    # Prediction arrays
    test_df["pred_xgb"] = test_preds_xgb
    test_df["pred_rf"] = test_preds_rf
    
    # Predict routing choices
    test_routing_df = profiles_df.copy()
    test_routing_df["router_best"] = router.predict_best_expert(test_routing_df)
    
    # Apply routing on test set
    test_df = test_df.merge(test_routing_df[["store", "item", "router_best"]], on=["store", "item"], how="left")
    test_df["selected_pred"] = np.where(
        test_df["router_best"] == "XGBoost", 
        test_df["pred_xgb"], 
        test_df["pred_rf"]
    )
    
    # Calculate aggregate test metrics
    test_mae_xgb = mean_xgb_mae = np.mean(np.abs(test_df[target] - test_df["pred_xgb"]))
    test_mae_rf = np.mean(np.abs(test_df[target] - test_df["pred_rf"]))
    test_mae_router = np.mean(np.abs(test_df[target] - test_df["selected_pred"]))
    
    test_rmse_xgb = np.sqrt(np.mean((test_df[target] - test_df["pred_xgb"])**2))
    test_rmse_rf = np.sqrt(np.mean((test_df[target] - test_df["pred_rf"])**2))
    test_rmse_router = np.sqrt(np.mean((test_df[target] - test_df["selected_pred"])**2))
    
    # Save final test results to a summary file
    test_summary = pd.DataFrame({
        "Model_Strategy": ["Always_XGBoost", "Always_Random_Forest", "Adaptive_Router"],
        "Test_MAE": [round(test_mae_xgb, 4), round(test_mae_rf, 4), round(test_mae_router, 4)],
        "Test_RMSE": [round(test_rmse_xgb, 4), round(test_rmse_rf, 4), round(test_rmse_router, 4)],
        "MAE_Improvement_pct": [
            0.0, 
            round(((test_mae_xgb - test_mae_rf)/test_mae_xgb)*100, 4),
            round(((test_mae_xgb - test_mae_router)/test_mae_xgb)*100, 4)
        ]
    })
    test_summary.to_csv(os.path.join(output_dir, "final_test_routing_results.csv"), index=False)
    logger.info(f"FINAL Test MAE: XGBoost={test_mae_xgb:.4f}, RF={test_mae_rf:.4f}, Router={test_mae_router:.4f}")
    
    # 11. Generate Research Figures
    logger.info("Generating routing research plots...")
    # Plot A: Feature Importance
    plt.figure(figsize=(8, 5))
    sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False)
    plt.title("Meta-Router Feature Importances (Decision Tree Classifier)", pad=15)
    plt.xlabel("Gini Importance")
    plt.ylabel("Demand Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "router_feature_importance.png"), dpi=300)
    plt.close()
    
    # Plot B: Strategy MAE Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=test_summary, 
        x="Model_Strategy", 
        y="Test_MAE", 
        hue="Model_Strategy",
        palette="coolwarm",
        legend=False
    )
    plt.title("Out-of-Sample Test MAE Comparison Across Strategies", pad=15)
    plt.ylabel("Test MAE (Units)")
    plt.xlabel("Strategy")
    plt.ylim(min(test_mae_xgb, test_mae_rf, test_mae_router) - 0.1, max(test_mae_xgb, test_mae_rf, test_mae_router) + 0.1)
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_mae_comparison.png"), dpi=300)
    plt.close()
    
    logger.info("Adaptive routing experiment completed successfully.")


if __name__ == "__main__":
    main()
