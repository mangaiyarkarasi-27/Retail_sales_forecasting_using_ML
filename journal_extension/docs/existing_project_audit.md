# Existing Project Audit

This document provides a technical audit of the **existing conference implementation** of the retail sales forecasting and decision support system (E-IDSS). The purpose is to map out current data pipelines, feature engineering steps, models, validation strategies, and decision scoring logic, ensuring the journal extension integrates with and reuses this codebase cleanly.

---

## 1. Project Structure

The existing conference codebase is structured as a modular pipeline in the root directory:

* `main.py`: Coordinates the execution of Modules 1 through 6.
* `src/preprocessing.py`: Implements raw data loading, schema validation, quality checks, cleaning, and date-based feature engineering.
* `src/train_model.py`: Implements advanced lag/rolling feature engineering, chronological train/test splitting, 5-fold TimeSeriesSplit walk-forward cross-validation, and final model retraining/serialization.
* `src/model_comparison.py`: Implements the 5-model benchmark under identical experimental conditions (Linear Regression, SVR, ARIMA, Random Forest, XGBoost).
* `src/shap_analysis.py`: Performs global and local feature importance explanations using TreeSHAP on the best XGBoost model.
* `src/recommendation_engine.py`: Evaluates forecast growth relative to historical averages to output operational stocking actions.
* `src/decision_intelligence.py`: Formulates multi-criteria Decision Scores and Risk Scores, priority rankings, opportunity scatter matrices, and ablation studies.
* `dashboard/app.py`: Streamlit application presenting the E-IDSS portal.
* `results/`: Directory housing generated plots, CSV metrics, and text reports from the pipeline execution.

---

## 2. Dataset

* **Exact Dataset Files Used**:
  * Raw files: `data/train.csv` and `data/test.csv`.
  * Preprocessed files: `data/processed_train.csv` and `data/processed_test.csv` (saved after cleaning and date feature extraction).
* **Dataset Dimensions**:
  * Raw train: 913,000 rows $\times$ 4 columns.
  * Processed train: 898,000 rows (after dropping 15,000 rows containing NaNs generated during rolling/lag feature calculations).
* **Columns**:
  * Target Variable: `sales` (daily sales quantities in units).
  * Date Column: `date` (datetime representation of transactions).
  * Identifiers: `store` (1 to 10) and `item` (1 to 50).
  * Additional Columns: `id` (test partition row ID), `year`, `month`, `day`, `weekday`, `weekofyear`, and `quarter`.

---

## 3. Preprocessing

* **Data Cleaning**:
  * Implemented in `src/preprocessing.py` -> `DataPreprocessor.clean_data()`.
  * Missing value handling: Imputed using forward-fill (`ffill()`) followed by backward-fill (`bfill()`) to preserve the time-series chronological order.
  * Duplicate handling: Checked and dropped using `drop_duplicates()`.
  * Outlier handling: Checked and reported, with sales values clipped if necessary.
* **Encoding & Scaling**:
  * Categorical encoding: Categorical identifiers `store` and `item` are kept as integer identifiers. The date components (`weekday`, `month`, etc.) are processed directly as numeric ordinal integers. No one-hot encoding is applied for the primary forecasting regressors.
  * Scaling: Implemented using `StandardScaler` in `src/model_comparison.py` solely for the Support Vector Regression (SVR) model. Tree-based regressors (XGBoost, Random Forest) and Linear Regression are trained on unscaled features to preserve scale interpretability.

---

## 4. Feature Engineering

* **Calendar Features**: Generated during preprocessing (`src/preprocessing.py`):
  * `year` (year integer)
  * `month` (month 1-12)
  * `day` (day of month 1-31)
  * `weekday` (day of week 0-6)
  * `weekofyear` (calendar week 1-53)
  * `quarter` (quarter 1-4)
  * `is_weekend` (binary flag, 1 for Saturday/Sunday, 0 otherwise)
  * `month_start` / `month_end` (binary indicators for month boundaries)
* **Lag Features**: Precomputed in `src/train_model.py` -> `SalesForecaster.add_advanced_features()`:
  * `sales_lag_7`: Sales exactly 7 days prior.
  * `sales_lag_14`: Sales exactly 14 days prior.
  * `sales_lag_30`: Sales exactly 30 days prior.
* **Rolling Features**: Precomputed in `src/train_model.py` -> `SalesForecaster.add_advanced_features()`:
  * `rolling_mean_7`: Moving average over the previous 7 days.
  * `rolling_mean_30`: Moving average over the previous 30 days.
* **Leakage Prevention**:
  * Lags are computed directly using `shift()`.
  * Rolling averages are computed on **shifted sales** (i.e. `x.shift(1).rolling()`) to ensure the current day's target sales value is never included in the sliding window calculation.
  * Rolling and lag calculations are grouped using `groupby(["store", "item"])` to execute independently for each store-item combination, preventing cross-series leakage.

---

## 5. Forecasting Models

### **XGBoost Regressor (Final Predictor)**
* **Source File**: `src/train_model.py` / `src/model_comparison.py`
* **Library**: `xgboost.XGBRegressor`
* **Tuned Hyperparameters**: `n_estimators = 50`, `learning_rate = 0.1`, `max_depth = 8`, `subsample = 1.0`, `colsample_bytree = 0.8`, `reg_alpha = 0.1`, `reg_lambda = 1.0`.
* **Training & Prediction**: Trained on the full chronological training partition (`X_train`, `y_train`); out-of-sample prediction via `.predict(X_test)`.
* **Pipeline Role**: Selected as the final best-performing model (Test MAE: 6.1748).

### **Random Forest Regressor**
* **Source File**: `src/train_model.py` / `src/model_comparison.py`
* **Library**: `sklearn.ensemble.RandomForestRegressor`
* **Tuned Hyperparameters**: `n_estimators = 50`, `max_depth = 15`, `min_samples_split = 2`, `min_samples_leaf = 1`.
* **Training & Prediction**: Trained on the full chronological training partition (`X_train`, `y_train`); prediction via `.predict(X_test)`.
* **Pipeline Role**: Active comparison model in hyperparameter tuning and model selection steps.

### **ARIMA**
* **Source File**: `src/model_comparison.py`
* **Library**: `statsmodels.tsa.ar_model.AutoReg`
* **Configuration**: Autoregressive lag model of order 7 (ARIMA(7,0,0)).
* **Training & Prediction**: Trained independently for each of the 500 store-item combinations on the training sales series; forecasts the test period out-of-sample.
* **Pipeline Role**: Used solely as a baseline comparison model.

### **Linear Regression & SVR**
* **Source File**: `src/model_comparison.py`
* **Library**: `sklearn.linear_model.LinearRegression` / `sklearn.svm.LinearSVR`
* **Tuned Hyperparameters**: SVR uses `loss='squared_epsilon_insensitive'`, `dual=False`, `C=1.0`, `max_iter=2000` (trained on `StandardScaler` scaled data).
* **Pipeline Role**: Baseline comparison models.

---

## 6. Cross-Validation

* **Validation Strategy**: `sklearn.model_selection.TimeSeriesSplit` is implemented to execute walk-forward validation.
* **Number of Folds**: 5 folds.
* **Splitting Procedure**: Splits the training partition (718,400 rows) chronologically:
  * Fold 1: Train 2013-01-31 to 2013-09-27 (119,735 rows) $\rightarrow$ Validate 2013-09-27 to 2014-05-24 (119,733 rows)
  * Fold 2: Train 2013-01-31 to 2014-05-24 (239,468 rows) $\rightarrow$ Validate 2014-05-24 to 2015-01-19 (119,733 rows)
  * Fold 3: Train 2013-01-31 to 2015-01-19 (359,201 rows) $\rightarrow$ Validate 2015-01-19 to 2015-09-15 (119,733 rows)
  * Fold 4: Train 2013-01-31 to 2015-09-15 (478,934 rows) $\rightarrow$ Validate 2015-09-15 to 2016-05-12 (119,733 rows)
  * Fold 5: Train 2013-01-31 to 2016-05-12 (598,667 rows) $\rightarrow$ Validate 2016-05-12 to 2017-01-06 (119,733 rows)
* **Leakage Prevention**: Since folds expand chronologically and lags/rolling metrics are computed on past values (`shift(1)`), future data never leaks into previous folds.
* **Optimization Metric**: Hyperparameters are optimized based on the **lowest average validation Mean Absolute Error (MAE)** across all 5 folds.

---

## 7. Final Model

* **Selected Model**: XGBoost Regressor.
* **Optimal Hyperparameters**:
  * `n_estimators`: 50
  * `learning_rate`: 0.1
  * `max_depth`: 8
  * `subsample`: 1.0
  * `colsample_bytree`: 0.8
  * `reg_alpha`: 0.1
  * `reg_lambda`: 1.0
* **Serialization Path**: Best model is saved as a pickle file at `results/models/best_model_v2.pkl`.
* **Metric Paths**:
  * Model Comparison CSV: `results/metrics/model_comparison.csv`
  * CV tuning metrics: `results/metrics/timeseries_cv_report.txt`

---

## 8. SHAP Explainability

* **Implementation**: `shap` package utilizing `shap.TreeExplainer` on the best serialized XGBoost model.
* **Target Sample**: TreeSHAP values are calculated on a representative sample of the test set (sampled from the latest operational date to capture current explainability dynamics).
* **Visualizations Generated**:
  * `results/plots/shap/shap_summary_plot.png` (Global beeswarm distribution)
  * `results/plots/shap/shap_bar_plot.png` (Global mean absolute SHAP feature importance)
  * `results/plots/shap/shap_dependence_plot.png` (Interaction plot for `sales_lag_7`)
  * `results/plots/shap/shap_waterfall_plot.png` (Local explanation for a single forecast sample)
* **Report Path**: `results/metrics/shap_report.txt` (describes top features and categorizes them into positive/negative directional contributors).

---

## 9. E-IDSS / Decision Support

* **Decision Scoring Engine**: Min-max normalizes and calculates a weighted Opportunity Decision Score ($S_D$) for each store-item combination:
  $$S_D = 0.30 \times \text{Growth} + 0.25 \times \text{Trend Strength} + 0.20 \times \text{SHAP Impact} + 0.15 \times \text{Seasonality} + 0.10 \times \text{Stability}$$
* **Risk Scoring Engine**: Combines four risk elements to calculate a Risk Score ($S_R$):
  $$S_R = 0.25 \times \text{Volatility} + 0.25 \times \text{Historical Variance} + 0.25 \times \text{Trend Instability} + 0.25 \times \text{Prediction Uncertainty}$$
  * Volatility: Forecast predicted standard deviation.
  * Historical Variance: Standard deviation of training sales.
  * Trend Instability: Variance of forecasting residuals.
  * Prediction Uncertainty: Out-of-sample RMSE per store-item pair.
* **Business Categories**: Categorizes each store-item pair into one of five slots:
  * `Growth Opportunity`: $S_D > 60$ and $S_R < 50$
  * `Stable Demand`: $S_D \in [40, 60]$ and $S_R < 50$
  * `Declining Demand`: $S_D < 40$ and $S_R < 50$
  * `Overstock Risk`: $S_R \ge 50$ and $S_D < 45$
  * `Strategic Monitoring`: $S_R \ge 50$ and $S_D \ge 45$
* **Output Paths**:
  * Priority Ranking CSV: `results/decision_intelligence/priority_ranking.csv`
  * Executive Decision Report: `results/decision_intelligence/executive_decision_report.txt`
  * Ablation Study: `results/decision_intelligence/ablation_study_report.txt`

---

## 10. Dashboard

The Streamlit application `dashboard/app.py` serves as the front-end interface, partitioned into 9 pages:

1. **Home**: Dashboard objectives, system architecture layout, and model KPI summary metrics (MAE, RMSE, $R^2$).
2. **Dataset Overview**: Interactive data table viewer showing dimensions and raw columns.
3. **EDA Analytics**: Displays correlation heatmap, sales distributions, and seasonal sales trends.
4. **Forecasting Performance**: Displays Table II model comparisons and actual vs predicted scatter plots.
5. **SHAP Explainability**: Displays beeswarm, bar, dependency, and waterfall visualizations.
6. **Recommendation Engine**: Displays count plots and operational action suggestions.
7. **Decision Intelligence**: Plots Decision and Risk score distributions, priority opportunity classification counts, and Decision Score vs Risk Score scatter plots.
8. **Executive Summary**: Renders the generated executive text report.
9. **Framework Comparison**: Displays comparative studies and ablation analysis text.

---

## 11. Existing Results

The existing results are serialized under `results/`:

* **Metrics**:
  * `results/metrics/model_comparison.csv`: Identical scale metrics for LR, SVR, ARIMA, RF, and XGBoost.
  * `results/metrics/timeseries_cv_report.txt`: Folds list, training periods, validation periods, optimal hyperparameters, and validation MAE/RMSE scores.
* **Plots**:
  * `results/plots/timeseries_split_visualization.png`: Gantt chart mapping expanding window folds.
  * `results/plots/model_comparison_bar.png`: Diagnostic bar charts comparing MAE and $R^2$ scores.
  * `results/plots/actual_vs_predicted_v2.png`: Scatter plot mapping prediction accuracy on test set.

---

## 12. Conference-to-Implementation Verification

The table below audits the status of various components against the implementation:

| Component | Status | File | Evidence / Function Name |
|---|---|---|---|
| **XGBoost Regressor** | IMPLEMENTED | `src/train_model.py` | `XGBRegressor` trained with tuned hyperparameters. |
| **ARIMA Baseline** | IMPLEMENTED | `src/model_comparison.py` | `AutoReg(lags=7)` fitted on store-item series. |
| **TimeSeriesSplit CV** | IMPLEMENTED | `src/train_model.py` | `TimeSeriesSplit(n_splits=5)` in `optimize_and_train_models()`. |
| **TreeSHAP XAI** | IMPLEMENTED | `src/shap_analysis.py` | `shap.TreeExplainer` on best XGBoost model. |
| **Decision Scoring ($S_D$)** | IMPLEMENTED | `src/decision_intelligence.py` | Multi-criteria scoring in `run_framework_pipeline()`. |
| **Risk Scoring ($S_R$)** | IMPLEMENTED | `src/decision_intelligence.py` | Combines Volatility, Std, Instability, and RMSE. |
| **Conformal Prediction** | NOT FOUND | — | No conformal interval implementation found. |
| **Inventory Policies** | DESCRIBED BUT NOT IMPLEMENTED | `src/recommendation_engine.py` | Stocking actions categorized (UP/DOWN/STEADY) but no safety stock quantities, ROPs, or order volumes calculated. |
| **Adaptive Multi-Model Router** | NOT FOUND | — | System runs a single XGBoost regressor as the final predictor rather than dynamically selecting experts. |

---

## Research Integrity Statement

We confirm that:
1. No existing conference source code, scripts, or pipelines were modified, deleted, or overwritten.
2. No existing experimental results, metrics, or logs were modified or fabricated.
3. No model training, tuning, or evaluation was executed during this audit.
4. No external code was copied or incorporated.
5. All findings in this document are strictly based on inspection of the existing project files.
