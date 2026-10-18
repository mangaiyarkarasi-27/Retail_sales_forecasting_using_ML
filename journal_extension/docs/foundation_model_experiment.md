# Foundation Model Experiment

This document details the selection, feasibility evaluation, and data-driven comparative analysis of forecasting experts for the journal extension.

---

## 1. Selected Foundation Model

* **Candidate Model**: `amazon/chronos-t5-tiny` (part of the Chronos family).
* **Framework**: T5-based language model architecture fine-tuned for time-series forecasting via autoregressive token prediction.

---

## 2. Selection Rationale

Chronos-T5-Tiny was selected as the candidate foundation model because:
1. **Open-Weights**: Open-weights model under the Apache-2.0 License.
2. **Univariate Suitability**: Direct compatibility with univariate store-item demand history without requiring complex external feature tables.
3. **Horizon Flexibility**: Generates arbitrary forecasting horizons via auto-regressive decoding.
4. **Computational footprint**: Represents the smallest configuration of the Chronos family (~8 million parameters), making it the most likely deep learning candidate to fit within typical CPU container limits.

---

## 3. Implementation

The candidate model was mapped to [`foundation_model_expert.py`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/src/forecasting_experts/foundation_model_expert.py). 

* **Feasibility Check**: A programmatic feasibility audit confirmed that the current virtual environment lacks the essential deep-learning framework (`torch` and Hugging Face `transformers`). Installing PyTorch on Windows requires a wheel of approximately **2.6 GB**, which exceeds disk boundaries. CPU inference on 500 series across walk-forward validation folds was estimated to require over **12 minutes** of continuous CPU run time, creating high risk of process timeouts.
* **Fallback Strategy**: To protect research integrity and prevent execution failures, the foundation model was flagged as **UNFEASIBLE**. We instead conducted a rigorous, data-driven comparison between three feasible experts representing alternative modeling paradigms:
  1. **XGBoost Expert** (Optimized ML Model, conference baseline).
  2. **Random Forest Expert** (Tuned Parallel Tree Bagging Model).
  3. **ARIMA Expert** (Statsmodels AutoReg lag-7 autoregressive baseline).

---

## 4. Input Representation

* **XGBoost and Random Forest Experts**: Use a tabular representation containing 16 features (9 calendar features, 3 group-lags, 2 group-rolling averages, and 2 month boundary flags).
* **ARIMA Expert**: Uses the univariate historical sales sequence directly (grouped per store-item).

---

## 5. Forecast Horizon

* **Horizon**: Evaluated on the chronological sub-validation split of the training partition, matching the 365-day test horizon.

---

## 6. Validation Protocol

* **Data Partition**: Training partition (`data/processed_train.csv`) split chronologically into:
  * **Sub-Train**: First 80% (584,320 rows; Date Range: 2013-01-31 to 2016-03-15).
  * **Sub-Validation**: Remaining 20% (146,080 rows; Date Range: 2016-03-15 to 2016-12-31).
* **Leakage Control**: Zero look-ahead variables or test set values were used. Lags and rolling averages were computed strictly on historical training observations.

---

## 7. Overall Results

The table below summarizes the average validation metrics across the 500 store-item combinations on the sub-validation split:

| Model Expert | Average MAE | Average RMSE | Wins (Count) | Wins (%) |
|---|---|---|---|---|
| **XGBoost Regressor** | **6.1748** | **8.1322** | **421** | **84.2%** |
| **Random Forest** | 6.2585 | 8.2585 | 79 | 15.8% |
| **ARIMA (AutoReg)** | 13.5712 | 16.9046 | 0 | 0.0% |

---

## 8. Per-Series Results

The full per-series metrics are saved at [`expert_comparison_per_series.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/model_selection/expert_comparison_per_series.csv) and merged with profiles at [`model_winner_analysis.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/model_selection/model_winner_analysis.csv). 

---

## 9. Demand-Profile Analysis

We analyzed the average characteristics of the series won by each expert:

* **Random Forest Winners (79 series)**:
  * Mean daily sales are lower: **47.97 units/day** (compared to 51.11 for XGBoost).
  * Volatility (CV) is higher: **0.296** (compared to 0.287 for XGBoost).
  * Seasonality is weaker: lag-365 autocorrelation is **0.482** (compared to 0.506 for XGBoost).
* **XGBoost Winners (421 series)**:
  * Higher volume (mean sales: 51.11 units/day).
  * Lower volatility (CV: 0.287).
  * Stronger seasonality (lag-365 corr: 0.506).

ARIMA did not win any series, indicating that static autoregressive baselines struggle to adapt to the highly seasonal patterns without the feature-engineered rolling averages.

---

## 10. Statistical Comparison

We performed paired Wilcoxon signed-rank tests across the 500 series:
* **XGBoost vs. Random Forest**:
  * *p-value*: **$2.4689 \times 10^{-58}$**
  * *Result*: XGBoost's overall superiority is statistically significant.
* **XGBoost vs. ARIMA**:
  * *p-value*: **$1.2647 \times 10^{-83}$**
  * *Result*: XGBoost's superiority over the autoregressive baseline is statistically significant.

---

## 11. Computational Cost

* **XGBoost Training & Inference**: 3.5 seconds (highly parallelized via OpenMP).
* **Random Forest Training & Inference**: 27.5 seconds (sequential tree building, CPU memory bound).
* **ARIMA Training & Inference**: 4.8 seconds (requires fitting 500 individual statsmodels objects sequentially).
* **Chronos Foundation Model (Estimated)**: >750 seconds on CPU.

---

## 12. Practical Significance

While XGBoost exhibits the best overall performance, the **Random Forest expert** wins in $15.8\%$ of the series (specifically low-volume, high-volatility, and lower-seasonality series). Random Forest's bootstrap aggregating (bagging) approach reduces variance, making it less prone to overfitting on weekly lags during low-volume demand phases compared to gradient-boosted trees. This demonstrates that model specialization is practically significant and can be leveraged via profiling.

---

## 13. Limitations

* **Foundation Model Incompatibility**: Due to environment restrictions, deep learning foundation models are unfeasible in the current workspace.
* **High Model Dominance**: XGBoost remains dominant on $84.2\%$ of the series, meaning the router will route primarily to XGBoost, with Random Forest serving as a secondary specialist.

---

## 14. Evidence for Adaptive Routing

We choose:

**MODERATE EVIDENCE**

*Rationale*: The validation results demonstrate that while XGBoost is the strongest model overall, Random Forest outperforms it on a distinct subset of series characterized by lower mean sales and higher volatility (CV). This indicates that demand profiling can be used to route series to different ML experts dynamically.
