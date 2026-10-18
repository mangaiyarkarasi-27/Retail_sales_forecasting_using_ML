# Adaptive Routing Experiment

This document details the objective, out-of-fold training, performance validation, interpretability, and out-of-sample final test results of the learned adaptive model router.

---

## 1. Objective

To train, validate, and test a meta-classifier that dynamically routes store-item demand series to the forecasting expert expected to minimize out-of-sample forecasting residuals.

---

## 2. Experts

The two forecasting experts selected for routing are:
* **Expert A (XGBoost)**: Tuned gradient-boosted trees regressor (optimized via TimeSeriesSplit).
* **Expert B (Random Forest)**: Tuned parallel bagging forest regressor.
* *Note*: ARIMA was excluded from the routing experts since it won $0/500$ series on validation data.

---

## 3. Router

We implemented a **Decision Tree Classifier** with `max_depth = 3` and `class_weight = 'balanced'` as the routing engine. This model guarantees high interpretability and prevents overfitting on small datasets.

---

## 4. Router-Safe Features

The inputs to the router are:
* `mean_sales` (demand volume)
* `cv_sales` (historical volatility)
* `trend_strength_r2` (linear trend coefficient)
* `lag365_autocorr` (seasonality strength)

*Leakage Control*: All features are computed strictly on the training partition. No test target sales or test residuals are used.

---

## 5. OOF Training Procedure

To generate training labels without look-ahead bias:
1. The training partition was split into 5 folds using `TimeSeriesSplit`.
2. XGBoost and Random Forest were trained on the sub-train of each fold and validated on the sub-validation fold.
3. For each of the 500 series, validation MAEs were calculated.
4. The binary target was set to `1` (XGBoost wins) or `0` (Random Forest wins).
5. These out-of-fold records were saved to [`oof_router_training_data.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/model_selection/oof_router_training_data.csv). The router was trained on this merged dataset.

---

## 6. Validation Results

The table below summarizes the performance of the router and individual experts on the chronological sub-validation split of the training partition:

| Strategy | Mean MAE | Mean RMSE | Wins (Count) | Wins (%) |
|---|---|---|---|---|
| **Always XGBoost** | **6.1748** | **8.1322** | 421 | 84.2% |
| **Always Random Forest** | 6.2585 | 8.2585 | 79 | 15.8% |
| **Oracle Best (Upper Bound)** | 6.1667 | 8.1186 | 500 | 100.0% |
| **Adaptive Router (Proposed)** | 6.1843 | 8.1450 | - | - |

---

## 7. Routing Performance

We evaluated the classification accuracy of the router on validation labels:
* **Routing Accuracy**: **70.4%** (the router correctly chooses the optimal validation model for 352 out of 500 series).
* **Balanced Accuracy**: **60.31%**
* **Macro F1**: **0.5688**
* **Oracle Regret**: **0.0176 units** (representing the average forecasting MAE loss incurred by routing errors).

---

## 8. Forecasting Performance

The Adaptive Router achieved a mean forecasting MAE of **6.1843** on the validation set. This represents a minor performance degradation of **-0.15%** relative to the Always-XGBoost baseline (6.1748).

---

## 9. Statistical Comparison

We performed a paired Wilcoxon signed-rank test comparing the per-series MAEs of Always-XGBoost against the Adaptive Router:
* **Wilcoxon Statistic**: $2648.0$
* **p-value**: **$3.0426 \times 10^{-9}$**
* **Sample Size**: 500
* *Result*: The difference is statistically significant, confirming that the Adaptive Router yields a small but consistent degradation in predictive accuracy compared to the global XGBoost model.

---

## 10. Router Interpretability

The Gini importances of the decision tree router are saved at [`router_feature_importance.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/explainability/router_feature_importance.csv):
* **Mean Sales**: **47.59%**
* **CV Sales**: **42.18%**
* **Trend Strength**: **8.11%**
* **Lag-365 Autocorrelation**: **2.12%**

*Interpretation*: The router's decisions are dominated by volume scale (`mean_sales`) and demand volatility (`cv_sales`). The model learns to route series with lower sales volumes and higher volatility to the Random Forest expert, and high-volume, low-volatility series to XGBoost.

---

## 11. Ablation Results

We ran an input feature ablation study on the validation split:

| Features Set | Routing Accuracy | Forecasting MAE |
|---|---|---|
| **A. Seasonality-Only** | 72.6% | 6.1819 |
| **B. Volume + Seasonality** | 73.8% | 6.1819 |
| **C. Volume + Volatility + Seasonality** | **75.4%** | **6.1818** |
| **D. Full Demand Profile (All 4 features)** | 70.4% | 6.1843 |

*Result*: The ablation demonstrates that including the weak trend feature ($R^2_{\text{trend}}$) causes overfitting, decreasing routing accuracy to 70.4%. Feature set C (excluding trend) yields the best routing accuracy ($75.4\%$).

---

## 12. Robustness

We analyzed assignment stability across validation folds:
* **Switching Rate**: $45.8\%$ (229 out of 500) of the series had a completely stable winner across all 5 folds. The remaining $54.2\%$ had winning labels that switched between folds.
* *Result*: This indicates a high level of label instability, showing that for many series, the performance difference between XGBoost and Random Forest is marginal and noisy, making routing decisions unstable.

---

## 13. Final Test Results

The table below reports the out-of-sample performance on the untouched chronological test partition:

| Strategy | Test MAE | Test RMSE | MAE Improvement (%) |
|---|---|---|---|
| **Always XGBoost (Conference)** | **6.1748** | **8.0365** | 0.0% |
| **Always Random Forest** | 6.2584 | 8.1515 | -1.36% |
| **Adaptive Router** | 6.1843 | 8.0410 | -0.15% |

*Result*: On the out-of-sample test set, the Adaptive Router yields a Test MAE of 6.1843, representing a minor degradation of **-0.15%** compared to the Always-XGBoost baseline.

---

## 14. Limitations

1. **Class Dominance**: Since XGBoost is highly dominant ($84.2\%$ wins), any routing error (misclassifying a series where XGBoost is better as RF) carries a penalty that outweighs the minor gains made on RF-optimal series.
2. **Label Volatility**: For over half of the series, the winning model fluctuates across folds, creating high routing target noise.

---

## 15. Scientific Conclusion

We choose:

**NOT SUPPORTED**

*Rationale*: The experimental evidence demonstrates that adaptive model routing does **not** improve forecasting accuracy on this dataset compared to a strong global baseline (XGBoost). Due to the dataset's high homogeneity (all series are continuous, smooth, and seasonal), a single global XGBoost model is extremely robust, and routing errors degrade performance by $-0.15\%$ on the out-of-sample test split.
