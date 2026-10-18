# Adaptive Routing Negative Result

This document records the official negative result of the adaptive model routing experiment for the retail sales forecasting pipeline. 

---

## 1. Objective

To train a Decision Tree classifier meta-model to route each store-item demand series to the optimal forecasting expert (XGBoost vs. Random Forest) based on static demand profiles.

---

## 2. Experimental Setup

* **Feasible Experts**: XGBoost (optimized conference model) and Random Forest (tuned bagging regressor). ARIMA was excluded as a routing candidate since it won 0/500 series in baseline testing.
* **Input Features**: `mean_sales`, `cv_sales`, `trend_strength_r2`, and `lag365_autocorr` calculated strictly on the training partition.
* **Label Generation**: Out-of-Fold (OOF) labels generated via 5-fold TimeSeriesSplit on the training partition.
* **Router Model**: Decision Tree Classifier (`max_depth = 3`, `class_weight = 'balanced'`).

---

## 3. Validation Results

The table below reports the average validation metrics across the 500 store-item combinations on the sub-validation split:

* Always XGBoost MAE: **6.1748**
* Always Random Forest MAE: **6.2585**
* Oracle Best (Upper Bound) MAE: **6.1667**
* **Adaptive Router MAE**: **6.1843** (representing a performance degradation of **-0.15%** relative to the Always-XGBoost baseline).

---

## 4. Final Test Results

The table below reports the out-of-sample performance on the untouched chronological test partition:

* Always XGBoost MAE: **6.1748**
* Always Random Forest MAE: **6.2584**
* **Adaptive Router MAE**: **6.1843** (representing a performance degradation of **-0.15%** relative to Always-XGBoost).

---

## 5. Routing Accuracy

* **Routing Accuracy**: **70.4%** (the router correctly identified the optimal validation model for 352 out of 500 series).

---

## 6. Balanced Accuracy

* **Balanced Accuracy**: **60.31%** (accounting for class imbalance in winner distributions).

---

## 7. Macro F1

* **Macro F1 Score**: **0.5688**

---

## 8. Oracle Regret

* **Oracle Regret**: **0.0176 units** (the average absolute error difference between the router's selected model and the oracle best model per series).

---

## 9. Winner Instability

* **Stable Series**: $45.8\%$ (229 out of 500 combinations) had a stable winner across all 5 cross-validation folds.
* **Unstable Series**: $54.2\%$ (271 out of 500 combinations) had winning labels that switched between folds.
* *Analysis*: The high label volatility indicates that the performance difference between XGBoost and Random Forest is marginal and noisy for a majority of store-item pairs.

---

## 10. Statistical Comparison

A paired Wilcoxon signed-rank test comparing the per-series MAEs of Always-XGBoost against the Adaptive Router yielded:
* **Test Statistic**: $2648.0$
* **p-value**: **$3.0426 \times 10^{-9}$**
* *Result*: The difference is statistically significant, verifying that the Adaptive Router consistently underperforms the global XGBoost baseline.

---

## 11. Limitations

1. **Class Imbalance**: XGBoost is highly dominant ($84.2\%$ wins), meaning classification errors carry a high penalty that offsets the marginal gains made on Random Forest-optimal series.
2. **Homogeneity**: The dataset lacks intermittency and has highly uniform volatility, making a single global model like XGBoost extremely robust and difficult to outperform.

---

## 12. Scientific Conclusion

"Adaptive model routing between XGBoost and Random Forest did not improve forecasting accuracy on the current retail dataset and therefore is not adopted as the primary journal methodology."
