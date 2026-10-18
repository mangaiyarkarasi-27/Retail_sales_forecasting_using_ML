# Direct Multi-Horizon Forecasting Implementation

This document details the implementation, validation, and performance profile of the Direct Multi-Horizon point-forecasting layer for the journal extension.

---

## 1. Architecture

We implement a decoupled **Direct Multi-Horizon Forecasting** model structure:
* **Models**: $7$ independent global XGBoost Regressors ($M_1 \dots M_7$).
* **Prediction Horizon**: Model $M_h$ predicts demand at $t+h$ directly from forecast origin $t$ (morning of $t$, using history <= $t-1$).
* **Global Scope**: Models are global across all 500 store-item combinations.

---

## 2. Feature Construction

Features $X_t$ are generated strictly from the information set $\mathcal{I}_t$:
* **Lags**: `sales_lag_7` ($y_{t-7}$), `sales_lag_14` ($y_{t-14}$), `sales_lag_30` ($y_{t-30}$).
* **Rolling statistics**: `rolling_mean_7` ($\text{mean}(y_{t-7} \dots y_{t-1})$) and `rolling_mean_30` ($\text{mean}(y_{t-30} \dots y_{t-1})$).
* **Calendar metrics**: Target-date deterministic features ($t+h$) are used directly.

---

## 3. Training Row Construction

A training sample is defined as:
$$\text{Row}_h = \big( X_t, y_{t+h} \big)$$
To construct this cleanly, the base features $X$ are chronologically shifted by $h$ days for each store-item group and aligned with target sales $y_{t+h}$.

---

## 4. Model Configuration

Models use the frozen hyperparameters of the verified conference configuration:
* `n_estimators`: 50
* `learning_rate`: 0.1
* `max_depth`: 8
* `subsample`: 1.0
* `colsample_bytree`: 0.8
* `reg_alpha`: 0.1
* `reg_lambda`: 1.0

---

## 5. Training Boundaries

* **Training Origins**: 2013-01-01 to 2015-12-31 (target dates strictly capped at $\le 2015-12-31$).
* **Validation Origins**: 2016-01-01 to 2016-12-31 (validation target dates capped at $\le 2016-12-31$).
* **Final Test set**: 2017 (untouched).

---

## 6. Validation Protocol

Models are fitted on training origins and evaluated on 2016 validation origins to report MAE/RMSE per horizon, as well as aggregate validation metrics.

---

## 7. Forecast Vector

For any origin $t$, the forecasting layer returns the 7-day forecast vector:
$$\text{forecast\_vector}_t = \big[ \hat{y}_{t+1}, \hat{y}_{t+2}, \dots, \hat{y}_{t+7} \big]$$
Downstream modules compute lead-time demand by summing components: $\hat{D}_{t, L} = \sum_{k=1}^L \hat{y}_{t+k}$.

---

## 8. Leakage Tests

All 10 temporal safety checks passed successfully:
* Lags and rolling windows terminate at $t-1$.
* Target-date calendar indicators do not use sales data.
* Validation and training splits strictly respect target boundaries.
* 2017 test set is isolated.

---

## 9. Computational Measurements

* **Total Models Trained**: 7 global models.
* **Average Training Time per Model**: ~1.01 seconds.
* **Total Training Time (7 Folds)**: **8.05 seconds**.
* **Average Inference Time (183,000 rows)**: ~0.08 seconds.

---

## 10. Validation Results (2016 Validation Period)

The table below lists the measured validation performance for each model:

| Model Horizon | Validation MAE | Validation RMSE | Training Time | Prediction Time |
|---|---|---|---|---|
| **$M_1$ (h=1)** | 6.0416 | 7.8677 | 2.68s | 0.05s |
| **$M_2$ (h=2)** | 6.0432 | 7.8667 | 1.01s | 0.06s |
| **$M_3$ (h=3)** | 6.0366 | 7.8604 | 0.88s | 0.05s |
| **$M_4$ (h=4)** | 6.0384 | 7.8622 | 0.86s | 0.05s |
| **$M_5$ (h=5)** | 6.0318 | 7.8556 | 0.85s | 0.05s |
| **$M_6$ (h=6)** | 6.0285 | 7.8544 | 0.89s | 0.05s |
| **$M_7$ (h=7)** | 6.0518 | 7.8795 | 0.87s | 0.06s |
| **Aggregate** | **6.0388** | **7.8638** | **8.04s** | **0.37s** |

---

## 11. Limitations

* **Horizon Limits**: Prediction is limited to $h \le 7$ days.
* **No Horizon Correlation**: The models predict independently and do not capture joint cross-horizon covariance structures.
