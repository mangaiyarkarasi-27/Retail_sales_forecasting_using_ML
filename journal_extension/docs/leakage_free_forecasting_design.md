# Leakage-Free Journal Forecasting Design

This document details the design of the decision-time-valid point-forecasting layer for the journal extension, correcting the temporal feature leakage identified in the conference pipeline.

---

## 1. Motivation

Downstream inventory replenishment decisions occur daily in the morning and require point forecasts of cumulative lead-time demand $\hat{D}_{t, L}$ over horizons $L \in \{1, 3, 5, 7\}$ days. To prevent future-demand leakage, the point predictions must be generated strictly from data available at the decision point.

---

## 2. Conference Pipeline Limitation

The original conference pipeline utilizes rolling features (`rolling_mean_7`, `rolling_mean_30`) and lag features (`sales_lag_7`) that are shifted by only 1 day:
* Predicting tomorrow ($t+1$) requires `rolling_mean_7` at $t+1$, which contains $y_t$ (today's demand).
* Predicting day $t+7$ requires `sales_lag_7` at $t+7$, which contains $y_t$.
Because replenishment decisions occur in the morning before today's demand $y_t$ is observed, these features are temporally unavailable, resulting in look-ahead leakage when evaluated in a rolling daily simulation.

---

## 3. Decision-Time Information Set

At the morning of day $t$, the information set $\mathcal{I}_t$ comprises:
$$\mathcal{I}_t = \{ y_k \mid k \le t-1 \} \cup \{ \text{Calendar}_k \mid \forall k \} \cup \{ \hat{y}_k \mid k \le t-1 \}$$
where $y_k$ are realized daily demand values, $\text{Calendar}_k$ represents calendar metadata (known deterministically in advance), and $\hat{y}_k$ are historical forecasts.

Any forecast generated at day $t$ for day $t+k$ ($k \ge 1$) must be a function of the information set:
$$\hat{y}_{t+k} = f(\mathcal{I}_t)$$

---

## 4. Feature Availability

The table below outlines feature availability at decision point $t$ (morning) for predictions at horizon $t+k$ ($k \ge 1$):

| Feature Name | Horizon $t+k$ | Latest Required Obs | Available at $t$? | Leakage Resolution |
|:---|:---:|:---:|:---:|:---|
| `store`, `item` | $\forall k$ | Static | **YES** | No modification required. |
| Calendar Parts | $\forall k$ | Deterministic | **YES** | No modification required. |
| `sales_lag_7` | $k < 7$ | $t+k-7 \le t-1$ | **YES** | Shifted by $\max(7, k)$ days. |
| `sales_lag_7` | $k \ge 7$ | $t+k-7 \ge t$ | **NO** | Shifted by $k$ days. |
| `rolling_mean_7` | $\forall k \ge 1$ | $t+k-1 \ge t$ | **NO** | Shifted by $k$ days. |
| `rolling_mean_30`| $\forall k \ge 1$ | $t+k-1 \ge t$ | **NO** | Shifted by $k$ days. |

---

## 5. Recursive Forecasting Design

A recursive strategy uses a single one-step-ahead forecasting model $M$:
1. At day $t$, predict tomorrow's sales: $\hat{y}_{t+1} = M(X_{t+1})$, where $X_{t+1}$ contains lags up to $y_{t-1}$ and rolling statistics up to $y_{t-1}$ (which is valid).
2. For day $t+2$, update the feature state by substituting $\hat{y}_{t+1}$ for the unobserved $y_t$.
3. Predict $\hat{y}_{t+2} = M(X_{t+2})$.
4. Repeat recursively up to $\hat{y}_{t+7}$.

*Limitation*: High complexity to reconstruct features recursively in Python, and error propagation (errors in $\hat{y}_{t+1}$ corrupt subsequent predictions).

---

## 6. Direct Multi-Horizon Design

A direct strategy trains $H = 7$ distinct, horizon-specific models $M_1 \dots M_7$:
* Model $M_h$ is trained to predict $y_{t+h}$ directly using features shifted by exactly $h$ days.
* For model $M_h$, the features are:
  * `sales_lag_h` (sales $h$ days ago)
  * `sales_lag_{h+7}` (sales $h+7$ days ago)
  * `sales_lag_{h+23}` (sales $h+23$ days ago)
  * `rolling_mean_7` shifted by $h$ days: $\text{mean}(y_{t-7} \dots y_{t-1})$
  * `rolling_mean_30` shifted by $h$ days: $\text{mean}(y_{t-30} \dots y_{t-1})$

At decision time $t$, all inputs for $M_h$ are fully realized and known.

---

## 7. Methodological Comparison

| Criterion | Recursive XGBoost | Direct Horizon-Specific XGBoost |
|:---|:---:|:---:|
| **Leakage Safety** | Safe | Safe |
| **Error Propagation** | High (predictions used as features) | None (models are decoupled) |
| **Implementation Complexity**| Very High (recursive state loop) | Low (adjust feature shift by horizon) |
| **Feature Set Alignment** | Modified (features generated dynamically) | Preserved (standard lag and roll) |
| **Computation Cost** | Low prediction loop | Moderate training (7 models) |
| **Reproducibility** | Moderate | High |

---

## 8. Validation Strategy

To choose between Recursive and Direct models:
1. Split the training partition (years 2013-2016) chronologically:
   * Sub-Train: 2013-2015
   * Sub-Val: 2016
2. Train both models on Sub-Train and evaluate MAE, RMSE, and Winkler scores on Sub-Val.
3. Keep the 2017 final test partition strictly untouched.

---

## 9. Conformal Integration

For each decision origin $t$ and lead time $L$:
1. The direct/recursive model generates point forecasts $\hat{y}_{t+1} \dots \hat{y}_{t+L}$.
2. The cumulative point demand is $\hat{D}_{t, L} = \sum_{k=1}^L \hat{y}_{t+k}$.
3. When the true cumulative sales $D_{t, L}$ is observed (after $L$ days), the out-of-sample residual $E_{t, L} = |D_{t, L} - \hat{D}_{t, L}|$ is calculated.
4. Conformal quantiles are calculated using the eligible window: $r \le t - L - 1$ (preventing temporal calibration errors).

---

## 10. Inventory Integration

The downstream inventory position $IP_t$ is compared to the reorder point:
$$s_t = \hat{D}_{t, L} + q_{t, L}$$
If $IP_t < s_t$, an order of size $Q$ (computed via training-set EOQ) is triggered.

---

## 11. Leakage Controls

* **Decoupled Pipelines**: The conference pipeline remains unmodified.
* **OOS residuals**: Enforced $r \le t - L - 1$ residual availability.
* **OOS features**: Lags and rolling windows shifted by at least $h$ days for horizon $h$.

---

## 12. Reproducibility

* Random seed frozen at `42` for XGBoost models.
* Script execution and data split configurations saved to `configs/`.

---

## 13. Final Architecture Recommendation

**Direct Multi-Horizon XGBoost**

*Rationale*: Direct multi-horizon forecasting completely eliminates recursive error propagation, preserves standard pandas feature-shift structures (low implementation complexity), and is computationally light (XGBoost fits in under 1 second per model). At decision time $t$, each model $M_h$ predicts $t+h$ using inputs containing zero future-demand leakage.
