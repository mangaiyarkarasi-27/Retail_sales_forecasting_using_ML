# Forecast Availability Audit

This document presents a rigorous feature-by-feature temporal audit of the point-forecasting pipeline to verify whether predictions used for downstream inventory decisions are free from look-ahead leakage.

---

## 1. XGBoost Features

The frozen XGBoost model uses 16 active features. Below is the timing and availability analysis for each feature at decision time $t$ (morning), assuming the latest observed sales demand is $y_{t-1}$ (yesterday's sales):

| Feature Name | Definition / Calculation | Latest Observation Date required at $t+k$ | Available at morning $t$ (for $k \ge 1$)? | Requires Future Demand? |
|:---|:---|:---|:---:|:---:|
| `store` | Categorical store ID | N/A (Static) | **YES** | **NO** |
| `item` | Categorical item ID | N/A (Static) | **YES** | **NO** |
| `year` | Calendar year | $t+k$ | **YES** | **NO** |
| `month` | Calendar month | $t+k$ | **YES** | **NO** |
| `day` | Day of month | $t+k$ | **YES** | **NO** |
| `weekday` | Day of week | $t+k$ | **YES** | **NO** |
| `weekofyear` | ISO week index | $t+k$ | **YES** | **NO** |
| `quarter` | Calendar quarter | $t+k$ | **YES** | **NO** |
| `is_weekend` | `1` if weekend, `0` otherwise | $t+k$ | **YES** | **NO** |
| `month_start` | `1` if day is month start | $t+k$ | **YES** | **NO** |
| `month_end` | `1` if day is month end | $t+k$ | **YES** | **NO** |
| `sales_lag_7` | $y_{t+k-7}$ | $t+k-7$ | **NO** for $k \ge 7$ | **YES** (at $t+7$, requires $y_t$) |
| `sales_lag_14` | $y_{t+k-14}$ | $t+k-14$ | **YES** for $k \le 7$ | **NO** |
| `sales_lag_30` | $y_{t+k-30}$ | $t+k-30$ | **YES** for $k \le 7$ | **NO** |
| `rolling_mean_7` | $\text{mean}(y_{t+k-7} \dots y_{t+k-1})$ | $t+k-1$ | **NO** for all $k \ge 1$ | **YES** (at $t+1$, requires $y_t$) |
| `rolling_mean_30`| $\text{mean}(y_{t+k-30} \dots y_{t+k-1})$| $t+k-1$ | **NO** for all $k \ge 1$ | **YES** (at $t+1$, requires $y_t$) |

---

## 2. Feature Information Timing

At decision time $t$ (morning), the latest observed demand value is $y_{t-1}$. 
* **Lag Features**: For $L \le 7$, any lag of 7 days or more for target dates $t+1 \dots t+6$ only requires sales up to $y_{t-1}$. However, at $t+7$, `sales_lag_7` requires $y_t$ (today's sales), which has not occurred yet.
* **Rolling Features**: Because the rolling features are shifted by only 1 day (`shift(1)`), calculating `rolling_mean_7` for tomorrow ($t+1$) requires $y_t$ (today's demand). Calculating it for $t+2$ requires $y_{t+1}$, and so on.
* *Conclusion*: Rolling features and `sales_lag_7` (at horizon $t+7$) are **temporally unavailable** at the morning of $t$.

---

## 3. Forecast Generation Mechanism

The current system uses a **Direct forecasting approach on pre-computed feature tables** for test-set evaluation. The test set predictions `test_df["pred_sales"]` were pre-calculated using the true realized sales values of the test partition to populate the rolling mean columns. 
In a physical rolling deployment, this mechanism is **not realizable** because the rolling inputs for tomorrow do not exist yet.

---

## 4. Horizon Availability

| Horizon | Forecast available at t? | Reason |
|:---:|:---:|:---|
| **$t+1$** | **NO** | Requires $y_t$ (today's sales) to compute `rolling_mean_7` and `rolling_mean_30`. |
| **$t+2$** | **NO** | Requires $y_t, y_{t+1}$ to compute rolling features. |
| **$t+3$** | **NO** | Requires $y_t \dots y_{t+2}$ to compute rolling features. |
| **$t+4$** | **NO** | Requires $y_t \dots y_{t+3}$ to compute rolling features. |
| **$t+5$** | **NO** | Requires $y_t \dots y_{t+4}$ to compute rolling features. |
| **$t+6$** | **NO** | Requires $y_t \dots y_{t+5}$ to compute rolling features. |
| **$t+7$** | **NO** | Requires $y_t \dots y_{t+6}$ for rolling features, and $y_t$ for `sales_lag_7`. |

---

## 5. Rolling Features & Calendar Features

* **Rolling Features**: Enforce a look-ahead shift of size $L$ rather than 1 to remain leakage-free without recursive imputation.
* **Calendar Features**: Fully available. Future day-of-week and month indicators are deterministic calendar variables.

---

## 6. Lead-Time Forecast Construction

The cumulative lead-time demand forecast $\hat{D}_{t, L} = \sum_{k=1}^L \hat{y}_{t+k}$ is currently built from component predictions that contain look-ahead features.
* **Recommended Modification**: Re-train a multi-step forecasting model where features are shifted by $L$ days instead of 1, or recursively forecast future values by feeding predictions back as inputs.

---

## 7. Conformal Residual Timing

To prevent **temporal calibration errors**, each residual $E_{r, L}$ used in calibration at day $t$ must be fully realized:
$$r \le t - L - 1$$
* **$L=1$**: $r \le t - 2$ (calibration window uses $[t-101, t-2]$).
* **$L=3$**: $r \le t - 4$ (calibration window uses $[t-103, t-4]$).
* **$L=7$**: $r \le t - 8$ (calibration window uses $[t-107, t-8]$).

---

## 8. Test-Target Isolation & Inventory Initialization

* **Test-Target Isolation**: No 2017 target observations are used for fitting models or residual calculations before their respective simulation step.
* **Inventory Initialization**: opening stock is initialized deterministically:
  $$OH_1 = L \times \mu$$
  where $\mu$ is calculated exclusively from training data (years 2013-2016).

---

## 9. Final Decision

**DO NOT IMPLEMENT YET (MAJOR FORECASTING CORRECTION REQUIRED)**

*Rationale*: The frozen point model (XGBoost) utilizes rolling average features shifted by only 1 day, which requires today's (and future) sales to predict demand from tomorrow onwards. While acceptable for static offline test evaluations, this constitutes look-ahead leakage in a rolling inventory simulation and cannot be deployed in practice without re-aligning features or generating forecasts recursively.
