# Inventory Simulation Leakage Audit

This document presents the detailed audit of the data flows, decision boundaries, and information constraints in the uncertainty-aware inventory simulator, distinguishing between different leakage types and identifying temporal feature constraints.

---

## 1. Information Availability Matrix

At day $t$ (morning), before a replenishment order is triggered, the simulation policy may access the following variables:

| Information | Available at decision t? | Leakage Type if Violated | Reason / Constraint |
|:---|:---:|:---:|:---|
| **Historical demand** | **YES** | N/A | Realized daily sales up to yesterday ($y_{t-1}$) are known. |
| **Today's demand $y_t$** | **NO** | **Future-Demand Leakage** | Replenishment decisions occur in the morning, before daily demand. |
| **Future demand ($y_{t+k}$)** | **NO** | **Future-Demand Leakage** | Realized sales during and after the lead-time period are unknown. |
| **Current inventory** | **YES** | N/A | Physical stock on-hand ($OH_t$) is updated at the end of day $t-1$. |
| **Outstanding orders** | **YES** | N/A | Outstanding orders ($OO_t$) are tracked chronologically. |
| **Point forecast ($\hat{D}_{t, L}$)** | **QUALIFIED** | **Future-Demand Leakage** | Available in offline data, but requires today's/future sales in rolling features. |
| **Conformal quantile ($q_{t, L}$)** | **YES** | N/A | Derived from historically realized residuals. |
| **Unobserved future residual** | **NO** | **Temporal Calibration Error**| Residuals whose outcomes are not fully realized cannot be accessed. |
| **Test-Period Targets** | **NO** | **Test-Target Leakage** | 2017 actual sales must never be used for training or calibration. |

---

## 2. Leakage vs. Temporal Error Definitions

To maintain research integrity, we distinguish between three distinct information failures:
1. **Future-Demand Leakage**: Occurs when a decision at day $t$ utilizes demand information from day $t$ onwards (e.g. using today's sales to compute tomorrow's features).
2. **Test-Target Leakage**: Occurs when target variables from the out-of-sample test partition (year 2017) are used to train, tune, or select parameters of the point model or calibrator prior to test evaluation.
3. **Temporal Calibration Error**: Occurs when the sliding window of calibration residuals at day $t$ includes residuals that are not yet fully realized (originating at $t - L$ to $t - 1$, which require future demand observations to calculate).

---

## 3. Point Forecast Feature Look-Ahead Audit

* **The Problem**: The frozen XGBoost model uses rolling features (`rolling_mean_7` and `rolling_mean_30`) shifted by only 1 day. At day $t$, predicting tomorrow $t+1$ requires the feature `rolling_mean_7` at $t+1$, which includes $y_t$ (today's sales).
* **Impact**: Because decisions occur in the morning of day $t$ before today's demand occurs, $y_t$ is unknown. Therefore, tomorrow's forecast $\hat{y}_{t+1}$ cannot be computed at day $t$ without future-demand leakage.
* **Resolution**: Re-train the model with features shifted by $L$ days (or use recursive forecasting where predictions replace missing future demand values).

---

## 4. Corrected Conformal Residual Timing

To prevent **temporal calibration errors**, each residual $E_{r, L}$ used in calibration at day $t$ must be fully realized:
$$r \le t - L - 1$$
* **$L=1$**: $r \le t - 2$ (calibration window uses $[t-101, t-2]$).
* **$L=3$**: $r \le t - 4$ (calibration window uses $[t-103, t-4]$).
* **$L=7$**: $r \le t - 8$ (calibration window uses $[t-107, t-8]$).
Enforcing this boundary prevents temporal calibration errors for all lead times.
