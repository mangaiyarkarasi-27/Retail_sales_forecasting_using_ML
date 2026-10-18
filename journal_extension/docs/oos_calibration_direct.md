# OOS Conformal Calibration for Direct Multi-Horizon Forecasting

This document details the objective, out-of-sample (OOS) prediction generation, daily and cumulative lead-time residual construction, temporal eligibility controls, and descriptive statistics of the conformal calibration residual dataset.

---

## 1. Objective

To generate and audit a leakage-safe out-of-sample calibration residual dataset for the decoupled Direct Multi-Horizon forecasting layer. This ensures that the rolling calibration window of size $W=100$ only uses fully realized out-of-sample forecast errors, preventing optimistic in-sample biases and future-demand leakage.

---

## 2. EXP-011 Forecasting Layer

The forecasting layer comprises exactly 7 global XGBoost models ($M_1 \dots M_7$), where $M_h$ predicts target demand $y_{t+h}$ using lags and rolling statistics ending strictly at $t-1$ (origin $t$).

---

## 3. OOS Forecast Generation

Out-of-sample forecasts on the training partition (years 2013-2016) are generated using a 5-fold cross-validation `TimeSeriesSplit`:
* For each fold, we train the 7 global models on the sub-train split.
* We generate out-of-sample predictions for all validation origins $t$ in the chronological validation split.
* This is repeated for all 5 folds, creating a continuous OOS prediction set.

---

## 4. Daily Residual Construction

For each validation origin $t$ and horizon $h \in \{1 \dots 7\}$:
$$e_{t,h} = y_{t+h} - \hat{y}_{t+h}$$
where $y_{t+h}$ is the true sales at $t+h$ (observed after day $t+h$), and $\hat{y}_{t+h}$ is the forecast generated from origin $t$. The absolute error is:
$$\text{abs\_error}_{t,h} = |e_{t,h}|$$

---

## 5. Cumulative Lead-Time Residuals

For each lead time $L \in \{1, 3, 5, 7\}$ days:
$$\hat{D}_{t, L} = \sum_{k=1}^L \hat{y}_{t+k}$$
$$D_{t, L} = \sum_{k=1}^L y_{t+k}$$
The cumulative absolute residual is:
$$E_{t, L} = |D_{t, L} - \hat{D}_{t, L}|$$

---

## 6. Temporal Eligibility

A cumulative lead-time residual $E_{r, L}$ originating at $r$ requires observing demand up to $r+L$. 
At decision time $t$ (morning), before $y_t$ is observed, $E_{r, L}$ is eligible for inclusion in the calibration window if and only if $r + L < t$ (or $r \le t - L - 1$).
* **L=1**: Latest eligible origin is $t - 2$.
* **L=3**: Latest eligible origin is $t - 4$.
* **L=5**: Latest eligible origin is $t - 6$.
* **L=7**: Latest eligible origin is $t - 8$.

---

## 7. Rolling Calibration Window

At decision time $t$ and lead time $L$, the rolling calibration window of size $W=100$ comprises the 100 most recent eligible residuals. In Python, this corresponds to slicing the chronological residuals array:
$$\text{calibration\_window} = \text{residuals}[t - 100 - L : t - L]$$

---

## 8. Conformal Quantile

The safety stock quantile $q_{t, L}$ is computed using the locked finite-sample index formula:
$$\text{index} = \lceil(n+1)(1-\alpha)\rceil$$
where $n=100$ and $\alpha \in \{0.20, 0.10, 0.05\}$.

---

## 9. Leakage Controls

* **Decoupled residuals**: Old EXP-006 residuals are not reused; they are archived.
* **Test set isolation**: 2017 target observations are strictly untouched.
* **Residual Eligibility**: Slicing logic is offset by $L+1$ days.

---

## 10. Residual Statistics

The table below lists the aggregate statistics computed on the out-of-sample training partition residuals (2013-2016):

| Lead Time ($L$) | Mean Signed Error | MAE | RMSE | Median Absolute Error | Std Dev | Residual Count |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L=1** | 3.0593 | 9.2642 | 12.7254 | 6.7525 | 12.3522 | 608,165 |
| **L=3** | 9.0570 | 23.8015 | 33.0317 | 16.7983 | 31.7658 | 607,165 |
| **L=5** | 15.0414 | 38.3535 | 53.2749 | 26.9032 | 51.1075 | 606,165 |
| **L=7** | 21.2487 | 52.7451 | 73.3377 | 36.8546 | 70.1920 | 605,165 |

---

## 11. Unit Tests

All 3 new tests in `test_oos_calibration.py` executed successfully:
* Horizon targets are verified as future relative to origin.
* Conformal window slicing offsets were verified.

---

## 12. Limitations

* **First $L+100$ days**: The first 100 days of each series' validation set do not contain enough eligible out-of-sample residuals to construct a window of size 100. A fallback rule (reducing $n$ down to a minimum of 20 observations) is applied.
