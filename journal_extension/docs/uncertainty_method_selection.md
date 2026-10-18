# Uncertainty Method Selection

This document records the design decisions and theoretical rationale for selecting the predictive uncertainty quantification method in the retail forecasting pipeline.

---

## 1. Candidate Methods

We evaluated the following frameworks for estimating forecasting uncertainty intervals:
1. **Standard Split Conformal Prediction**: Uses a fixed split of training and calibration data to find empirical residual quantiles.
2. **Rolling/Blocked Conformal Calibration**: Computes residuals in sliding chronological blocks to adapt to time-series drift.
3. **Ensemble Batch Prediction Intervals (EnbPI)**: Combines bootstrap ensemble models with sliding-window residual updates for time-series.
4. **Parametric Gaussian / Heuristic Intervals**: Assumes residuals are normally distributed and scales standard errors.

---

## 2. Time-Series Dependence

Time-series forecasting violates the fundamental **exchangeability assumption** required for formal distribution-free coverage guarantees in standard conformal prediction. Due to:
* **Autocorrelation**: Today's forecasting error is highly correlated with yesterday's.
* **Annual Seasonality**: Residual distributions vary systematically between peak summer periods and low winter periods.
* **Non-stationarity**: Long-term changes in baseline sales affect residual scale.

As a result, standard split conformal prediction will not provide exact mathematical coverage guarantees, but rather serves as a robust empirical distribution-free interval builder.

---

## 3. Selected Method

We select **Rolling-Origin Conformal Calibration** as our primary method. 

* **How it works**: For each step in the out-of-sample period, the forecasting point model generates multi-step predictions. The prediction interval boundaries are adjusted using the $1-\alpha$ quantile of absolute residuals computed over a sliding historical window (calibration block) of size $W = 100$ days prior to the prediction origin.
* **Alternative selection**: If a rolling origin is computationally too expensive for the 500 series, we fall back to a **Chronological Split Conformal** model with local per-series quantiles, explicitly documenting the exchangeability violations.

---

## 4. Why It Was Selected

1. **Distribution-Free**: Does not assume residuals are Gaussian.
2. **Adapts to Non-stationarity**: The sliding calibration window dynamically scales prediction intervals upward during high-volatility periods and downward during stable regimes.
3. **Simplicity & Speed**: Can be implemented efficiently on CPU without deep learning library requirements.

---

## 5. Calibration Strategy

* **Dataset Split**: Training partition (`data/processed_train.csv` first 80%) split into:
  * **Calibration Train**: First 80% of training split (584,320 rows; Date Range: 2013-01-31 to 2016-03-15).
  * **Calibration Set (Residuals)**: Remaining 20% of training split (146,080 rows; Date Range: 2016-03-15 to 2016-12-31).
* **Local vs. Pooled**: We adopt **local per-series calibration** (each store-item pair has its own residual distribution). With 292 daily observations in the calibration set per series, there is sufficient sample size to estimate the 90% quantile ($\approx$ 263rd sorted value) and 95% quantile ($\approx$ 278th sorted value) stably.

---

## 6. Lead-Time Uncertainty Strategy

We reject the standard scaling heuristic:
$$SS_t = q \times \sqrt{L}$$
This formula assumes that daily forecasting errors are i.i.d. random variables. In retail, daily residuals have high serial correlation, meaning scaling by $\sqrt{L}$ systematically under-estimates the actual lead-time demand variance.

### **Selected Lead-Time Demand Uncertainty Method**
We directly estimate the uncertainty of cumulative lead-time demand:
1. Define the true lead-time demand over $L$ days:
   $$D_{t, L} = \sum_{k=1}^L y_{t+k}$$
2. Generate the multi-step point forecast for the lead-time period:
   $$\hat{D}_{t, L} = \sum_{k=1}^L \hat{y}_{t+k}$$
3. Compute the absolute lead-time prediction error on the calibration set:
   $$E_{t, L} = |D_{t, L} - \hat{D}_{t, L}|$$
4. Sort $E_{t, L}$ and find the conformal quantile $q_L$ at significance level $\alpha$.
5. The Safety Stock is set directly to this lead-time quantile:
   $$SS_t = q_L$$
6. The Reorder Point is set to:
   $$s_t = \hat{D}_{t, L} + q_L$$

This approach directly connects prediction intervals to the actual decision quantity ($D_{t, L}$) being protected in the inventory simulator, bypassing the incorrect $\sqrt{L}$ i.i.d. assumption.

---

## 7. Coverage Metrics

* **Prediction Interval Coverage Probability (PICP)**: Percentage of test sales values falling within the prediction bounds (target: 80%, 90%, 95%).
* **Mean Interval Width (MIW)**: The average width of the prediction bounds.

---

## 8. Inventory Decision Connection

* **Safety Stock**: Set to the lead-time conformal residual quantile $q_L$.
* **Reorder Point**: Sum of point forecasts over the lead time plus safety stock.
* **Risk Flags**: Mapped to interval width; if the interval width exceeds a specified percentile, an alert flag is raised.
* **Decision Rule**: Procurement orders are triggered daily in the simulator if the current inventory level $I_t$ falls below the reorder point $s_t$.

---

## 9. Known Assumptions & Limitations

* **Stationary Residuals**: Assumes the out-of-sample lead-time forecast error distribution is statistically similar to the calibration residuals.
* **Asymmetric Penalties**: Conformal intervals are symmetric around the point forecast, whereas supply chain penalties are highly asymmetric (shortages are more expensive than carrying inventory).

---

## 10. Rejected Alternatives

* **Pooled Calibration**: Rejected because the 500 store-item combinations have high volume heterogeneity (sales range from 12 to 109 units/day), meaning a pooled residual distribution would over-estimate error for low-volume series and under-estimate it for high-volume series.
* **Square Root of Lead-Time Scaling**: Rejected due to serial autocorrelation in daily forecasting residuals.
