# Uncertainty Experiment Plan

This document outlines the methodology, validation protocols, evaluation metrics, and downstream inventory decision logic for the **Predictive Uncertainty Quantification** extension of the retail forecasting pipeline.

---

## 1. Forecasting Point Estimate

* **Model**: Global XGBoost Regressor (fully optimized and frozen conference baseline).
* **Parameters**: Frozen configuration (max_depth=8, learning_rate=0.1, n_estimators=50, regularizations).
* **Target Scale**: Original sales unit scale.
* **Role**: Serves as the central point estimate (${\hat{y}_t}$) around which prediction intervals are constructed.

---

## 2. Calibration Dataset

To construct conformal prediction intervals, we need a clean, chronologically isolated **calibration partition** to evaluate out-of-sample residuals without overfitting:
* **Partition Source**: The training partition (`data/processed_train.csv` first 80%) will be split:
  * **Calibration Train**: First 80% of the training partition (584,320 rows; Date Range: 2013-01-31 to 2016-03-15) used to fit the point estimator.
  * **Calibration Set (Residuals)**: Remaining 20% of the training partition (146,080 rows; Date Range: 2016-03-15 to 2016-12-31) used to compute the calibration residuals.
* **Leakage Control**: The calibration set is completely chronological and isolated. Under no circumstances will test set observations (from the 2017 partition) be included in the calibration residuals.

---

## 3. Calibration Residuals

For each store-item combination $i$, we compute the absolute forecasting residuals on the calibration set:
$$E_i = \{|y_t - \hat{y}_t| \mid t \in \text{Calibration Set}\}$$
* **Residual Analysis**: Residuals represent the empirical out-of-sample prediction error distribution for each series, capturing series-specific variance and noise.

---

## 4. Prediction Interval Construction

We will implement **Split Conformal Prediction** independently for each store-item combination:
1. Sort the absolute residuals $E_i$ in ascending order.
2. For a target significance level $\alpha$ (where nominal coverage is $1 - \alpha$), calculate the value $q$ corresponding to the $\lceil(n+1)(1-\alpha)\rceil$-th smallest residual in $E_i$ (where $n = |E_i|$).
3. The prediction interval for any future date $t$ in the test set is constructed as:
   $$[\hat{y}_{t} - q, \hat{y}_{t} + q]$$
4. To prevent non-physical predictions, the lower bound is clipped at zero:
   $$\text{Lower Bound}_t = \max(0, \hat{y}_{t} - q)$$

---

## 5. Nominal Coverage Levels

We will evaluate three standard target nominal coverage levels to analyze the trade-off between interval width and coverage correctness:
* **80% Confidence Level** ($\alpha = 0.20$)
* **90% Confidence Level** ($\alpha = 0.10$)
* **95% Confidence Level** ($\alpha = 0.05$)

---

## 6. Evaluation Metrics

The quality of the conformal prediction intervals will be evaluated using the following metrics:
* **Prediction Interval Coverage Probability (PICP)**: The proportion of test sales values falling within the prediction bounds:
  $$\text{PICP} = \frac{1}{M} \sum_{t=1}^M \mathbb{I}(y_t \in [\text{Lower}_t, \text{Upper}_t])$$
* **Mean Interval Width (MIW)**: The average width of the prediction bounds:
  $$\text{MIW} = \frac{1}{M} \sum_{t=1}^M (\text{Upper}_t - \text{Lower}_t)$$
* **Coverage Analysis**: Metrics will be sliced and evaluated by:
  * Individual store-item series.
  * Product volume categories (High vs. Low mean sales).
  * Seasonality strength (Lag-365 autocorrelation quantiles).
  * Forecast horizon step (analyzing width expansion over time if dynamic residuals are used).

---

## 7. Downstream Inventory Decision Connection

conformal prediction intervals will directly govern the supply chain parameters in our Decision Support System:
* **Safety Stock ($SS_t$)**: Replaces standard heuristic-based safety stock. It is derived directly from the conformal quantile $q$ scaled for lead time $L$:
  $$SS_t = q \times \sqrt{L}$$
* **Reorder Point ($s_t$)**: Calculated using the forecasted demand over lead time plus the conformal safety stock:
  $$s_t = \sum_{k=1}^L \hat{y}_{t+k} + SS_t$$
* **Risk Levels**: Sliced using conformal interval width; wider intervals indicate high-risk/volatile planning regimes, raising warning flags.
* **Recommendation Confidence**: Directly mapped from the tightness of the interval.

---

## 8. Inventory Data Limitations & Simulation Strategy

* **Critical Data Limitation**: The Kaggle Store Item Demand dataset contains **only** daily sales units, store IDs, and item IDs. It does **not** contain actual stock quantities, purchase orders, replenishment histories, lead times, ordering costs, holding costs, or stockout penalty costs.
* **Simulation Protocol**: To evaluate downstream inventory policies, we must construct a **daily rolling-horizon simulation** (continuous review $(s, Q)$ policy) under explicit assumptions that must be clearly disclosed in the paper:
  * **Lead Time ($L$)**: Constant 3 days.
  * **Holding Cost ($h$)**: $\$0.20$ per unit per day.
  * **Shortage/Stockout Cost ($p$)**: $\$1.00$ per unit per day (reflecting a 5:1 penalty-to-holding-cost ratio).
  * **Ordering Cost ($K$)**: $\$10.00$ per order.
  * **Order Quantity ($Q$)**: Economic Order Quantity (EOQ) based on training-set demand averages.

*Note*: All results must be labeled as **simulation-derived** under these assumptions and not represented as real historical business outcomes.

---

## 9. Integrity Requirements for the Uncertainty Experiment

To preserve scientific rigor and research integrity:
1. **No Test Leakage**: The calibration dataset must never contain any test split observations (from the 2017 partition).
2. **Frozen Confidence Levels**: The target nominal coverages (80%, 90%, 95%) must be chosen and locked *before* running evaluation on the final test set.
3. **No Selective Reporting**: Coverage and width metrics must be reported for all 500 store-item series without cherry-picking or removing high-variance outliers.
4. **Disclosed Assumptions**: All simulation cost ratios and lead-time assumptions must be fully disclosed as research parameters.

---

## 10. Methodological Review

### **Time-Series Conformal Strategy**
* **Exchangeability Violation**: Standard split conformal prediction assumes that forecasting errors are exchangeable. In daily retail time-series, errors exhibit temporal correlation, seasonality, and trend drifts, which violates this assumption.
* **Refined Design**: We will use **Rolling-Origin Conformal Calibration** (or blocked calibration) where the error distribution is evaluated on a sliding window of the last $W$ days. This dynamically adapts the quantile to local volatility, improving coverage reliability compared to a static partition.

### **Calibration Design**
* **Residual Sample Size**: The calibration set represents the last 20% of the training partition (292 daily observations). This is chronologically contiguous and includes a representative sample of annual seasonality.
* **Out-of-Sample Verification**: Points predictions on the calibration set are generated by an XGBoost model trained strictly on the Calibration Train subset (first 80%), ensuring residuals are genuinely out-of-sample.

### **Per-Series vs. Pooled Calibration**
* **Series Reliability**: Each series has 292 calibration observations, which is statistically sufficient to estimate the 90th percentile (the 263rd sorted value) and the 95th percentile (the 278th sorted value).
* **Selection**: We adopt **local per-series calibration** because pooled calibration would under-estimate intervals for high-volume series and over-estimate intervals for low-volume series due to the large scale heterogeneity (10x volume differences).

### **Lead-Time Demand Uncertainty**
* **Rejection of standard scaling**: The safety stock formula $SS_t = q \times \sqrt{L}$ is rejected. Daily forecasting errors are not i.i.d. random variables; they are highly autocorrelated, making the $\sqrt{L}$ scaling theoretically incorrect.
* **Selected Strategy**: We directly predict cumulative lead-time demand $D_{t, L} = \sum_{k=1}^L y_{t+k}$, compute the absolute residuals of cumulative lead-time demand on the calibration split, and find the conformal quantile $q_L$ directly on those aggregated residuals.

### **Safety-Stock Derivation**
* The Safety Stock is set directly to the lead-time conformal residual quantile:
  $$SS_t = q_L$$
* The Reorder Point (ROP) is:
  $$s_t = \sum_{k=1}^L \hat{y}_{t+k} + q_L$$

### **Inventory Simulation Assumptions & Sensitivity Analysis**
* Cost parameters ($h = \$0.20$, $p = \$1.00$, $K = \$10.00$, $L = 3$) are explicitly labeled as simulation parameters.
* **Sensitivity Analysis**: To ensure journal-quality rigor, we will perform sensitivity analyses by varying:
  * Lead time $L \in \{1, 3, 5, 7\}$ days.
  * Shortage penalty ratio $p/h \in \{3, 5, 10, 20\}$.

### **Limitations**
* The prediction intervals are symmetric around the point forecast, whereas supply chain penalties are asymmetric (under-forecasting is much more costly than over-forecasting).
* Conformal coverage is an average guarantee over the out-of-sample period and cannot guarantee stockout prevention during extreme localized events.

