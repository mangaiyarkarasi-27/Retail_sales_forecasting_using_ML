# FINAL METHODOLOGY FREEZE — JOURNAL EXTENSION

**Document Version**: 1.0 (Frozen)  
**Date**: 2026-09-19  
**Experiment Reference**: EXP-020 (Audit of EXP-011 through EXP-019)  
**Status**: PERMANENT METHODOLOGY FREEZE (No further hyperparameter tuning, model modification, or calibration re-estimation permitted)

---

## 1. Dataset & Partition Protocol

- **Dataset**: Kaggle Retail Sales Forecasting Benchmark (10 stores × 50 items = 500 time series).
- **Time Range**: 2013-01-01 to 2017-12-31 (1,826 days per series, 913,000 observations total).
- **Training Partition (Model Fitting)**: 2013-01-01 to 2015-12-31 (1,095 days).
- **Validation Partition (Model & Parameter Selection)**: 2016-01-01 to 2016-12-31 (366 days).
- **Final Out-of-Sample Test Partition**: 2017-01-01 to 2017-12-31 (365 days).
- **Temporal Constraint**: Zero 2017 observations are used in model training, feature selection, hyperparameter tuning, or calibration parameter estimation.

---

## 2. Forecasting Layer (EXP-011)

- **Architecture**: Direct Multi-Horizon Forecasting with 7 Global XGBoost Regressors ($M_1, \dots, M_7$).
- **Horizon Mapping**: Model $M_h$ directly predicts $y_{t+h}$ at forecast origin $t$.
- **Features (16 Decision-Time Causal Features)**:
  - Temporal Lags: `sales_lag_7`, `sales_lag_14`, `sales_lag_30` (strictly $\ge 7$ days to prevent look-ahead for all $h \in \{1,\dots,7\}$).
  - Rolling Windows: `rolling_mean_7`, `rolling_mean_30` (computed strictly on sales $\le t-7$).
  - Deterministic Calendar: `year`, `month`, `day`, `weekday`, `weekofyear`, `quarter`, `is_weekend`, `month_start`, `month_end`.
  - Identifiers: `store`, `item`.
- **Model Hyperparameters**:
  - `n_estimators = 100`
  - `max_depth = 6`
  - `learning_rate = 0.1`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8`
  - `random_state = 42`
  - `objective = 'reg:squarederror'`
- **Validation Performance (2016)**: Aggregate MAE = **6.0388**, RMSE = **7.8638**.

---

## 3. Conformal Uncertainty Layer (EXP-018 Frozen)

- **Methodology Name**: Rolling Forecast-Conditional Conformal Calibration.
- **Terminology Rule**: Because the optimal calibration parameter $K^*=100$ spans the full sliding window $W=100$, do NOT refer to the final method as a KNN/local-neighbor method in the journal manuscript. Use *rolling forecast-conditional conformal calibration* or *scale-normalized conformal prediction*.
- **Sliding Calibration Window ($W$)**: $W = 100$ days.
- **Calibration Pool Eligibility**: Residuals must satisfy $r \le t - L - 1$ at forecast origin $t$.
- **Scale-Normalized Nonconformity Score**:
  $$s_{r,L} = rac{|y_{r,L} - \hat{D}_{r,L}|}{\hat{D}_{r,L} + \epsilon}, \quad \epsilon = 10^{-4}$$
- **Quantile Computation**:
  $$q_{t,L}^{(eta)} = 	ext{Quantile}_{\lceil (W+1)eta ceil / W}\left(\{s_{r,L}\}_{r \in \mathcal{C}_{t,L}}ight)$$
- **Upper Prediction Bound & Safety Stock**:
  $$	ext{Upper}_{t,L}^{(eta)} = \hat{D}_{t,L} \cdot (1 + q_{t,L}^{(eta)})$$
  $$SS_{t,L}^{(eta)} = q_{t,L}^{(eta)} \cdot \hat{D}_{t,L}$$

---

## 4. Inventory Decision & Simulation Layer (EXP-019)

- **Simulation Type**: Daily Discrete-Event Simulation with Lost Sales.
- **Replenishment Policies**:
  - **Policy A (Deterministic)**: $ROP_{t,L} = \hat{D}_{t,L} = \sum_{k=1}^L \hat{y}_{t+k}$, $SS_{t,L} = 0$.
  - **Policy B (Conformal Uncertainty-Aware)**: $ROP_{t,L} = \hat{D}_{t,L} + SS_{t,L}^{(eta)}$, where $SS_{t,L}^{(eta)} = q_{t,L}^{(eta)} \cdot \hat{D}_{t,L}$.
- **Lead Times Evaluated ($L$)**: $L \in \{1, 3, 5, 7\}$ days.
- **Nominal Coverage Targets ($eta$)**: $eta \in \{0.80, 0.90, 0.95\}$.
- **Daily Event Execution Sequence (Strict 8 Steps)**:
  1. Receive arrivals from $t-L$: $OH_t^{\text{start}} = OH_{t-1}^{\text{close}} + \text{Arrivals}_t$.
  2. Evaluate inventory position: $IP_t = OH_t^{\text{start}} + OO_t$.
  3. Evaluate $ROP_{t,L}$ using morning forecasts and eligible conformal quantiles.
  4. Replenishment decision: If $IP_t < ROP_{t,L}$, $Q_t = ROP_{t,L} - IP_t$, else $Q_t = 0$.
  5. Schedule order for arrival at $t+L$: $OO_t \leftarrow OO_t + Q_t$.
  6. Realize daily actual sales $y_{i,t}$.
  7. Fulfill demand: $\text{Fulfilled}_t = \min(OH_t^{\text{start}}, y_{i,t})$; Shortage $\text{Shortage}_t = \max(0, y_{i,t} - OH_t^{\text{start}})$; Closing stock $OH_t^{\text{close}} = \max(0, OH_t^{\text{start}} - y_{i,t})$. Lost sales permanently dropped.
  8. Settle daily costs: Holding $h \cdot OH_t^{\text{close}}$, Ordering $K_{\text{order}} \cdot \mathbb{I}(Q_t > 0)$, Shortage $p \cdot \text{Shortage}_t$.
- **Initial Inventory**: $OH_1 = L \cdot \mu_i^{\text{pre-2017}}$ computed strictly from 2013-2016 training sales. $OO_0 = 0$.
- **End-of-Horizon Protection**: Final $L$ days excluded from metric aggregation to prevent pipeline truncation artifacts.
- **Cost Parameters**:
  - Fixed Order Cost: $K_{\text{order}} = \$10.00$ per order.
  - Daily Holding Cost: $h = \$0.20$ per unit per day on closing on-hand stock.
  - Shortage Penalty: $p = \$1.00$ per unit (Primary); Sensitivities $p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$.

---

## 5. Statistical Evaluation & Reproducibility Protocol

- **Paired Statistical Testing**: Two-sided paired Wilcoxon signed-rank tests across all $N=500$ individual store-item time series.
- **Bootstrap Protocol**: 1,000 paired bootstrap iterations resampling across the 500 store-item series to construct 95% confidence intervals.
- **Demand Regimes**: Sliced by pre-2017 median volume into Medium and High demand series.
- **Random Seed**: 42 fixed across all random number generators.
- **Reproducibility Guarantee**: Codebase in `journal_extension/src/` with frozen test suites in `journal_extension/tests/`.
