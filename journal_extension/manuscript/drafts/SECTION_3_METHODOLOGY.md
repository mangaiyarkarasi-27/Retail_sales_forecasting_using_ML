# SECTION 3: METHODOLOGY

## 3.1 End-to-End Decision Framework and System Architecture

The proposed decision-support framework establishes an end-to-end, leakage-free operational pipeline that translates raw multi-item retail sales time series into risk-buffered, cost-optimized inventory replenishment decisions. The methodology integrates four distinct functional layers: (1) a **Strictly Causal Feature Engineering and Direct Multi-Horizon Forecasting Layer**, (2) a **Rolling Forecast-Conditional Conformal Calibration Layer**, (3) an **Empirical Distribution Shift Diagnostic Layer**, and (4) a **Discrete-Event Multi-Horizon Inventory Simulation and Cost Optimization Layer**.

Figure 1 illustrates the overall system architecture and information flow across the four layers.

```
+---------------------------------------------------------------------------------------------------+
| LAYER 1: STRICTLY CAUSAL DIRECT MULTI-HORIZON FORECASTING (EXP-011)                               |
|                                                                                                   |
|  Retail Sales History          16 Decision-Time Features (Lag >= 7)       7 Global XGBoost Regressors |
|  {y_{i,tau}}_{tau <= t}   -->  x_{i,t} = [Lags, Rolling, Calendar, ID]  --> M_1(x_{i,t}) ... M_7(x)   |
|                                                                                |                  |
|                                                                                v                  |
|                                                                        Lead-Time Forecast         |
|                                                                        \hat{D}_{i,t,L} = \sum \hat{y} |
+--------------------------------------------------------------------------------|------------------+
                                                                                 |
                                                                                 v
+---------------------------------------------------------------------------------------------------+
| LAYER 2: ROLLING FORECAST-CONDITIONAL CONFORMAL CALIBRATION (EXP-018)                             |
|                                                                                                   |
|  Rolling-Origin OOS Pool       Scale-Normalized Nonconformity        Empirical Quantile (W=100)   |
|  {e_{i,r,L}}_{r <= t-L-1}  --> s_{i,r,L} = |e_{i,r,L}| / (\hat{D}+\epsilon) -> q_{i,t,L}^{(\beta)}         |
|                                                                                |                  |
|                                                                                v                  |
|                                                                        Conformal Safety Stock     |
|                                                                        SS_{i,t,L}^{(\beta)} = q \cdot \hat{D} |
+--------------------------------------------------------------------------------|------------------+
                                                                                 |
                                                                                 v
+---------------------------------------------------------------------------------------------------+
| LAYER 3: MULTI-HORIZON INVENTORY REPLENISHMENT POLICY (EXP-019)                                   |
|                                                                                                   |
|  Dynamic Reorder Point         Inventory Position Evaluation         Order Placement Decision     |
|  ROP_{i,t,L} = \hat{D} + SS    IP_{i,t} = OH_{i,t}^{start} + OO_{i,t}  --> Q_{i,t} = max(0, ROP - IP) |
+--------------------------------------------------------------------------------|------------------+
                                                                                 |
                                                                                 v
+---------------------------------------------------------------------------------------------------+
| LAYER 4: DISCRETE-EVENT SIMULATION & FINANCIAL SETTLEMENT (EXP-019)                               |
|                                                                                                   |
|  Daily 8-Step Timing Loop      Lost Sales Realization               Operational Cost Accounting   |
|  Arrivals -> IP -> ROP -> Q    Fulfilled = min(OH, y_{i,t})         C_{total} = Holding +         |
|  -> Demand -> Settle Costs     Shortage = max(0, y_{i,t} - OH)                  Ordering +        |
|                                OH^{close} = max(0, OH - y_{i,t})                Shortage Penalties|
+---------------------------------------------------------------------------------------------------+
```
*Figure 1: End-to-end architecture of the conformal uncertainty-aware retail decision support system.*

To preserve scientific rigor and eliminate look-ahead bias, every layer is governed by strict chronological separation:
1. Historical point-forecasting models are trained exclusively on pre-2016 data (2013–2015).
2. Hyperparameters and uncertainty calibration mechanisms are tuned on the 2016 validation partition.
3. The out-of-sample residual calibration pool is populated via rolling-origin cross-validation over 2013–2016 without accessing forward test records.
4. The entire 2017 calendar year is preserved as an untouched test environment for operational simulation and financial evaluation.

---

## 3.2 Mathematical Formulation of Retail Demand and Multi-Horizon Planning

Consider a multi-item, multi-store retail portfolio comprising $N$ distinct store-item time series indexed by $i \in \{1, \dots, N\}$, observed over a discrete daily time horizon $t \in \{1, \dots, T\}$. Let $y_{i,t} \in \mathbb{R}_{\ge 0}$ denote the realized demand (sales quantity in units) for store-item pair $i$ on calendar day $t$. 

Let $\mathcal{F}_t = \sigma(\{y_{i,\tau} : 1 \le \tau \le t, 1 \le i \le N\})$ represent the historical information filtration available up to and including the close of day $t$. At the morning of decision day $t$, an inventory manager must place replenishment orders to cover customer demand over a replenishment lead time $L \in \{1, \dots, H\}$ days, where $H$ denotes the maximum planning horizon (in our implementation, $H = 7$ days).

The cumulative lead-time demand over the forward interval from $t+1$ through $t+L$ is defined as:

$$D_{i,t,L} = \sum_{k=1}^L y_{i,t+k} \tag{1}$$

Because physical supplier delivery requires $L$ days, an order placed at decision time $t$ will arrive at the beginning of day $t+L$. Consequently, the inventory position must be sized to cover the cumulative stochastic lead-time demand $D_{i,t,L}$. The forecasting objective is therefore not merely to predict single-step sales $y_{i,t+1}$, but to construct accurate, calibrated point and interval forecasts for the multi-horizon vector $(y_{i,t+1}, y_{i,t+2}, \dots, y_{i,t+L})$ and their cumulative sum $D_{i,t,L}$ using only information available prior to the decision point.

---

## 3.3 Strictly Causal Feature Engineering and Zero-Leakage Protocol

A frequent source of optimistic bias in empirical time-series literature is feature leakage, where predictive features inadvertently aggregate future ground-truth demand within the forecast lead-time window \cite{tiwari2026explainable}. In a real-world multi-horizon setting where forecasts must be generated for horizons up to $H=7$ days ahead without intermediate model retraining, any autoregressive lag smaller than 7 days (e.g., $y_{i,t-1}$) is unavailable for direct horizon models targeting $t+7$ unless recursive predictions are substituted.

To eliminate recursive error compounding and guarantee strict decision-time causality across all horizons $h \in \{1, \dots, 7\}$, we establish a **Zero-Leakage Minimum-Lag Protocol**. Under this protocol, all dynamic autoregressive and rolling features are extracted strictly from historical sales realized at or before day $t-7$.

Formally, at forecast origin $t$, the input feature vector $\mathbf{x}_{i,t} \in \mathbb{R}^{16}$ is defined by 16 causal predictors structured into four distinct feature classes:

```
TABLE 2: Specification of the 16 Decision-Time Causal Features
========================================================================================================================
Feature Name          Class            Mathematical Definition / Source                 Temporal Availability
------------------------------------------------------------------------------------------------------------------------
store                 Identifier       Categorical store index s_i \in {1, ..., 10}     Static / Time-invariant
item                  Identifier       Categorical item index j_i \in {1, ..., 50}      Static / Time-invariant
year                  Calendar         Year of target date: Year(t+h)                   Deterministic future calendar
month                 Calendar         Month of target date: Month(t+h) \in {1, ..., 12} Deterministic future calendar
day                   Calendar         Day of target date: Day(t+h) \in {1, ..., 31}    Deterministic future calendar
weekday               Calendar         Day of week: Weekday(t+h) \in {0, ..., 6}        Deterministic future calendar
weekofyear            Calendar         ISO week number: Week(t+h) \in {1, ..., 53}      Deterministic future calendar
quarter               Calendar         Quarter of year: Quarter(t+h) \in {1, ..., 4}    Deterministic future calendar
is_weekend            Calendar         Indicator: \mathbb{I}(Weekday(t+h) \in {5, 6})   Deterministic future calendar
month_start           Calendar         Indicator: \mathbb{I}(Day(t+h) == 1)             Deterministic future calendar
month_end             Calendar         Indicator: \mathbb{I}(Day(t+h) == LastDayOfMonth) Deterministic future calendar
sales_lag_7           Autoregressive   Realized sales 7 days prior: y_{i,t-7}           Known at decision time t
sales_lag_14          Autoregressive   Realized sales 14 days prior: y_{i,t-14}         Known at decision time t
sales_lag_30          Autoregressive   Realized sales 30 days prior: y_{i,t-30}         Known at decision time t
rolling_mean_7        Rolling Window   \mu_{i,t-7}^{(7)} = (1/7) \sum_{j=7}^{13} y_{i,t-j} Known at decision time t
rolling_mean_30       Rolling Window   \mu_{i,t-7}^{(30)} = (1/30) \sum_{j=7}^{36} y_{i,t-j} Known at decision time t
========================================================================================================================
```

### Mathematical Proof of Temporal Isolation
Let $t$ be the decision time origin, and let $t+h$ be the target forecast date where $h \in \{1, 2, \dots, 7\}$. The earliest historical observation utilized across all autoregressive and rolling features is $y_{i, t - \tau}$ where $\tau \ge 7$. The temporal separation $\Delta \tau$ between the target date $t+h$ and the most recent utilized feature observation $t-7$ is given by:

$$\Delta \tau = (t + h) - (t - 7) = h + 7 \ge 1 + 7 = 8 \text{ days} \tag{2}$$

Because $\Delta \tau \ge 8 > 0$ for all $h \in \{1, \dots, 7\}$, no feature observation intersects with the forecast execution window $[t+1, t+h]$. Furthermore, the calendar features (e.g., `weekday`, `is_weekend`, `month`) are strictly deterministic functions of the future calendar date $t+h$ and contain zero demand realization leakage. Thus, the feature matrix $\mathbf{X}_t$ satisfies strict decision-time causality:

$$\mathbf{X}_t \in \sigma\left(\{y_{i,\tau}\}_{\tau \le t-7} \cup \{\text{Calendar}(t+k)\}_{k=1}^H\right) \subset \mathcal{F}_t \tag{3}$$

---

## 3.4 Decoupled Global Direct Multi-Horizon Gradient Boosting Architecture

To forecast the multi-step demand trajectory without suffering from autoregressive error accumulation, we deploy a **Decoupled Direct Multi-Horizon Architecture** \cite{tiwari2026explainable}. Rather than iteratively feeding single-step predictions back into the model, we train $H = 7$ independent global regression models, denoted by $M_1, M_2, \dots, M_7$.

Each model $M_h$ is parameterized as an Extreme Gradient Boosting (XGBoost) regressor \cite{chen2016xgboost} specialized to predict sales exactly $h$ steps ahead:

$$\hat{y}_{i,t+h} = M_h(\mathbf{x}_{i,t}; \mathbf{\theta}_h) \tag{4}$$

where $\mathbf{\theta}_h$ represents the learned tree parameters for horizon $h$, and $\mathbf{x}_{i,t}$ is the 16-dimensional causal feature vector available at time $t$.

### Global Tree Ensemble Formulation
Each horizon-specific model $M_h$ consists of an ensemble of $K_T = 100$ regression trees:

$$M_h(\mathbf{x}_{i,t}) = \sum_{k=1}^{K_T} f_k(\mathbf{x}_{i,t}), \quad f_k \in \mathcal{T} \tag{5}$$

where $\mathcal{T}$ is the space of Classification and Regression Trees (CART). The objective function minimized during training at horizon $h$ combines the mean squared error loss with structural model regularization:

$$\mathcal{L}(\mathbf{\theta}_h) = \sum_{i=1}^N \sum_{t \in \mathcal{T}_{\text{train}}} \left( y_{i,t+h} - \hat{y}_{i,t+h} \right)^2 + \sum_{k=1}^{K_T} \left( \gamma T_k + \frac{1}{2}\lambda \|\mathbf{w}_k\|^2 + \alpha \|\mathbf{w}_k\|_1 \right) \tag{6}$$

where $T_k$ is the number of leaf nodes in tree $k$, $\mathbf{w}_k$ is the leaf weight vector, $\gamma$ is the tree complexity penalty, $\lambda$ is the $L_2$ regularization parameter, and $\alpha$ is the $L_1$ regularization parameter.

The global models are trained simultaneously across all $N=500$ store-item series on the training partition (2013–2015). Hyperparameters were optimized on the 2016 validation set: `n_estimators = 100`, `max_depth = 6`, `learning_rate = 0.1`, `subsample = 0.8`, `colsample_bytree = 0.8`, `reg_alpha = 0.1`, `reg_lambda = 1.0`, and `random_state = 42`.

### Cumulative Lead-Time Forecast
At decision time $t$, the expected cumulative lead-time demand forecast for store-item $i$ over lead time $L \in \{1, \dots, 7\}$ is obtained by summing the point predictions of the first $L$ decoupled horizon models:

$$\hat{D}_{i,t,L} = \sum_{k=1}^L \hat{y}_{i,t+k} = \sum_{k=1}^L M_k(\mathbf{x}_{i,t}) \tag{7}$$

---

## 3.5 Rolling-Origin Out-of-Sample (OOS) Residual Generation Protocol

Standard conformal prediction requires a dedicated calibration dataset to compute empirical nonconformity scores \cite{vovk2005algorithmic}. Evaluating nonconformity on in-sample training residuals leads to severely under-dispersed intervals due to empirical overfitting. Conversely, holding out a single static validation year permanently reduces the volume of data available for model fitting.

To generate a realistic, out-of-sample residual pool without sacrificing training data, we implement a **Rolling-Origin Cross-Validation Protocol** using `TimeSeriesSplit` across the historical development dataset (2013–2016) \cite{huber2019integrating}.

```
Split 1: [ Train: 2013-2014 ] ---> [ OOS Predict: 2015 Fold 1 ]  --> Record OOS Residuals
Split 2: [ Train: 2013-2015 ] ---> [ OOS Predict: 2016 Fold 2 ]  --> Record OOS Residuals
...
Master Pool: Consolidated OOS Residual Dataset (N = 608,165 calibration observations)
```

For each time origin $r$ in the cross-validation folds, the cumulative lead-time forecast error (residual) for store-item $i$ over lead time $L$ is computed as:

$$e_{i,r,L} = y_{i,r,L} - \hat{D}_{i,r,L} = \sum_{k=1}^L y_{i,r+k} - \sum_{k=1}^L \hat{y}_{i,r+k} \tag{8}$$

### Decision-Time Eligibility Safeguard
When constructing the calibration pool at decision day $t$ in the forward test environment (2017), a historical residual $e_{i,r,L}$ is deemed **eligible** if and only if its lead-time evaluation window $[r+1, r+L]$ was completely finalized prior to day $t$:

$$\mathcal{E}_{i,t,L} = \left\{ r \in \mathcal{T}_{\text{hist}} : r + L < t \iff r \le t - L - 1 \right\} \tag{9}$$

Condition (9) guarantees that zero demand realizations from the ongoing forecast interval $[t, t+L]$ are included in the calibration set, preventing temporal leakage into the nonconformity quantile.

---

## 3.6 Rolling Forecast-Conditional Conformal Calibration

In retail portfolios, demand error distributions exhibit strong heteroskedasticity: high-volume SKUs exhibit substantially larger absolute forecast deviations than low-velocity items \cite{mitra2022comparative}. Sizing uncertainty intervals using a single global additive margin yields severe over-coverage on low-demand series and dangerous under-coverage on high-demand series.

To resolve this scale-mismatch, we formulate **Rolling Forecast-Conditional Conformal Calibration** (frozen in EXP-018), which normalizes nonconformity scores by the forecasted demand magnitude and extracts non-parametric quantiles over a localized rolling historical window.

### Step 1: Scale-Normalized Nonconformity Score
For any eligible historical calibration observation $r \in \mathcal{E}_{i,t,L}$, the scale-normalized nonconformity score $s_{i,r,L}$ is defined as the absolute relative forecast error:

$$s_{i,r,L} = \frac{|y_{i,r,L} - \hat{D}_{i,r,L}|}{\hat{D}_{i,r,L} + \epsilon} \tag{10}$$

where $\epsilon = 10^{-4}$ is a strictly positive numerical stabilization constant preventing division by zero during intermittent zero-demand periods. The score $s_{i,r,L}$ measures the relative percentage dispersion of actual demand around the point forecast.

### Step 2: Sliding Calibration Window Formulation
At decision day $t$, for each store-item series $i$ and lead time $L$, we extract the most recent $W = 100$ eligible historical nonconformity scores:

$$\mathcal{C}_{i,t,L} = \left\{ s_{i,r,L} : r \in \mathcal{E}_{i,t,L} \text{ and } r \ge t - L - W \right\}, \quad |\mathcal{C}_{i,t,L}| = W = 100 \tag{11}$$

If the available historical window contains fewer than $W_{\min} = 20$ observations (e.g., during the initial warm-up period), the pool falls back to the pooled cross-series calibration set for that demand regime.

### Step 3: Non-Parametric Quantile Extraction
Let $\beta \in (0, 1)$ denote the target nominal confidence level (e.g., $\beta \in \{0.80, 0.90, 0.95\}$). In classical split conformal prediction with finite calibration size $W$, the empirical quantile index is computed using the standard finite-sample correction \cite{vovk2005algorithmic}, \cite{angelopoulos2021gentle}:

$$k_{\beta} = \left\lceil \frac{(W + 1)\beta}{W} \right\rceil \tag{12}$$

The critical nonconformity quantile $q_{i,t,L}^{(\beta)}$ is extracted as the $k_{\beta}$-th order statistic from the sorted calibration set $\mathcal{C}_{i,t,L}$:

$$q_{i,t,L}^{(\beta)} = \text{Quantile}\left( \mathcal{C}_{i,t,L}, \frac{\lceil (W + 1)\beta \rceil}{W} \right) \tag{13}$$

### Step 4: Upper Prediction Bound and Uncertainty Buffer Formulation
Because inventory replenishment is an asymmetric single-sided risk problem—where stockouts occur exclusively when actual demand exceeds available stock—the operational safety buffer is derived from the **upper conformal prediction bound**:

$$\text{Upper}_{i,t,L}^{(\beta)} = \hat{D}_{i,t,L} \cdot \left( 1 + q_{i,t,L}^{(\beta)} \right) \tag{14}$$

The uncertainty-derived safety stock buffer $SS_{i,t,L}^{(\beta)}$ is formally defined as the non-negative spread between the upper conformal bound and the deterministic point forecast:

$$SS_{i,t,L}^{(\beta)} = \text{Upper}_{i,t,L}^{(\beta)} - \hat{D}_{i,t,L} = q_{i,t,L}^{(\beta)} \cdot \hat{D}_{i,t,L} \tag{15}$$

Equation (15) provides an intuitive and mathematically grounded property: the safety stock scales proportionally with both the expected demand volume $\hat{D}_{i,t,L}$ and the empirical relative forecast volatility $q_{i,t,L}^{(\beta)}$ observed over recent history.

---

## 3.7 Distribution Shift Diagnostic and Location-Scale Decomposition Framework

Conformal prediction guarantees exact finite-sample coverage $P(Y \in \Gamma(X)) \ge 1 - \alpha$ under the foundational assumption that calibration and test observations are **exchangeable** \cite{vovk2005algorithmic}. In retail demand applications, macroeconomic growth, trend shifts, and volatility shocks frequently violate exchangeability \cite{barber2023conformal}.

To systematically detect and diagnose nonstationarity between the historical calibration period (2016) and the forward operational test horizon (2017), we deploy a **Location-Scale and Violation-Symmetry Diagnostic Framework** (EXP-014).

### 1. Residual Variance Expansion Ratio
For each horizon $h \in \{1, \dots, 7\}$, the empirical residual variance expansion ratio $\rho_{\sigma^2}(h)$ measures the inflation of test error dispersion relative to validation error dispersion:

$$\rho_{\sigma^2}(h) = \frac{\text{Var}(e_{2017, h})}{\text{Var}(e_{2016, h})} = \frac{\frac{1}{N T_{2017}} \sum_{i,t} (y_{i,t+h} - \hat{y}_{i,t+h} - \bar{e}_{2017,h})^2}{\frac{1}{N T_{2016}} \sum_{i,t} (y_{i,t+h} - \hat{y}_{i,t+h} - \bar{e}_{2016,h})^2} \tag{16}$$

An expansion ratio $\rho_{\sigma^2} > 1.0$ indicates that forward test errors are significantly more dispersed than historical calibration errors, diagnosing an intrinsic temporal distribution shift.

### 2. Quarterly Signed Bias Tracking
To evaluate whether distribution shift is driven by systematic directional drift (location shift) or unmodeled volatility (scale shift), we compute the quarterly mean signed bias:

$$\text{Bias}_{Q}(h) = \frac{1}{|Q|} \sum_{t \in Q} \frac{1}{N} \sum_{i=1}^N \left( y_{i,t+h} - \hat{y}_{i,t+h} \right) \tag{17}$$

where $Q \in \{Q_1, Q_2, Q_3, Q_4\}$ partitions the 2017 test year into four 3-month quarters.

### 3. Interval Violation Asymmetry Ratio
Under a symmetric two-sided prediction interval $[\text{Lower}_{i,t}, \text{Upper}_{i,t}]$ at nominal miscoverage rate $\alpha$, exchangeability implies that upper and lower tail violations should be equally balanced: $\alpha_{\text{upper}} \approx \alpha_{\text{lower}} \approx \alpha / 2$. We measure violation asymmetry via:

$$\alpha_{\text{upper}} = \frac{1}{N T} \sum_{i=1}^N \sum_{t=1}^T \mathbb{I}\left( y_{i,t} > \text{Upper}_{i,t} \right) \tag{18}$$

$$\alpha_{\text{lower}} = \frac{1}{N T} \sum_{i=1}^N \sum_{t=1}^T \mathbb{I}\left( y_{i,t} < \text{Lower}_{i,t} \right) \tag{19}$$

$$\text{Asymmetry Ratio} = \frac{\alpha_{\text{upper}}}{\alpha_{\text{lower}}} \tag{20}$$

Pronounced deviation from parity ($\text{Asymmetry Ratio} \gg 1.0$) confirms asymmetric demand surges during the forward test period, providing clear empirical justification for why downstream inventory safety stock must focus explicitly on upper-tail quantile bounds.

---

## 3.8 Multi-Horizon Inventory Replenishment Policy Formulation

We formulate the store-level inventory control problem as a **Multi-Horizon Continuous-Review Order-Up-To Policy** $(ROP, S)$ operating under discrete daily reviews with non-zero deterministic supplier lead time $L$ and lost sales \cite{axsaeter2015inventory}, \cite{snyder2019fundamentals}.

At the beginning of each day $t$, the decision maker observes the starting on-hand inventory $OH_{i,t}^{\text{start}}$ and the pipeline of open orders in transit $OO_{i,t}$. The total **Inventory Position** $IP_{i,t}$ is defined as:

$$IP_{i,t} = OH_{i,t}^{\text{start}} + OO_{i,t} \tag{21}$$

where $OO_{i,t} = \sum_{k=1}^{L-1} Q_{i, t-k}$ represents all replenishment orders placed prior to day $t$ that have not yet arrived at the store.

We evaluate and compare two distinct replenishment policies:

### Policy A: Deterministic Point-Forecast Replenishment (Baseline)
Under Policy A, the reorder point $ROP_{i,t,L}^{\text{det}}$ is set strictly equal to the expected cumulative point forecast over lead time $L$, without safety stock:

$$ROP_{i,t,L}^{\text{det}} = \hat{D}_{i,t,L} = \sum_{k=1}^L \hat{y}_{i,t+k} \tag{22}$$

$$SS_{i,t,L}^{\text{det}} = 0 \tag{23}$$

Policy A represents the standard practice of risk-neutral machine learning forecasting, where point predictions are directly passed to supply chain execution systems without explicit uncertainty buffering.

### Policy B: Conformal Uncertainty-Aware Replenishment (Proposed)
Under Policy B, the reorder point $ROP_{i,t,L}^{\text{conf}}$ integrates the non-parametric conformal safety stock buffer $SS_{i,t,L}^{(\beta)}$ derived from the scale-normalized calibration layer:

$$ROP_{i,t,L}^{\text{conf}} = \hat{D}_{i,t,L} + SS_{i,t,L}^{(\beta)} = \hat{D}_{i,t,L} \cdot \left( 1 + q_{i,t,L}^{(\beta)} \right) = \text{Upper}_{i,t,L}^{(\beta)} \tag{24}$$

Under both policies, the replenishment order quantity $Q_{i,t}$ triggered on morning $t$ follows the standard order-up-to decision rule:

$$Q_{i,t} = \begin{cases} 
ROP_{i,t,L} - IP_{i,t}, & \text{if } IP_{i,t} < ROP_{i,t,L} \\ 
0, & \text{if } IP_{i,t} \ge ROP_{i,t,L} 
\end{cases} \tag{25}$$

---

## 3.9 Discrete-Event Inventory Simulation Engine and Daily Event Sequence

To evaluate the operational and financial impact of Policies A and B under realistic operating conditions, we developed a high-fidelity **Discrete-Event Simulation Engine** executed across all $N=500$ store-item series over the full 365 days of the 2017 out-of-sample test partition.

The simulation strictly enforces an **8-step daily discrete-event timing sequence** that governs physical inventory transitions, demand fulfillment, lost sales tracking, and financial cost settlement without intra-day ambiguity:

```
DAILY DISCRETE-EVENT TIMING SEQUENCE (DAY t)
========================================================================================
[ Step 1: Morning Delivery Arrival ] 
  OH_{i,t}^{start} = OH_{i,t-1}^{close} + Arrivals_{i,t}   (Arrivals from order placed at t - L)
        |
        v
[ Step 2: Inventory Position Computation ]
  IP_{i,t} = OH_{i,t}^{start} + OO_{i,t}                   (On-hand plus active pipeline stock)
        |
        v
[ Step 3: Morning Reorder Point Evaluation ]
  Compute ROP_{i,t,L} using morning forecast and eligible conformal quantile
        |
        v
[ Step 4: Replenishment Order Decision ]
  Q_{i,t} = max(0, ROP_{i,t,L} - IP_{i,t})
        |
        v
[ Step 5: Order Dispatch & Pipeline Update ]
  If Q_{i,t} > 0: Schedule delivery for day t + L; Update pipeline OO_{i,t} <- OO_{i,t} + Q_{i,t}
        |
        v
[ Step 6: Customer Demand Realization ]
  Observe actual customer demand realization y_{i,t} during retail business hours
        |
        v
[ Step 7: Demand Fulfillment & Lost Sales Allocation ]
  Fulfilled_{i,t} = min(OH_{i,t}^{start}, y_{i,t})
  Shortage_{i,t}  = max(0, y_{i,t} - OH_{i,t}^{start})
  OH_{i,t}^{close} = max(0, OH_{i,t}^{start} - y_{i,t})   (Lost sales dropped; no backlogging)
        |
        v
[ Step 8: Day-End Financial Settlement ]
  Compute Holding Cost:   C_{hold, i, t}  = h * OH_{i,t}^{close}
  Compute Ordering Cost:  C_{order, i, t} = K_{order} * \mathbb{I}(Q_{i,t} > 0)
  Compute Shortage Cost:  C_{short, i, t} = p * Shortage_{i,t}
========================================================================================
```

### Initial Inventory and Boundary Safeguards
To guarantee realistic operational dynamics and eliminate boundary simulation artifacts:
1. **Pre-2017 Initial Inventory Initialization**: Initial on-hand inventory at $t=1$ is initialized strictly using historical daily sales volume computed across the pre-2017 training set (2013–2016):
   $$OH_{i,1}^{\text{start}} = L \cdot \mu_i^{\text{pre-2017}}, \quad \mu_i^{\text{pre-2017}} = \frac{1}{|\mathcal{T}_{\text{pre}}|} \sum_{\tau \in \mathcal{T}_{\text{pre}}} y_{i,\tau} \tag{26}$$
   Zero 2017 demand realizations were accessed to set initial stock.
2. **Pipeline Initialization**: The initial on-order pipeline is set to empty: $OO_{i,0} = 0$.
3. **End-of-Horizon Truncation Protection**: Orders placed within the final $L$ days of the test year ($t > T - L$) cannot physically arrive before the simulation terminates at $t=T$. To prevent unfulfilled pipeline penalization, the final $L$ calendar days are excluded from aggregate financial and service metrics, ensuring that every evaluated day operated with a fully active upstream pipeline.

---

## 3.10 Financial Cost Model and Operational Accounting Framework

The financial performance of the inventory system is evaluated through a total operational replenishment cost function comprising three standard commercial cost components:

$$C_{\text{total}} = \sum_{i=1}^N \sum_{t=1}^{T-L} \left( C_{\text{hold}, i, t} + C_{\text{order}, i, t} + C_{\text{short}, i, t} \right) \tag{27}$$

where:

1. **Holding Cost ($C_{\text{hold}}$)**: Incurred daily on closing on-hand stock:
   $$C_{\text{hold}, i, t} = h \cdot OH_{i,t}^{\text{close}} \tag{28}$$
   where $h = \$0.20$ per unit per day represents warehouse carrying, storage, insurance, and working capital costs.
2. **Fixed Ordering Cost ($C_{\text{order}}$)**: Incurred each time a replenishment purchase order is transmitted:
   $$C_{\text{order}, i, t} = K_{\text{order}} \cdot \mathbb{I}(Q_{i,t} > 0) \tag{29}$$
   where $K_{\text{order}} = \$10.00$ per order captures electronic data interchange (EDI), administrative processing, and transportation fixed overhead.
3. **Shortage / Lost Sales Penalty ($C_{\text{short}}$)**: Incurred on every unit of unfulfilled customer demand:
   $$C_{\text{short}, i, t} = p \cdot \text{Shortage}_{i,t} = p \cdot \max\left(0, y_{i,t} - OH_{i,t}^{\text{start}}\right) \tag{30}$$
   where $p$ represents the unit gross margin loss, brand goodwill impairment, and customer defection cost. The primary baseline configuration fixes $p = \$1.00/\text{unit}$. To evaluate policy robustness across diverse retail operating environments, a comprehensive sensitivity grid is evaluated across $p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$ per unit.

---

## 3.11 Operational Evaluation Metrics and Service Level Formulations

To provide a multidimensional assessment of forecasting accuracy, uncertainty calibration, service quality, and operational cost efficiency, we compute the following formal metrics:

### 1. Demand-Weighted Portfolio Fill Rate
The primary service level metric in retail operations is the demand-weighted fill rate, representing the fraction of total customer demand fulfilled directly from on-hand inventory:

$$\text{Fill Rate} = \frac{\sum_{i=1}^N \sum_{t=1}^{T-L} \text{Fulfilled}_{i,t}}{\sum_{i=1}^N \sum_{t=1}^{T-L} y_{i,t}} = 1 - \frac{\sum_{i=1}^N \sum_{t=1}^{T-L} \text{Shortage}_{i,t}}{\sum_{i=1}^N \sum_{t=1}^{T-L} y_{i,t}} \tag{31}$$

### 2. Day-Based Stockout Rate and OTIF Service Level
The portfolio stockout rate measures the proportion of store-item-days that experienced an inventory shortage:

$$\text{Stockout Rate} = \frac{1}{N(T-L)} \sum_{i=1}^N \sum_{t=1}^{T-L} \mathbb{I}\left( \text{Shortage}_{i,t} > 0 \right) \tag{32}$$

The complementary On-Time-In-Full (OTIF) day-based service level is defined as:

$$\text{Service Level} = 1 - \text{Stockout Rate} = \frac{1}{N(T-L)} \sum_{i=1}^N \sum_{t=1}^{T-L} \mathbb{I}\left( \text{Shortage}_{i,t} = 0 \right) \tag{33}$$

### 3. Average On-Hand Inventory Investment
The mean physical inventory carried across the retail network:

$$\overline{OH} = \frac{1}{N(T-L)} \sum_{i=1}^N \sum_{t=1}^{T-L} OH_{i,t}^{\text{close}} \tag{34}$$

### 4. Unit Normalized Cost Metrics
To evaluate economic efficiency relative to sales volume, we compute the total cost per demanded unit and cost per fulfilled unit:

$$\text{Cost per Demanded Unit} = \frac{C_{\text{total}}}{\sum_{i=1}^N \sum_{t=1}^{T-L} y_{i,t}} \tag{35}$$

$$\text{Cost per Fulfilled Unit} = \frac{C_{\text{total}}}{\sum_{i=1}^N \sum_{t=1}^{T-L} \text{Fulfilled}_{i,t}} \tag{36}$$

### 5. Winkler Score (Interval Loss)
The statistical efficiency of the conformal prediction intervals is assessed via the Winkler score \cite{gneiting2007strictly} at nominal miscoverage $\alpha = 1 - \beta$:

$$W_{\alpha}(l, u, y) = (u - l) + \frac{2}{\alpha}(l - y)\mathbb{I}(y < l) + \frac{2}{\alpha}(y - u)\mathbb{I}(y > u) \tag{37}$$

where $l = \text{Lower}_{i,t}$ and $u = \text{Upper}_{i,t}$. The Winkler score strictly penalizes wider intervals while applying heavy linear penalties to observations falling outside interval bounds.

---

## 3.12 Statistical Inference and Bootstrap Significance Protocol

To confirm that observed differences between Policy A (Deterministic) and Policy B (Conformal) are statistically significant and generalizable across the entire retail portfolio, we implement a rigorous statistical testing protocol comprising:

### 1. Paired Two-Sided Wilcoxon Signed-Rank Test
Because retail store-item demand metrics are non-normally distributed, skewed, and heteroskedastic across different store locations, parametric paired $t$-tests are methodologically inappropriate. We deploy the non-parametric **Paired Two-Sided Wilcoxon Signed-Rank Test** across the $N=500$ individual series metric differences \cite{mitra2022comparative}:

$$\Delta_i = \text{Metric}_i(\text{Policy B}) - \text{Metric}_i(\text{Policy A}), \quad i \in \{1, \dots, N\} \tag{38}$$

The Wilcoxon test ranks the absolute differences $|\Delta_i|$ and computes the test statistic $W^+ = \sum_{i: \Delta_i > 0} \text{Rank}(|\Delta_i|)$. The null hypothesis $H_0: \text{Median}(\Delta) = 0$ is evaluated against the two-sided alternative $H_1: \text{Median}(\Delta) \neq 0$ at significance threshold $\alpha = 0.01$.

### 2. Paired Bootstrap Confidence Intervals
To construct robust confidence intervals for mean portfolio improvements without parametric assumptions, we execute a **1,000-Iteration Paired Bootstrap Resampling Procedure**. 

In each bootstrap iteration $b \in \{1, \dots, B\}$ ($B=1,000$), a resampled cohort of $N=500$ store-item pairs is drawn with replacement:

$$\mathcal{S}^{(b)} = \{ i_1^{(b)}, i_2^{(b)}, \dots, i_N^{(b)} \}, \quad i_k^{(b)} \sim \text{Uniform}(\{1, \dots, N\}) \tag{39}$$

The bootstrap mean difference $\bar{\Delta}^{(b)}$ is computed for each resample:

$$\bar{\Delta}^{(b)} = \frac{1}{N} \sum_{k=1}^N \left( \text{Metric}_{i_k^{(b)}}(\text{Policy B}) - \text{Metric}_{i_k^{(b)}}(\text{Policy A}) \right) \tag{40}$$

The empirical 95% bootstrap confidence interval $[\Delta_{\text{lower}}, \Delta_{\text{upper}}]$ is extracted from the 2.5th and 97.5th percentiles of the bootstrap distribution $\{\bar{\Delta}^{(b)}\}_{b=1}^{1000}$. A performance improvement is declared statistically significant at the 95% level if the confidence interval strictly excludes zero.

### 3. Sliced Demand Regime Evaluation
To assess performance across different product velocity tiers, all 500 store-item pairs are partitioned into two volume regimes based on pre-2017 historical median demand:
- **Medium Demand Regime**: Series with pre-2017 median daily sales $\le 45$ units (250 series).
- **High Demand Regime**: Series with pre-2017 median daily sales $> 45$ units (250 series).
Separate metric aggregations are generated for both tiers to evaluate scale robustness.
