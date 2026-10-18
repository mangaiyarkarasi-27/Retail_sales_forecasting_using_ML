# Inventory Simulation Methodology

This document details the objective, policy designs, order quantity derivation, shortage definitions, cost models, metrics, and experimental boundaries for the uncertainty-aware inventory simulation extension.

---

## 1. Objective

To evaluate whether integrating distribution-free predictive uncertainty intervals (via Rolling-Origin Conformal Calibration) into replenishment policies improves downstream operational performance (fill rates, stockout rates, carrying costs, and total supply chain costs) compared to a standard deterministic point-forecast baseline.

---

## 2. Baseline Policy

The deterministic point-forecast baseline policy uses the frozen XGBoost point forecasts without safety stock:
* **Safety Stock**: $SS_t^{\text{baseline}} = 0$
* **Reorder Point (ROP)**:
  $$s_t^{\text{baseline}} = \hat{D}_{t, L} = \sum_{k=1}^L \hat{y}_{t+k}$$
  where $\hat{D}_{t, L}$ is the point forecast of cumulative demand over lead-time $L$.

This provides a fair comparison as it uses the same demand data, the same point models, the same lead times, and the same ordering mechanism, with the only difference being the absence of the conformal uncertainty margin.

---

## 3. Proposed Uncertainty-Aware Policy

The proposed policy integrates the calibrated conformal quantile $q_{t, L}$ directly into the reorder logic:
* **Safety Stock**:
  $$SS_t^{\text{proposed}} = q_{t, L}$$
  where $q_{t, L}$ is the conformal quantile of cumulative lead-time demand error computed over the rolling 100-day window preceding $t$.
* **Reorder Point (ROP)**:
  $$s_t^{\text{proposed}} = \hat{D}_{t, L} + q_{t, L}$$

---

## 4. Inventory Terminology & State

At any day $t$, the simulator tracks the following distinct inventory variables:
1. **On-Hand Inventory ($OH_t$)**: The physical stock currently sitting on shelves, available to satisfy daily customer demand.
2. **On-Order Inventory ($OO_t$)**: Replenishment stock that has been ordered but not yet received.
3. **Inventory Position ($IP_t$)**: The sum of on-hand stock and on-order stock:
    $$IP_t = OH_t + OO_t$$
    *Note*: Reorder decisions are triggered based on the Inventory Position $IP_t$ falling below ROP ($s_t$). On-order stock cannot satisfy daily customer demand until it arrives and transitions to on-hand stock.

---

## 5. Decision Timing

To prevent look-ahead bias and off-by-one errors, the simulation sequence is executed in a strict chronological daily cycle:
1. **Morning (Receive Deliveries)**: On-hand inventory increases by orders placed $L$ days ago (which arrive at the start of day $t$).
2. **Morning (Evaluate Inventory Position)**: Compute available inventory position: $IP_t = OH_t + OO_t$.
3. **Morning (Replenishment Decision)**: Compare $IP_t$ to the reorder point $s_t$. If $IP_t < s_t$, trigger an order of size $Q$. Add $Q$ to $OO_t$. The order is scheduled for delivery on day $t+L$.
4. **Afternoon (Demand Occurs & Fulfillment)**: Customer demand $y_t$ is revealed. Fulfill from on-hand stock:
   $$\text{Sales}_t = \min(y_t, OH_t)$$
   $$\text{Shortage}_t = \max(0, y_t - OH_t)$$
5. **Night (Close Day)**: Update on-hand inventory for the next day: $OH_{t+1} = OH_t - \text{Sales}_t$. Compute daily holding, shortage, and ordering costs.

---

## 6. Lead-Time Handling

Replenishment lead time is constant: $L \in \{1, 3, 5, 7\}$ days.
* **Timeline Example for $L = 3$ days**:
  * **Day $t$ (Morning)**: Order of size $Q$ is placed. It is added to $OO$.
  * **Day $t$, $t+1$, $t+2$**: Order remains outstanding.
  * **Day $t+3$ (Morning)**: Order arrives at the start of the day and is added to $OH_{t+3}$, becoming available to satisfy Day $t+3$ demand.

---

## 7. Initial Inventory

To ensure complete fairness, both the baseline and proposed policies are initialized with identical starting values:
* **Initial On-Hand Stock ($OH_1$)**:
  $$OH_1 = L \times \mu$$
  where $\mu$ is the daily mean sales demand calculated strictly over the training partition.
* **Initial On-Order Stock ($OO_1$)**: $OO_1 = 0$.
* *Rationale*: Starting with the expected lead-time demand is a neutral initialization that avoids favoring either policy or cherry-picking initial values.

---

## 8. Forecast Availability

The point forecast $\hat{D}_{t, L} = \sum_{k=1}^L \hat{y}_{t+k}$ must be generated using only information available at day $t$.
* **Feature Extraction**: The XGBoost model utilizes lag features (minimum lag is 7 days, e.g. `sales_lag_7`) and rolling averages (shift-1 rolling mean).
* **Horizon Feasibility**: Because the minimum lag is 7 days, all features required to predict $\hat{y}_{t+1}, \dots, \hat{y}_{t+7}$ depend on sales observed up to day $t-1$ or earlier, which are fully known at decision time $t$. 
* *Deployment Note*: During simulation, predictions are drawn from pre-computed test tables generated using actual test feature records. In real-world deployment, recursive predictions would require recalculating rolling averages using model-generated feedback if lags were shorter than the forecast horizon.

---

## 9. Corrected Residual Availability

For lead time $L > 1$, the cumulative lead-time demand residual originating at day $r$ is:
$$E_{r, L} = \left| \sum_{k=1}^L y_{r+k} - \sum_{k=1}^L \hat{y}_{r+k} \right|$$
This residual requires observing actual sales up to day $r+L$. 
* **Eligibility Rule**: At decision time $t$ (morning), a residual originating at $r$ is eligible for inclusion in the rolling calibration window if and only if all $L$ sales observations have occurred, meaning:
  $$r + L < t$$
  (or $r \le t - L - 1$).
* **Calibration Window ($W = 100$)**: The conformal quantile $q_{t, L}$ is computed using the most recent 100 eligible residuals meeting this constraint. Residuals originating at $t - L$ to $t - 1$ are excluded as they require future unobserved demand.

---

## 10. Simulation Boundary Handling

* **Warm-up Period**: The first $L$ days of the test set are simulated normally using the initial stock $OH_1$.
* **End-of-Horizon Handling**: Replenishment orders are stopped when the remaining simulation days are less than $L$, as those orders would arrive after the simulation ends.

---

## 11. Order Quantity

We utilize the **Economic Order Quantity (EOQ)** model to determine replenishment lot sizes:
$$Q = \sqrt{\frac{2 \times K \times \mu}{h}}$$
where $K$ is the ordering cost ($\$10.00$), $h$ is the holding cost ($\$0.20$), and $\mu$ is the training-set daily mean sales. $Q$ is computed once prior to test simulation and remains constant.

---

## 12. Shortage Definition

We adopt a **Lost Sales** model:
$$\text{Shortage}_t = \max(0, y_t - OH_t)$$
Unfulfilled demand is lost immediately and does not accumulate.

---

## 13. Cost Model

Daily costs are computed as:
* Carrying Cost: $h \times OH_t^{\text{close}}$.
* Shortage Cost: $p \times \text{Shortage}_t$.
* Ordering Cost: $K \times \mathbb{I}(\text{Order Triggered})$.
$$\text{Total Cost}_t = C_{\text{hold}, t} + C_{\text{short}, t} + C_{\text{order}, t}$$

---

## 14. Service Metrics

Metrics evaluated over the 365-day test partition:
* **Fill Rate**: $1 - \frac{\sum \text{Shortage}_t}{\sum y_t}$.
* **Stockout Rate**: $\frac{1}{T} \sum \mathbb{I}(OH_t < y_t)$.
* **Average On-Hand Inventory**, **Average Safety Stock**, **Number of Orders**, and accumulated **Holding/Shortage/Total Costs**.

---

## 15. Sensitivity Analysis

We evaluate across:
* Lead times $L \in \{1, 3, 5, 7\}$ days.
* Shortage penalty ratios $p/h \in \{3, 5, 10, 20\}$ by varying $p \in \{0.60, 1.00, 2.00, 4.00\}$ with $h$ fixed at $\$0.20$.

---

## 16. Statistical Evaluation

We will perform a paired **Wilcoxon signed-rank test** across the 500 store-item combinations to compare the distribution of total simulated costs under both policies.
