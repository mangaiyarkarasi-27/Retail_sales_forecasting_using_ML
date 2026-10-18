# Inventory Methodology Correction

This document details the critical corrections applied to the inventory decision support methodology of the journal extension.

---

## 1. Original Issue

The previous inventory plan and leakage audit assumed that the daily lead-time residual originating at day $t-1$ was fully available for conformal calibration at decision time $t$ for all lead times $L$.

---

## 2. Why It Was Incorrect

A cumulative lead-time demand residual originating at day $r$ is:
$$E_{r, L} = \left| \sum_{k=1}^L y_{r+k} - \sum_{k=1}^L \hat{y}_{r+k} \right|$$
This residual requires observing actual sales up to day $r+L$. 

If $L = 3$ and the current day is $t$:
* The residual originating at $t-1$ requires observing actual sales $y_t, y_{t+1}, y_{t+2}$.
* Slicing residuals up to $t-1$ at day $t$ introduces look-ahead leakage, as it uses sales demand observations that have not yet occurred relative to the decision origin.

---

## 3. Corrected Temporal Logic

For a decision made at the beginning of day $t$ (before today's demand $y_t$ occurs, meaning $y_{t-1}$ is the latest observed demand value), a residual originating at $r$ is eligible for conformal calibration if and only if:
$$r + L < t$$
(or $r \le t - L - 1$).

### **Conformal Calibration Windows by Lead Time**
* **$L=1$**: Eligible residuals originate at $[t-101, t-2]$ (latest is $t-2$).
* **$L=3$**: Eligible residuals originate at $[t-103, t-4]$ (latest is $t-4$).
* **$L=5$**: Eligible residuals originate at $[t-105, t-6]$ (latest is $t-6$).
* **$L=7$**: Eligible residuals originate at $[t-107, t-8]$ (latest is $t-8$).

---

## 4. Initial Inventory Decision

We define a deterministic, training-set-derived initialization rule to ensure a fair starting point for both policies:
$$OH_1 = L \times \mu$$
where $\mu$ is the daily mean demand in the training partition. This represents the expected lead-time demand and avoids any policy optimization bias.

---

## 5. Inventory Position Terminology

We corrected the terminology to align with supply chain theory:
* **On-Hand Stock ($OH_t$)**: Physical stock available to meet demand.
* **On-Order Stock ($OO_t$)**: Outstanding orders scheduled to arrive.
* **Inventory Position ($IP_t$)**: $OH_t + OO_t$.
The reorder decision is triggered strictly when $IP_t < s_t$, whereas physical demand fulfillment is constrained to $OH_t$.

---

## 6. Forecast Availability

The XGBoost model utilizes lag features (minimum lag is 7 days). This means that for any lead time $L \le 7$, all input features are fully realized and known at day $t$, ensuring point forecasts are theoretically available without look-ahead feature leakage.

---

## 7. Scientific Impact

1. **Elimination of Leakage**: Enforcing the $r \le t - L - 1$ boundary guarantees that conformal prediction intervals are mathematically leakage-free during rolling simulations.
2. **True Out-of-Sample Evaluation**: Bypassing in-sample residual and future feature look-ahead biases ensures that the simulated inventory savings are robust and reproducible.
