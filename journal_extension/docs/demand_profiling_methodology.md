# Demand Profiling Methodology

This document outlines the design, features, equations, and implementation of the demand profiling and regime characterization module for the retail sales forecasting pipeline.

---

## 1. Purpose

The demand profiling module characterizes the statistical behavior of individual sales series (at the store-item level) prior to model routing. This characterization helps to identify structural properties—such as seasonality, trends, and sparsity—and enables risk-aware inventory routing decisions.

---

## 2. Input Dataset

* **Source File**: `data/processed_train.csv`.
* **Chronological Scope**: Training partition only (first 80% chronologically, corresponding to years 2013-01-01 to 2016-12-31, 730,400 rows).
* **Target Columns**: `store`, `item`, `sales`, and `date`.

---

## 3. Store-Item Grouping

The profiling calculations are performed independently for each `(store, item)` combination using pandas groupby operations:
```python
df.groupby(["store", "item"])
```
This isolates the demand profile of each product series in the retail network.

---

## 4. Calculated Profile Features

For each time series $Y = \{y_1, y_2, \dots, y_N\}$ of length $N$, the following features are calculated:

* **Mean Demand**: Average daily sales quantity.
* **Median Demand**: Median daily sales quantity.
* **Standard Deviation ($\sigma$)**: Standard deviation of daily sales.
* **Coefficient of Variation (CV)**: Standard deviation divided by mean, indicating relative volatility.
* **Coefficient of Variation Squared ($\text{CV}^2$)**: Volatility proxy used in standard classification schemes.
* **Zero-Demand Proportion ($p_0$)**: Proportion of days with zero sales.
* **Average Non-Zero Demand**: Mean of $y_t$ where $y_t > 0$.
* **Average Demand Interval (ADI)**: Average time elapsed between days with non-zero demand.
* **Lag-1 Autocorrelation ($\rho_1$)**: Correlation of the series with its 1-day lagged values.
* **Lag-7 Autocorrelation ($\rho_7$)**: Correlation of the series with its 7-day lagged values, tracking weekly cyclicality.
* **Trend Strength ($R^2_{\text{trend}}$)**: Coefficient of determination ($R^2$) obtained from regressing sales values against a linear time index.
* **Lag-365 Autocorrelation ($\rho_{365}$)**: Correlation of the series with its 365-day lagged values, tracking annual seasonality.
* **Recent-vs-Historical Demand Change**: Ratio of the mean of the last 30 days to the mean of all preceding days.

---

## 5. Mathematical Definitions

### **Coefficient of Variation (CV)**
$$\text{CV} = \frac{\sigma}{\mu} = \frac{\sqrt{\frac{1}{N}\sum_{t=1}^N (y_t - \mu)^2}}{\frac{1}{N}\sum_{t=1}^N y_t}$$

### **Average Demand Interval (ADI)**
Let $T = \{t \in \{1,\dots,N\} \mid y_t > 0\}$ be the set of time indices with positive demand, ordered chronologically. The ADI is defined as:
$$\text{ADI} = \frac{1}{|T|-1} \sum_{i=2}^{|T|} (t_i - t_{i-1})$$

### **Trend Strength ($R^2_{\text{trend}}$)**
Regressing demand on $t = 1, \dots, N$ yields predicted sales $\hat{y}_t = \beta_0 + \beta_1 t$. The strength is defined as:
$$R^2_{\text{trend}} = 1 - \frac{\sum_{t=1}^N (y_t - \hat{y}_t)^2}{\sum_{t=1}^N (y_t - \mu)^2}$$

---

## 6. Leakage Prevention

* **Strict Temporal Isolation**: All profiling statistics are computed **only** on the training partition (first 80% chronological split).
* **Metadata Mapping**: Profile statistics are saved as a static metadata lookup mapped by `(store, item)`. When the future adaptive router is implemented, it will lookup these pre-computed metrics without accessing any test-set target values ($y_{t}$ for $t \in \text{test}$).

---

## 7. Regime-Label Methodology

Series are classified into multi-attribute demand profiles using both standard and custom criteria:
* **Syntetos-Boylan Classification (SBC)**: Classifies series into one of four mutually exclusive bins based on $\text{ADI}$ and $\text{CV}^2$:
  * **Smooth**: $\text{ADI} \le 1.32$ and $\text{CV}^2 \le 0.49$.
  * **Erratic**: $\text{ADI} \le 1.32$ and $\text{CV}^2 > 0.49$.
  * **Intermittent**: $\text{ADI} > 1.32$ and $\text{CV}^2 \le 0.49$.
  * **Lumpy**: $\text{ADI} > 1.32$ and $\text{CV}^2 > 0.49$.
* **Overlapping Descriptors**: Each series is tagged with active descriptors:
  * `trending`: if $R^2_{\text{trend}} \ge \text{trend\_threshold}$.
  * `seasonal`: if $\rho_{365} \ge \text{seasonality\_threshold}$.
  * `volatile`: if $\text{CV}^2 > \text{cv\_sq\_threshold}$.
  * `sparse`: if $\text{ADI} > \text{adi\_threshold}$.
* **Primary Routing Label**: A deterministic hierarchy assigns a candidate expert path:
  1. `Seasonal_Expert` (if `seasonal` is active).
  2. `Trend_Expert` (if `trending` is active and `seasonal` is inactive).
  3. `Robust_Baseline_Expert` (if `volatile` or `sparse` are active).
  4. `Standard_ML_Expert` (otherwise).

---

## 8. Threshold Selection Methodology

* **ADI & CV²**: Thresholds ($\text{ADI} = 1.32$, $\text{CV}^2 = 0.49$) are based on standard literature conventions (Boylan and Syntetos).
* **Trend & Seasonality**: Set empirically in config as:
  * `trend_threshold`: 0.20.
  * `seasonality_threshold`: 0.30.
  These serve as research parameters and are loaded dynamically from `journal_config.yaml`.

---

## 9. Limitations

* **Stationarity Assumption**: Profiling assumes that demand profiles computed on the training set remain stable during the test period.
* **Low-Sparsity Dataset**: The Kaggle Store Item Demand dataset has continuous demand with virtually no zero-demand days (only 1 zero observation in 913,000 records). Consequently, all 500 series fall into the `Smooth` category under SBC classification.

---

## 10. Reproducibility Information

* **Configuration**: `journal_extension/configs/journal_config.yaml`.
* **Execution Script**: `journal_extension/src/demand_profiling/run_profiling.py`.
* **Tests**: `journal_extension/tests/test_demand_profiling.py`.
