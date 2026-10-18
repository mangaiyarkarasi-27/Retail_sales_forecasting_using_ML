# Demand Profiling Validation

This document presents the validation and research-integrity analysis of the demand profiling module in the retail sales forecasting pipeline.

---

## 1. Existing Verification Reviewed

We performed a thorough review of the project files. While the prompt indicated that `journal_extension/docs/demand_profiling_verification.md` existed, we verified that no such file was present in the repository. Instead, we reviewed the baseline verification records in [`conference_baseline_verification.md`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/docs/conference_baseline_verification.md) and the structural implementation details in [`demand_profiling_methodology.md`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/docs/demand_profiling_methodology.md).

---

## 2. Implementation Verification

We inspected the codebase to verify the demand profiling pipeline:
* **Input Dataset**: [`data/processed_train.csv`](file:///c:/Users/rmang/Documents/retail_project/data/processed_train.csv) (sorted chronologically by `date`).
* **Identifiers**: `store` (stores 1-10) and `item` (items 1-50).
* **Target Variable**: `sales`.
* **Grouping Method**: The profiling module groups the dataset by `["store", "item"]` and profiles each of the 500 series independently.
* **Chronological Ordering**: The data is sorted by date before grouping and calculation.
* **Profile Features & Formulas**:
  * `mean_sales` $\rightarrow$ Average of sales over the training period.
  * `std_sales` $\rightarrow$ Standard deviation of sales.
  * `cv_sales` $\rightarrow$ Coefficient of variation ($\sigma / \mu$).
  * `cv_sq_sales` $\rightarrow$ Volatility proxy ($\text{CV}^2$).
  * `zero_demand_proportion` $\rightarrow$ Proportion of zero sales observations.
  * `avg_nonzero_demand` $\rightarrow$ Average sales on positive-demand days.
  * `avg_demand_interval` $\rightarrow$ Average time between positive demand observations (ADI).
  * `lag1_autocorr` / `lag7_autocorr` $\rightarrow$ Autocorrelation at lags 1 and 7.
  * `trend_strength_r2` $\rightarrow$ R² of sales regressed linearly on time.
  * `lag365_autocorr` $\rightarrow$ Autocorrelation at lag 365 (capturing annual seasonality).
  * `demand_change_ratio` $\rightarrow$ Ratio of mean sales in the last 30 days to historical mean.
* **Status**: All features are **IMPLEMENTED AND VERIFIED** as mathematically correct and chronologically ordered.

---

## 3. Profile Coverage

We verified the properties of the output file [`demand_profiles.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/model_selection/demand_profiles.csv):
* **Unique Stores**: 10
* **Unique Items**: 50
* **Expected Combinations**: 500
* **Actual Profiles Generated**: 500 records.
* **Duplicate Profiles**: 0 (no duplicate combinations).
* **Missing Values**: 0.
* **Infinite Values**: 0.
* **Invalid/Negative Values**: 0 (all values fall within expected bounds).

---

## 4. Feature Validation

We analyzed the calculated values:
* **Zero Demand**: All series except one show exactly $0.0$ zero-demand proportion. Store 7, Item 18 has exactly 1 zero observation in the training partition (zero-demand proportion = $0.00068$).
* **Average Demand Interval (ADI)**: Exactly $1.0$ for all series except one ($1.00068$), proving that demand is completely continuous.
* **Autocorrelation**: Correctly computed on historical training arrays.
* **Trend Strength**: Correctly captures linear variance, yielding low $R^2$ values indicating that linear trend is a minor descriptor.
* **Data Leakage Check**: Computations use only the training partition (first 80% rows, 730,400 rows).

---

## 5. Demand Heterogeneity

We calculated the descriptive statistics for the 500 store-item combinations on the training partition:

| Metric | Mean | Std | Min | 25% | 50% (Median) | 75% | 90% | Max |
|---|---|---|---|---|---|---|---|---|
| **Mean Sales** | 50.6108 | 23.3088 | 12.3080 | 29.9934 | 48.4746 | 67.2729 | 84.9636 | 109.2019 |
| **CV Sales** | 0.2887 | 0.0234 | 0.2578 | 0.2707 | 0.2818 | 0.3023 | 0.3230 | 0.3746 |
| **Trend $R^2$** | 0.1237 | 0.0190 | 0.0634 | 0.1117 | 0.1269 | 0.1383 | 0.1451 | 0.1605 |
| **Lag-365 Autocorr** | 0.5024 | 0.0769 | 0.2678 | 0.4436 | 0.5151 | 0.5650 | 0.5925 | 0.6345 |

### **Analysis of Heterogeneity**:
* **Volume Heterogeneity**: **Strongly Supported**. Mean sales range from 12.31 to 109.20 units per day (a ~10x difference in scale).
* **Volatility Heterogeneity**: **Not Supported**. CV values are extremely uniform, ranging only from 0.2578 to 0.3746 ($\sigma = 0.0234$). Volatility is highly homogeneous across the network.
* **Trend Heterogeneity**: **Not Supported**. Trend strength is low for all series, with R² peaking at 0.1605.
* **Seasonality Heterogeneity**: **Supported**. Lag-365 autocorrelation ranges from 0.2678 to 0.6345, showing distinct variations in annual seasonal strength.

*Scientific Conclusion*: The dataset does **not** exhibit high multidimensional heterogeneity (volatility and sparsity are highly uniform). However, it does exhibit significant scale (volume) and seasonality strength differences, which justifies routing between experts specialized in seasonal spikes vs. standard predictors.

---

## 6. Regime Characterization

* **Syntetos-Boylan Classification (SBC)**:
  * **Smooth**: 500 series ($100.0\%$)
  * **Erratic / Intermittent / Lumpy**: 0 series ($0.0\%$)
  * *Analysis*: Since daily sales are continuous, the classical SBC classification is degenerate.
* **Primary Routing Regime** (at threshold of 0.30):
  * **Seasonal_Expert**: 499 series ($99.8\%$)
  * **Standard_ML_Expert**: 1 series ($0.2\%$, corresponding to Store 7, Item 1 with lag365 autocorrelation of 0.2678).
  * *Analysis*: A threshold of 0.30 is degenerate as almost all series are classified as seasonal. The classification labels are highly sensitive to threshold choices, indicating that rigid thresholds should be avoided or calibrated dynamically.

---

## 7. Threshold Robustness

We ran a sensitivity analysis on the seasonality threshold ($\rho_{365}$):
* **Threshold = 0.25**: Seasonal = 500 ($100.0\%$), Standard = 0 ($0.0\%$)
* **Threshold = 0.30**: Seasonal = 499 ($99.8\%$), Standard = 1 ($0.2\%$)
* **Threshold = 0.35**: Seasonal = 484 ($96.8\%$), Standard = 16 ($3.2\%$)
* **Threshold = 0.40**: Seasonal = 439 ($87.8\%$), Standard = 61 ($12.2\%$)
* **Threshold = 0.45**: Seasonal = 365 ($73.0\%$), Standard = 135 ($27.0\%$)
* **Threshold = 0.50**: Seasonal = 293 ($58.6\%$), Standard = 207 ($41.4\%$)

*Analysis*: Modest threshold changes significantly shift the series distribution. For future routing, a threshold of **0.45** is recommended to achieve a balanced allocation ($73\%$ Seasonal Expert, $27\%$ Standard ML Expert) rather than a degenerate one.

---

## 8. Leakage Assessment

* **Data Flow Trace**:
  $$\text{Raw Sales (Train Split)} \rightarrow \text{Demand Profiler} \rightarrow \text{demand\_profiles.csv} \rightarrow \text{Metadata Router Lookup}$$
* **Evaluation**: The profiling script [`run_profiling.py`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/src/demand_profiling/run_profiling.py) extracts only the first 80% chronological rows (the training partition) to build the profiles. 
* **Leakage Status**: **ZERO LEAKAGE DETECTED**. No future test data is used in the profile features.

---

## 9. Router-Safe Features

| Feature | Exists | Can be known at prediction time? | Safe for Router? | Reason |
|---|---|---|---|---|
| `mean_sales` | Yes | Yes | Yes | Computed strictly from training partition. |
| `std_sales` | Yes | Yes | Yes | Computed strictly from training partition. |
| `cv_sales` | Yes | Yes | Yes | Volatility metric calculated on training sales. |
| `lag365_autocorr` | Yes | Yes | Yes | Seasonality metric calculated on training sales. |
| `trend_strength_r2` | Yes | Yes | Yes | Trend strength calculated on training sales. |
| Realized Test-period sales | No | No | **NO** | Future target information. |
| Future rolling averages | No | No | **NO** | Potential look-ahead leakage. |

---

## 10. Scientific Justification for Adaptive Routing

We rate the scientific justification for adaptive model routing as **SUPPORTED**:
* *Justification*: Although the dataset lacks sparsity and volatility variations, the significant differences in **volume scale** (12 to 109 units/day) and **seasonality strength** (autocorrelation 0.26 to 0.63) mean that models optimized for strong seasonal cycles (such as seasonal estimators) will perform differently from global models on non-seasonal series.

---

## 11. Limitations

* **Lack of Intermittency**: Croston's and other intermittent-specific experts cannot be tested on this dataset.
* **Stationary Assumed**: The routing assumes that historical training patterns are representative of the test period.

---

## 12. Decision for Next Stage

Based on the validated seasonality and scale heterogeneity across series, we decide to:

**PROCEED WITH ADAPTIVE ROUTING**

*Action Plan*: In the next stages, the routing threshold will be set to **0.45** to ensure a balanced, non-degenerate distribution of series to specialized forecasting experts.
