# JOURNAL CONTRIBUTIONS & VALUE PROPOSITION

This document formally articulates the **five primary scientific and methodological contributions** of the journal manuscript. Each contribution is directly backed by frozen experimental evidence from EXP-011 through EXP-020.

---

## Contribution 1: Causal, Leakage-Free Direct Multi-Horizon Retail Forecasting
> **Contribution Statement**:  
> *Develops and validates a strictly causal, direct multi-horizon gradient boosting architecture for multi-item retail sales forecasting that guarantees decision-time feature availability across all forecast horizons ($h=1,\dots,7$).*

- **Methodological Advance**: Unlike recursive single-step forecasting pipelines that risk autoregressive error accumulation and look-ahead lag leakage during multi-day inference, our framework decouples the multi-horizon forecasting problem into seven dedicated global XGBoost regressors ($M_1,\dots,M_7$).
- **Key Empirical Evidence**: Demonstrates stable validation performance (MAE: 6.0285 to 6.0518; aggregate MAE: 6.0388, RMSE: 7.8638) on the 2016 validation split, satisfying 10/10 automated temporal leakage audit tests with zero future target data entering the feature pipeline.

---

## Contribution 2: Rolling Forecast-Conditional Conformal Uncertainty Layer
> **Contribution Statement**:  
> *Designs and implements a rolling forecast-conditional conformal calibration layer that dynamically scales nonconformity scores by point-forecast magnitudes to generate distribution-free prediction bounds from out-of-sample calibration residuals.*

- **Methodological Advance**: Replaces rigid parametric Gaussian assumptions and unscaled raw conformal residuals with a scale-normalized nonconformity metric ($s_{r,L} = |y_{r,L} - \hat{D}_{r,L}| / (\hat{D}_{r,L} + \epsilon)$) evaluated over an eligible rolling-origin out-of-sample residual pool ($r \le t - L - 1$).
- **Key Empirical Evidence**: Significantly reduces mean Winkler interval loss from 2244.38 (Raw Conformal) to 2198.81 ($p < 0.02$) and expands high-volume demand interval widths to capture large demand spikes, boosting high-regime coverage by +8.8 percentage points.

---

## Contribution 3: Diagnostic Characterization of Conformal Calibration Under Temporal Shift
> **Contribution Statement**:  
> *Provides a comprehensive empirical diagnosis of conformal prediction behavior under real-world nonstationary retail demand, explaining the mechanism behind empirical coverage degradation.*

- **Methodological Advance**: Conducts a location-scale and violation-symmetry decomposition to demonstrate that empirical under-coverage in out-of-sample time series is driven by forward residual variance expansion (a 3.55x–4.58x increase in 2017) and seasonal phase misalignment, rather than simple point-forecast bias.
- **Key Empirical Evidence**: Identifies symmetric interval violations (35.3% lower vs. 35.3% upper violations at nominal 90%), establishing that conformal coverage guarantees are bounded by distribution exchangeability, and demonstrates that calibration-window performance exhibits a broad robustness plateau across neighborhood sizes $K \in [30, 100]$.

---

## Contribution 4: Uncertainty-Aware Inventory Decision Engine with Lost Sales
> **Contribution Statement**:  
> *Formulates and simulates an operational decision-support mechanism that directly translates forecast-conditional conformal uncertainty margins into dynamic safety stock buffers within a multi-horizon discrete-event replenishment environment.*

- **Methodological Advance**: Bridges the gap between predictive machine learning and prescriptive supply chain operations by coupling conformal upper prediction bounds to continuous-review reorder points ($ROP_{t,L} = \hat{D}_{t,L} + SS_{t,L}^{(\beta)}$), simulated under strict 8-step daily event timing and lost-sales dynamics.
- **Key Empirical Evidence**: Across all 500 store-item series on the untouched 2017 test set, uncertainty-aware replenishment achieves massive, statistically significant fill rate gains (+7.28 to +10.64 pp, paired Wilcoxon $p < 10^{-75}$) and cuts lost sales shortages by 20.0% to 29.6%, preventing up to 1.14 million units of unfulfilled customer demand per year.

---

## Contribution 5: Portfolio-Level Cost–Service Efficient Frontier Analysis
> **Contribution Statement**:  
> *Characterizes the portfolio-level economic and service trade-offs of uncertainty-aware replenishment across varying lead times ($L \in \{1, 3, 5, 7\}$) and asymmetric shortage penalties ($p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$).*

- **Methodological Advance**: Moves beyond simplistic single-cost evaluations to quantify how safety stock investments balance inventory holding, order placement, and stockout penalty costs across 500 heterogeneous retail series.
- **Key Empirical Evidence**: Proves that for single-day replenishment ($L=1$), uncertainty awareness is an unconditional economic win-win (fill rate +10.6 pp, total costs -10.8%, net savings of $\$790,834$, $p = 6.65 \times 10^{-55}$); for multi-day pipelines ($L \ge 3$), it establishes an actionable cost-service efficient frontier where conformal safety stock becomes total-cost dominant whenever shortage penalties exceed $\$2.00/\text{unit}$.
