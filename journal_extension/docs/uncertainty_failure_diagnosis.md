# EXP-014: Uncertainty Failure Diagnosis

This document presents a comprehensive diagnostic investigation into the empirical under-coverage observed in the frozen EXP-013 out-of-sample uncertainty evaluation on the 2017 test period.

---

## 1. Objective

To rigorously analyze and identify the empirical mechanisms underlying the under-coverage observed in EXP-013 (where 80–95% nominal intervals achieved only 15–35% empirical coverage) without modifying forecasting models, changing calibration parameters ($W=100$), or implementing adaptive tuning against the final test set.

---

## 2. Frozen EXP-013 Result

EXP-013 evaluated the Direct Multi-Horizon point models paired with rolling-origin conformal calibration on the untouched 2017 test set (500 store-item combinations, 365 days):
* **L=1**: 80% Nom $\rightarrow$ 23.70% Emp (Error: 56.30%), 90% Nom $\rightarrow$ 29.37% Emp, 95% Nom $\rightarrow$ 34.69% Emp.
* **L=3**: 80% Nom $\rightarrow$ 17.98% Emp (Error: 62.02%), 90% Nom $\rightarrow$ 21.67% Emp, 95% Nom $\rightarrow$ 24.86% Emp.
* **L=5**: 80% Nom $\rightarrow$ 16.57% Emp (Error: 63.43%), 90% Nom $\rightarrow$ 19.47% Emp, 95% Nom $\rightarrow$ 21.83% Emp.
* **L=7**: 80% Nom $\rightarrow$ 15.50% Emp (Error: 64.50%), 90% Nom $\rightarrow$ 17.90% Emp, 95% Nom $\rightarrow$ 19.92% Emp.

---

## 3. Point Forecast Bias

Evaluating daily point forecasts in 2017 across horizons $h=1 \dots 7$:
* **Mean Actual Demand**: 58.84 ($h=1$) to 59.17 ($h=7$) units.
* **Mean Prediction**: 56.88 ($h=1$) to 57.11 ($h=7$) units.
* **Mean Signed Error**: $+1.82$ to $+2.53$ units across horizons.
* **Bias Ratio ($\text{Mean Error}/\text{Mean Demand}$)**: $+3.09\%$ to $+4.27\%$.
* **MAE**: ~34.81 to 34.96 units.
* **RMSE**: ~43.89 to 44.12 units.

While the aggregate annual point bias is small positive (+3.1% to +4.3%), the point forecast dispersion in 2017 is substantial ($\text{MAE} \approx 34.8$).

---

## 4. Temporal Forecast Drift (Quarterly Breakdown)

Decomposing 2017 point forecast bias across predefined chronological quarters reveals severe seasonal phase bias:

| Quarter | Horizon ($h$) | Mean Actual | Mean Pred | Mean Error | Bias Ratio | MAE | RMSE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | $h=1$ | 46.05 | 64.90 | **-18.84** | **-40.92%** | 34.43 | 43.66 |
| **Q2** | $h=1$ | 66.64 | 56.35 | **+10.29** | **+15.44%** | 35.77 | 44.93 |
| **Q3** | $h=1$ | 67.92 | 46.97 | **+20.95** | **+30.85%** | 37.34 | 47.27 |
| **Q4** | $h=1$ | 54.52 | 59.48 | **-4.97** | **-9.11%** | 31.66 | 39.48 |

* **Observation**: Point models heavily over-predict in winter Q1 (Mean Error: $-18.84$) and heavily under-predict in peak summer Q2/Q3 (Mean Error: $+10.29$ in Q2, $+20.95$ in Q3).

---

## 5. Calibration vs 2017 Residual Distribution

Comparing the EXP-012 OOS calibration residuals (2013–2016) against 2017 test set residuals shows a massive scale and dispersion explosion:

| Lead Time ($L$) | Calib MAE | 2017 MAE | MAE Ratio | Calib Std | 2017 Std | Std Ratio | Calib P90 | 2017 P90 | P90 Ratio | Calib P95 | 2017 P95 | P95 Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L=1** | 9.26 | 34.81 | **3.76x** | 12.35 | 43.89 | **3.55x** | 20.64 | 73.53 | **3.56x** | 26.72 | 86.92 | **3.25x** |
| **L=3** | 23.80 | 103.57 | **4.35x** | 31.77 | 129.81 | **4.09x** | 54.76 | 218.15 | **3.98x** | 70.33 | 255.95 | **3.64x** |
| **L=5** | 38.35 | 172.28 | **4.49x** | 51.11 | 215.47 | **4.22x** | 89.18 | 361.90 | **4.06x** | 113.81 | 424.95 | **3.73x** |
| **L=7** | 52.75 | 241.41 | **4.58x** | 70.19 | 301.60 | **4.30x** | 123.09 | 505.92 | **4.11x** | 156.89 | 594.62 | **3.79x** |

---

## 6. Location vs Scale Shift

Quantitative assessment indicates that the distribution shift from the training era to 2017 is **primarily a massive Scale / Variance Shift accompanied by seasonal Location Swings**:
* **Scale Shift**: Residual standard deviation expanded by **3.55x to 4.30x**, and 90th percentile absolute error expanded by **3.56x to 4.11x**.
* **Location Shift**: While annual aggregate mean signed error remained modest ($+1.96$ for $L=1$), quarterly mean signed error swung violently between $-18.84$ (Q1) and $+20.95$ (Q3).
* **Diagnosis**: **Both scale explosion and seasonal phase mismatch** are simultaneously present.

---

## 7. Historical Residual Drift Across Training Years (2013–2016)

Examining the EXP-012 OOS calibration dataset by calendar year reveals that residual distributions were already non-stationary prior to 2017:

| Year | Lead Time ($L$) | Mean Signed Error | MAE | RMSE | Std Dev | P90 Abs Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2013** | $L=1$ | -7.64 | 9.86 | 12.28 | 9.61 | 20.07 |
| **2014** | $L=1$ | +5.93 | 9.17 | 12.79 | 11.34 | 21.19 |
| **2015** | $L=1$ | +1.72 | 8.46 | 11.53 | 11.40 | 18.31 |
| **2016** | $L=1$ | +5.09 | 9.97 | 13.89 | 12.92 | 22.71 |

* **Observation**: In the training period, MAE remained stable between 8.46 and 9.97, and std dev between 9.61 and 12.92. However, mean signed error drifted from $-7.64$ in 2013 to $+5.09$ in 2016, confirming systematic non-stationarity.

---

## 8. Conformal Quantile Adequacy vs Actual 2017 Errors

Comparing the rolling conformal quantile $q_{t,L}$ against actual 2017 absolute lead-time errors:

| Lead Time ($L$) | Nominal Coverage | Mean $q$ | Mean Actual Error | P90 $q$ | P90 Actual Error | Underestimation Rate (%) | Target Gap (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L=1** | 80% | 14.13 | 34.81 | 24.77 | 73.53 | **76.30%** | +56.30% |
| **L=1** | 90% | 17.52 | 34.81 | 29.89 | 73.53 | **70.63%** | +60.63% |
| **L=1** | 95% | 20.72 | 34.81 | 34.70 | 73.53 | **65.31%** | +60.31% |
| **L=3** | 90% | 39.89 | 103.57 | 75.07 | 218.15 | **78.33%** | +68.33% |
| **L=7** | 90% | 79.05 | 241.41 | 158.16 | 505.92 | **82.10%** | +72.10% |

* **Finding**: The conformal quantile $q_{t,L}$ systematically underestimated the actual 2017 lead-time error in **65% to 84%** of instances (compared to the nominal non-coverage target of 5% to 20%).

---

## 9. Lower vs Upper Interval Violations

Decomposition of interval failures into lower vs. upper bound breaches:

| Lead Time ($L$) | Nominal Coverage | Empirical Coverage | Lower Violation Rate | Upper Violation Rate | Total Violations |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **L=1** | 80% | 23.70% | **37.67%** | **38.64%** | 76.30% |
| **L=1** | 90% | 29.37% | **35.31%** | **35.32%** | 70.63% |
| **L=1** | 95% | 34.69% | **33.13%** | **32.18%** | 65.31% |
| **L=3** | 90% | 21.67% | **38.50%** | **39.84%** | 78.33% |
| **L=5** | 90% | 19.47% | **39.18%** | **41.35%** | 80.53% |
| **L=7** | 90% | 17.90% | **39.82%** | **42.29%** | 82.10% |

* **Crucial Finding**: Violation rates are **symmetrically split** (~35–40% lower violations and ~35–42% upper violations).
* If the under-coverage were purely caused by monotonic annual demand growth (one-sided under-prediction), upper violations would dominate (>70%) and lower violations would be near 0%. The near-equal split proves that **interval width collapse / scale underestimation** combined with seasonal oscillations is the dominant failure mode.

---

## 10. Demand-Volume Effects (Training-Derived Slices)

| Lead Time | Coverage | Volume Slice | Mean Actual | Mean Pred | Mean Signed Bias | Bias Ratio | Empirical Coverage | Mean Width | Mean Abs Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L=1** | 90% | **High** | 82.42 | 56.86 | +25.56 | +31.02% | **31.88%** | 45.98 | 39.22 |
| **L=1** | 90% | **Medium** | 39.80 | 56.86 | -17.05 | -42.84% | **28.81%** | 24.62 | 29.37 |
| **L=1** | 90% | **Low** | 20.00 | 57.12 | -37.12 | -185.59% | **16.68%** | 17.43 | 38.46 |

* **Observation**: Global point model predictions collapse toward the dataset mean (~57 units), heavily over-predicting Low Volume series (Mean Bias: $-37.12$) and under-predicting High Volume series (Mean Bias: $+25.56$). Consequently, Low Volume series suffer the lowest coverage (16.68%).

---

## 11. Seasonality Effects (Training-Derived Slices)

| Lead Time | Coverage | Seasonality Slice | Mean Actual | Mean Pred | Mean Signed Bias | Bias Ratio | Empirical Coverage | Mean Width | Mean Abs Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L=1** | 90% | **Moderate** | 22.78 | 57.01 | -34.24 | -150.32% | **20.75%** | 19.67 | 36.47 |
| **L=1** | 90% | **Strong** | 63.85 | 56.86 | +6.99 | +10.95% | **30.56%** | 36.41 | 34.57 |

* **Observation**: Moderate seasonality series (which correlate with lower mean volume) exhibit massive negative bias (over-prediction) and lower empirical coverage.

---

## 12. Series-Level Heterogeneity Across 500 Store-Items

| Lead Time | Nominal Cov | Mean Cov | Median Cov | P25 Cov | P75 Cov | Min Cov | Max Cov | % Series Within $\pm 5\%$ | % Series Within $\pm 10\%$ | % Series Below 50% |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L=1** | 80% | 23.70% | 22.80% | 19.45% | 27.75% | 5.75% | 54.95% | **0.0%** | **0.0%** | **99.8%** |
| **L=1** | 90% | 29.37% | 28.57% | 25.21% | 33.99% | 8.24% | 57.97% | **0.0%** | **0.0%** | **99.4%** |
| **L=1** | 95% | 34.69% | 34.29% | 29.54% | 40.38% | 9.62% | 61.26% | **0.0%** | **0.0%** | **96.0%** |
| **L=3** | 90% | 21.67% | 20.99% | 16.30% | 26.52% | 4.42% | 54.70% | **0.0%** | **0.0%** | **99.8%** |
| **L=7** | 90% | 17.90% | 17.32% | 12.00% | 22.91% | 3.06% | 48.04% | **0.0%** | **0.0%** | **100.0%** |

* **Pervasive Failure**: Under-coverage is not an isolated artifact of a few outlier series; **99.4% to 100% of all 500 series** achieved empirical coverage below 50% for 90% nominal intervals.

---

## 13. Calibration Window Analysis

The 100-day rolling calibration window ($W=100$) relies strictly on eligible historical OOS residuals originating prior to $t - L - 1$.
* In 2017, the calibration window draws almost entirely from the low-dispersion 2013–2016 historical residuals ($\text{std} \approx 12.35$).
* Because the true 2017 forecast error dispersion exploded to $\text{std} \approx 43.89$, the static historical window $W=100$ produces quantiles $q_{t,L}$ that are roughly $3.5\times$ to $4.5\times$ too narrow.

---

## 14. Evidence-Based Interpretation

* **Hypothesis Evaluation**: The previous hypothesis attributed coverage collapse solely to annual demand growth.
* **Empirical Evidence**:
  1. The annual aggregate point bias in 2017 is small ($+3.34\%$ at $h=1$).
  2. The interval violation breakdown is **symmetric** (~35% lower vs ~35% upper violations at 90% nominal).
  3. The error scale ($\text{std}$ and $\text{MAE}$) increased by **$3.5\times$ to $4.6\times$**.
* **Scientific Conclusion**: The empirical coverage failure is **consistent with a dual mechanism**:
  - **Primary driver**: Severe **scale / dispersion underestimation**, where rolling calibration quantiles based on historical training folds are $3.5\times$ too narrow for the 2017 forecast error variance.
  - **Secondary driver**: **Seasonal phase bias** (over-prediction in Q1, under-prediction in Q2/Q3), causing alternating lower and upper interval breaches across quarters.

---

## 15. Limitations

* EXP-014 is strictly a diagnostic investigation. No adaptive corrections or variance adjustments were applied.
* Residual distributions in 2017 reflect direct global point model behavior when evaluated on a disjoint chronological year.

---

## 16. Implications for Next Experiment

1. **Standard Conformal Assumptions Violated**: The exchangeability assumption between historical training-fold OOS residuals and out-of-sample test period residuals is severely violated in non-stationary retail time series.
2. **Need for Adaptive Uncertainty Calibration**: Uncertainty estimation in non-stationary retail demand requires:
   - Seasonality-aware and volume-scaled residual calibration.
   - Adaptive or locally scaled conformal prediction (e.g., normalized conformal prediction with variance scaling) to account for dispersion shifts across regimes.
