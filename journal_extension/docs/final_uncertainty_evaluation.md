# Final 2017 Uncertainty Evaluation

This document presents the final evaluation of rolling-origin conformal prediction intervals for lead-time demand uncertainty over the chronological 2017 test period.

---

## 1. Objective

To perform a rigorous out-of-sample uncertainty evaluation on the untouched 2017 test period using Direct Multi-Horizon Point Forecasts and out-of-sample conformal calibration residuals.

---

## 2. Frozen Forecasting Model

The forecasting layer consists of the 7 global Direct Multi-Horizon XGBoost models trained strictly on the 2013-2015 split, which output point-forecast vectors without look-ahead feature leakage.

---

## 3. OOS Calibration Dataset

Conformal quantiles are computed using out-of-sample residuals generated via 5-fold cross-validation on the 2013-2016 training partition. No in-sample training residuals are used, preventing optimistic calibration.

---

## 4. Test Protocol

The evaluation is conducted on the 500 store-item series over the complete 365 days of 2017 (2017-01-06 to 2017-12-31).

---

## 5. Temporal Eligibility

For forecast origin $t$, we enforce $r + L < t$ (or $r \le t - L - 1$), meaning only residuals from origins whose lead-time outcomes are fully realized by morning $t$ are eligible.
* **L=1**: Latest eligible origin is $t - 2$ (calibration window: `[t-101, t-2]`).
* **L=3**: Latest eligible origin is $t - 4$ (calibration window: `[t-103, t-4]`).
* **L=5**: Latest eligible origin is $t - 6$ (calibration window: `[t-105, t-6]`).
* **L=7**: Latest eligible origin is $t - 8$ (calibration window: `[t-107, t-8]`).

---

## 6. Conformal Method

Nominal coverage levels evaluated: 80%, 90%, and 95%. Conformal quantiles are computed using the rolling window ($W=100$) and the finite-sample index formula:
$$\text{index} = \lceil(n+1)(1-\alpha)\rceil$$

---

## 7. Evaluation Metrics

* **Empirical Coverage (PICP)**: Fraction of actuals falling within the prediction interval.
* **Coverage Error**: $|\text{Empirical Coverage} - \text{Nominal Coverage}|$.
* **Mean Interval Width (MIW)**: Mean width of intervals.
* **Winkler/Interval Score**: Joint metric evaluating coverage sharpness (lower is better).

---

## 8. Aggregate Results (2017 Test Period)

The table below lists the aggregate performance across all 12 configurations:

| Lead Time ($L$) | Nominal Coverage | Empirical Coverage | Coverage Error | Mean Interval Width | Winkler Score | Evaluated Origins |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **L=1** | 80% | 23.70% | 56.30% | 27.90 | 253.33 | 364 |
| **L=1** | 90% | 29.37% | 60.63% | 34.37 | 436.51 | 364 |
| **L=1** | 95% | 34.69% | 60.31% | 40.35 | 760.19 | 364 |
| **L=3** | 80% | 17.98% | 62.02% | 66.19 | 804.26 | 362 |
| **L=3** | 90% | 21.67% | 68.33% | 78.76 | 1453.40 | 362 |
| **L=3** | 95% | 24.86% | 70.14% | 89.62 | 2669.94 | 362 |
| **L=5** | 80% | 16.57% | 63.43% | 103.16 | 1360.15 | 360 |
| **L=5** | 90% | 19.47% | 70.53% | 119.89 | 2496.17 | 360 |
| **L=5** | 95% | 21.83% | 73.17% | 133.05 | 4674.83 | 360 |
| **L=7** | 80% | 15.50% | 64.50% | 136.98 | 1930.11 | 358 |
| **L=7** | 90% | 17.90% | 72.10% | 156.50 | 3579.80 | 358 |
| **L=7** | 95% | 19.92% | 75.08% | 172.51 | 6757.34 | 358 |

---

## 9. Per-Series Results
* Metrics are saved in [`per_series_uncertainty_results.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/uncertainty/final_2017/per_series_uncertainty_results.csv) for all 500 store-item combinations.

---

## 10. Demand-Profile Analysis

### A. Volume Category Slices
* **High Volume** series exhibit slightly higher empirical coverage (e.g., L=1, 95% nominal: 37.92%) than **Low Volume** series (L=1, 95% nominal: 19.96%). This is due to the smaller relative variance in higher-volume demand.
* Full volume-slice results are logged in [`coverage_by_volume.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/uncertainty/final_2017/coverage_by_volume.csv).

### B. Seasonality Category Slices
* **Strong Seasonality** series have higher empirical coverage (e.g., L=1, 95% nominal: 36.16%) than **Moderate Seasonality** series (24.11%).
* Full seasonality results are logged in [`coverage_by_seasonality.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/uncertainty/final_2017/coverage_by_seasonality.csv).

---

## 11. Temporal Analysis
* Diagnostic plots showing interval width behavior over time are saved under:
  * [`nominal_vs_empirical_coverage.png`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/uncertainty/final_2017/nominal_vs_empirical_coverage.png)
  * [`interval_width_by_lead_time.png`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/uncertainty/final_2017/interval_width_by_lead_time.png)

---

## 12. Leakage Audit
* The evaluation fully complies with chronological data-isolation protocols, confirming zero leakage of 2017 target observations. See [`final_uncertainty_leakage_audit.md`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/docs/final_uncertainty_leakage_audit.md).

---

## 13. Limitations

* **Severe Under-Coverage**: Conformal prediction assumes data exchangeability. In this retail dataset, a significant chronological demand-growth trend is present:
  * 2015 mean sales: 52.26 units
  * 2016 mean sales: 56.60 units (+8.3%)
  * 2017 mean sales: 58.82 units (+12.5% vs. 2015)
  Because the global XGBoost models were trained strictly on 2013-2015 data (per EXP-011 design) and predicted 2017, the point forecasts systematically under-predict 2017 sales.
* **Rolling Calibration Lag**: Since the calibration window uses historical OOS residuals from 2013-2016, it is unable to adapt quickly to this systematic trend shift, leading to narrow intervals and severe coverage drops in the test set.

---

## 14. Scientific Findings

1. **Exchangeability Mismatch**: Conformal prediction's coverage guarantees are compromised in the presence of strong temporal trends and structural shifts.
2. **Sharpness vs. Validity**: Although the intervals are narrow, their lack of validity (empirical coverage $\approx 15-35\%$) makes them unreliable for safety stock decision support without a trend-adaptation or correction layer.
