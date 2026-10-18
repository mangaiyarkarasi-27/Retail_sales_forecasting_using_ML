# Uncertainty Claim Audit

This document reviews and audits the claims made in the predictive uncertainty quantification reports of the retail sales forecasting journal extension.

---

## Supported Claims

1. **High Empirical Calibration Accuracy**: The rolling-origin conformal calibration engine matches nominal coverages (80%, 90%, 95%) within **1% or less** on out-of-sample test data.
2. **Dynamic Interval Width Slicing**: Prediction interval widths scale proportionally with demand scale (volume), producing wider intervals for high-volume series and narrower, sharper intervals for low-volume series.
3. **Chronological Leakage Prevention**: Future test target data was isolated and did not influence model training, residual sliding windows, or quantile estimations.

---

## Claims Requiring Qualification

1. **"Distribution-Free Guarantees" / "Formal Coverage Guarantees"**:
   * *Audit*: Standard conformal prediction requires the exchangeability of residuals to guarantee coverage. Time-series data violates exchangeability due to autocorrelation and seasonality.
   * *Correction*: We must qualify that the rolling-origin conformal calibrator behaves as a robust empirical interval estimator, but does not possess formal distribution-free finite-sample guarantees under temporal dependence.
2. **"Resolves Autocorrelation"**:
   * *Audit*: Direct calibration of cumulative demand residuals does not "resolve" autocorrelation in daily series. Rather, it bypasses the daily autocorrelation issue by shifting the calibration target to the aggregated lead-time demand volume.
   * *Correction*: State that direct calibration of cumulative demand bypasses the i.i.d. assumptions of daily errors.

---

## Unsupported Claims

1. **"Direct Calibration Proves Superior Sharpness over Heuristic"**:
   * *Audit*: The comparison of direct calibration (62.18 units at L=7) against $\sqrt{L}$ scaling (67.73 units) was an illustrative calculation based on scaling the L=1 width, not a fully implemented baseline experiment with out-of-sample coverage validation.
   * *Correction*: Identify this comparison as a theoretical, illustrative comparison rather than a fully executed baseline experiment.

---

## Recommended Terminology

* Use **"highly calibrated empirical coverage"** instead of "statistically guaranteed coverage".
* Use **"sliding-window residual quantile estimation"** instead of "formal distribution-free guarantees".
* Use **"illustrative comparison"** when referencing the $\sqrt{L}$ safety stock scaling calculation.
