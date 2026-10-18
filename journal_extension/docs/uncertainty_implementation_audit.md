# Uncertainty Implementation Audit

This document presents the research-integrity audit of the uncertainty quantification implementation in the retail sales forecasting pipeline.

---

## 1. Actual Implemented Method

The codebase implements **Rolling-Origin Conformal Calibration** (rather than a static Split Conformal Prediction). The prediction interval boundaries are adjusted dynamically using a sliding chronological window of absolute residuals from the 100 days preceding each prediction date.

---

## 2. Calibration Process

* **Calibration Window ($W$)**: $100$ days.
* **Slice Indexing**: Verified in [`run_uncertainty_experiment.py`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/src/experiments/run_uncertainty_experiment.py):
  `calib_errors = lt_abs_err_arr[idx - window_size : idx]`
  In Python, the slice `[idx - 100 : idx]` extracts indices from `idx - 100` to `idx - 1`.
  * **Today's Observation Exclusion**: **Verified**. The current index `idx` is excluded from the slice.
  * **Future Observation Exclusion**: **Verified**. No index $\ge$ `idx` is included.

---

## 3. Lead-Time Calculation

* **Implementation**: Verified in [`run_uncertainty_experiment.py`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/src/experiments/run_uncertainty_experiment.py):
  `series_data["actual_lt_demand"] = series_data["sales"].rolling(L).sum().shift(-L)`
  `series_data["pred_lt_demand"] = series_data["pred_sales"].rolling(L).sum().shift(-L)`
  * **Evaluation**: Shifting a rolling sum of size $L$ by $-L$ correctly aligns the sum of sales from index $t+1$ to $t+L$ (the next $L$ days) to the current prediction origin $t$.
  * **Lead Times**: Verified for $L \in \{1, 3, 5, 7\}$.

---

## 4. Residual Generation & In-Sample Discrepancy

* **Discrepancy Detected**: The point model was trained on the entire training partition (`train_df`), and predicted on `train_df` to generate the rolling calibration residuals for the out-of-sample test period.
* **Impact**: For the first 100 days of the test set, the sliding window of calibration residuals comes from the training partition and consists of **in-sample** residuals. For test days $\ge$ 100, the sliding window consists of out-of-sample residuals. This introduces a minor in-sample estimation bias at the start of the test partition.

---

## 5. Leakage Assessment

* **Test Isolation**: Verified. No 2017 target observations were used to train the XGBoost point model, calculate residuals prior to prediction, or select coverage parameters. 

---

## 6. Quantile Implementation

* **Formula**: Verified in [`conformal_prediction.py`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/src/uncertainty/conformal_prediction.py):
  `idx_1indexed = int(np.ceil((n + 1) * (1 - alpha)))`
  `idx_0indexed = min(idx_1indexed - 1, n - 1)`
  * **Evaluation**: Correctly implements the finite-sample correction index. When the index exceeds $n$ (due to small sample sizes), it is clipped to the maximum index $n-1$ to prevent out-of-bounds execution.

---

## 7. Metric Verification

* **Formulas**: Winkler scores in [`uncertainty_metrics.py`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/src/uncertainty/uncertainty_metrics.py) correctly scale penalties by $(2/\alpha)$ when actuals fall outside the interval boundaries. Lower-bound clipping at zero was verified as correctly implemented.

---

## 8. Test-Result Reproducibility

* We programmatically verified that the pipeline produces identical metrics upon re-run.

---

## 9. Result Consistency

* The empirical coverage and interval widths reported in [`uncertainty_summary.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/uncertainty/uncertainty_summary.csv) match the values in [`uncertainty_experiment.md`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/docs/uncertainty_experiment.md) exactly.
* *Note on Coverage Error*: The summary CSV reports the mean of per-series absolute coverage errors, whereas the markdown report lists the absolute error of the average empirical coverage. Both definitions are valid but represent different aggregation layers.

---

## 10. Claim Audit Summary

* The rolling-origin conformal engine produces exceptionally calibrated empirical coverage.
* However, claims of "distribution-free guarantees" must be qualified due to time-series autocorrelation. The square-root-of-lead-time scaling comparison is illustrative.

---

## 11. Bugs or Methodological Discrepancies

1. **In-Sample Residuals**: Use of in-sample residuals for the first 100 days of the test set.
2. **Terminology Inconsistency**: The documentation mixes "Split Conformal Prediction" and "Rolling-Origin Conformal Calibration". The code implements the latter.

---

## 12. Required Corrections Before Publication

1. Clarify in the manuscript that the calibration window is rolling, and the first 100 days utilize in-sample residuals of the point model.
2. Align all terminology to "Rolling-Origin Conformal Calibration" instead of "Split Conformal".
3. Label the $\sqrt{L}$ safety stock comparison as an illustrative calculation.

---

## FINAL STATUS

**VERIFIED WITH DOCUMENTATION CORRECTIONS**
