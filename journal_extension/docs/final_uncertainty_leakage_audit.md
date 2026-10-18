# Final Uncertainty Leakage Audit

This document reports the leakage audit for the final 2017 out-of-sample uncertainty evaluation.

---

## 1. Audit Checklist

### 1.1. 2017 Actual Demand Isolation
* **Requirement**: 2017 actual demand must never be used before its evaluation point.
* **Verification**: In `run_final_uncertainty_evaluation.py`, the 2017 test set (`test_df`) is isolated. Its sales column is only accessed in the evaluation loop at index `t + 1 : t + L + 1` to verify if the actual cumulative demand lies within the bounds. It is never used in feature engineering, model training, or conformal calibration.
* **Verdict**: **PASSED**.

### 1.2. Forecast Feature Safety
* **Requirement**: Forecast features at origin $t$ must use only information available at morning $t$.
* **Verification**: In `decision_time_features.py`, lags and rolling averages are calculated strictly using shifts of $\ge 1$ days relative to origin $t$, ensuring that only sales up to $t-1$ are read. Target-date calendar features are deterministic and known in advance.
* **Verdict**: **PASSED**.

### 1.3. Conformal Calibration Window Safety
* **Requirement**: Calibration residuals must satisfy $r + L < t$ (or $r \le t - L - 1$).
* **Verification**: In `run_final_uncertainty_evaluation.py`, the binary search for eligible residuals uses a cutoff date of $t - L - 1$ days, ensuring that no residual whose cumulative demand outcome is still unresolved is included.
* **Verdict**: **PASSED**.

### 1.4. Test set Independence
* **Requirement**: No 2017 outcomes may influence parameter tuning, model training, or quantile calculation.
* **Verification**: All model parameters and conformal window sizes ($W=100$) were frozen prior to 2017 evaluation. The conformal quantiles are computed solely from OOS calibration residuals generated on the training split (2013-2016).
* **Verdict**: **PASSED**.

---

## 2. Conclusion

The final uncertainty evaluation has zero look-ahead bias and fully complies with chronological data-isolation protocols.
