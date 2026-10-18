# JOURNAL-005A: NUMERICAL RECONCILIATION & CLAIM CORRECTION AUDIT

**Audit Date**: 2026-09-19  
**Audit Target**: Section 4 Results and Discussion (`SECTION_4_RESULTS_DISCUSSION.md`), Traceability Matrix (`RESULTS_TRACEABILITY_MATRIX.csv`), and Claim Audit (`SECTION_4_CLAIM_AUDIT.md`)  
**Authoritative Source**: Frozen Experimental Artifacts EXP-011 through EXP-020  
**Audit Status**: **ALL DISCREPANCIES RECONCILED AND CORRECTED**

---

## 1. Executive Summary

A strict numerical and epistemological audit of the initial Section 4 draft was conducted against the frozen source artifacts in `journal_extension/results/final_audit/` and `journal_extension/docs/`. 

### Key Findings & Corrections:
1. **Critical Bootstrap CI Reconciliation**: The apparent discrepancy where the $L=1$ fill-rate improvement point estimate ($+10.64\text{ pp}$) lay outside the 95% bootstrap confidence interval ($[+10.19\text{ pp}, +10.55\text{ pp}]$) was resolved mathematically:
   - **Portfolio Demand-Weighted Fill-Rate Improvement**: **$+10.64\text{ pp}$** ($46.87\% \to 57.51\%$), which weights each unit of demand equally across the entire portfolio.
   - **Mean Series-Level Paired Fill-Rate Improvement**: **$+10.36\text{ pp}$** ($55.28\% \to 65.64\%$), which takes the arithmetic mean of paired series deltas. The 1,000-iteration bootstrap CI ($[+10.19\text{ pp}, +10.55\text{ pp}]$) is the confidence interval for this series-level mean delta. Both metrics are now explicitly defined and distinguished in the manuscript text.
2. **Residual-Scale Ratio Reconciliation**: The ratio $3.55\times \dots 4.58\times$ in EXP-014 represents the **residual standard deviation expansion** ($3.55\times$ to $4.30\times$) and **mean absolute error expansion** ($3.76\times$ to $4.58\times$), not a mathematical variance ratio. All references to "variance expansion" have been corrected to "residual-scale expansion" and "residual standard deviation expansion."
3. **Coverage Precision Reconciliation**: All coverage values across Raw, Scale-Normalized, Forecast-Conditional Hard ($32.58\%$), and Frozen $K^*=100$ ($31.98\%$) have been aligned to exact frozen artifact precision.
4. **Terminology Governance**: All unsupported universal phrasing ("win-win," "universally cheaper," "strict monotonic controllability," "provably superior") has been systematically replaced with precise empirical terminology.

---

## 2. Master Numerical Reconciliation Table

| Item | Initial Manuscript Value | Frozen Source Value | Status | Reconciliation Action Taken |
| :--- | :---: | :---: | :---: | :--- |
| **L1 Portfolio Demand-Weighted Fill Rate Delta** | +10.64 pp | **+10.64 pp** ($46.87\% \to 57.51\%$) | **VERIFIED** | Retained as portfolio demand-weighted metric in Table 6 and text. |
| **L1 Mean Series-Level Fill Rate Delta** | Not explicitly distinguished | **+10.36 pp** ($55.28\% \to 65.64\%$) | **RECONCILED** | Explicitly introduced to match the bootstrap sampling distribution. |
| **L1 1,000-Bootstrap 95% CI** | [+10.19 pp, +10.55 pp] | **[+10.19 pp, +10.55 pp]** | **VERIFIED** | Clarified as the 95% CI for the mean series-level paired improvement (+10.36 pp). |
| **Residual-Scale Ratio ($L=1$)** | 3.55x "variance expansion" | **3.55x** residual standard deviation ratio / **3.76x** MAE ratio | **CORRECTED** | Corrected nomenclature from "variance expansion" to "residual-scale / standard deviation expansion." |
| **Residual-Scale Ratio ($L=7$)** | 4.58x "variance expansion" | **4.30x** residual standard deviation ratio / **4.58x** MAE ratio | **CORRECTED** | Corrected nomenclature to exact std/MAE metrics. |
| **L1 90% Conformal Coverage (Hard Regime)** | 32.58% | **32.58%** | **VERIFIED** | Retained exact value for EXP-016 Hard Regime. |
| **L1 90% Conformal Coverage (Frozen $K^*=100$)** | ~32.0% (rounded) | **31.98%** | **CORRECTED** | Replaced rounded text with exact frozen value (31.98%). |
| **$K^*=100$ 2017 Test Winkler ($L=1$)** | 409.66 | **409.6602** | **VERIFIED** | Confirmed exact value in Table 4. |
| **$K^*=100$ vs $K=50$ 2017 Portfolio Winkler** | 2198.81 vs 2200.27 | **2198.81** vs **2200.27** (Diff: 1.46 / 0.066%) | **VERIFIED** | Confirmed difference is $<0.1\%$ supporting window robustness. |
| **L1 Cost Difference ($p=\$1.00$)** | -$790,834 (-10.8%) | **-$790,833.91** (**-10.83%**, \$7.31M to \$6.51M) | **VERIFIED** | Retained exact dollar delta and percentage. |
| **L3 Cost Difference ($p=\$1.00$)** | +$413,090 (+6.0%) | **+$413,089.61** (**+5.99%**, \$6.89M to \$7.31M) | **VERIFIED** | Retained exact dollar delta and percentage. |
| **L5 Cost Difference ($p=\$1.00$)** | +$1,434,777 (+16.8%) | **+$1,434,777.35** (**+16.83%**, \$8.53M to \$9.96M) | **VERIFIED** | Retained exact dollar delta and percentage. |
| **L7 Cost Difference ($p=\$1.00$)** | +$2,356,599 (+22.3%) | **+$2,356,598.91** (**+22.25%**, \$10.59M to \$12.95M) | **VERIFIED** | Retained exact dollar delta and percentage. |

---

## 3. Detailed Reconciliation Analysis

### 3.1 Bootstrap CI vs. Demand-Weighted Portfolio Fill Rate
- **Source Inspection**:
  - `inventory_results.csv` computes aggregate portfolio metrics:
    $$\text{Portfolio Fill Rate} = \frac{\sum_{i=1}^{500} \sum_{t=1}^{T} \text{Fulfilled}_{i,t}}{\sum_{i=1}^{500} \sum_{t=1}^{T} y_{i,t}} \implies \text{Det: } 46.87\%, \text{ Conf: } 57.51\% \implies \Delta = +10.64\text{ pp}$$
    $$\text{Mean Fill Rate Across Series} = \frac{1}{500} \sum_{i=1}^{500} \text{FillRate}_i \implies \text{Det: } 55.28\%, \text{ Conf: } 65.64\% \implies \Delta = +10.36\text{ pp}$$
  - `statistical_results.csv` executes 1,000 paired bootstrap resamples across the 500 store-item pairs, calculating the bootstrap distribution of the **mean paired series delta**:
    $$\bar{\Delta}^{(b)} = \frac{1}{500} \sum_{k=1}^{500} (\text{FillRate}_{i_k}(\text{Conf}) - \text{FillRate}_{i_k}(\text{Det}))$$
    $$\text{Mean Delta} = +0.103630 \implies +10.36\text{ pp}, \quad 95\% \text{ CI: } [+0.101878, +0.105537] \implies [+10.19\text{ pp}, +10.55\text{ pp}]$$
- **Resolution**: Both numbers are 100% correct within their respective definitions. The manuscript now explicitly reports both metrics and explains that the bootstrap CI evaluates the series-level mean delta ($+10.36\text{ pp}$).

### 3.2 Residual-Scale Ratio vs. Variance Ratio
- **Source Inspection**:
  - In `docs/uncertainty_failure_diagnosis.md` (Table 5), the ratios between 2017 test residuals and 2013–2016 calibration residuals are:
    - Standard Deviation Ratios: $L=1: 3.55\text{x}$, $L=3: 4.09\text{x}$, $L=5: 4.22\text{x}$, $L=7: 4.30\text{x}$.
    - Mean Absolute Error Ratios: $L=1: 3.76\text{x}$, $L=3: 4.35\text{x}$, $L=5: 4.49\text{x}$, $L=7: 4.58\text{x}$.
- **Resolution**: Calling $3.55\times \dots 4.58\times$ "variance expansion" was inaccurate. The text and figure specifications have been corrected to state "residual standard deviation expansion (3.55x to 4.30x)" and "mean absolute error expansion (3.76x to 4.58x)."

### 3.3 Nearest-Neighbor Nomenclature for $K^*=100$
- **Source Inspection**: The sliding calibration window size is $W=100$. The frozen parameter $K^*=100$ retains all $W=100$ historical observations within the window.
- **Resolution**: All descriptions of $K^*=100$ as a "strongly localized nearest-neighbor algorithm" have been eliminated. The method is designated as **rolling forecast-conditional conformal calibration** (or scale-normalized conformal calibration over rolling window $W=100$), with $K$-sweeps presented accurately as a robustness audit across neighborhood sizes.

---

## 4. Terminology and Governance Status

| Term / Phrase Evaluated | Initial Usage in Draft | Reconciled Status | Action Taken in Manuscript |
| :--- | :--- | :---: | :--- |
| "Win-Win" at $L=1$ | Present in Section 4.10 | **REMOVED** | Replaced with "lower total simulated cost in the evaluated scenario." |
| "Cost Dominance" | Present in Section 4.10 | **REPLACED** | Replaced with "cost-preferred under high shortage penalty settings." |
| "Strict Monotonic Controllability" | Present in Section 4.11 | **REPLACED** | Replaced with "Uncertainty-level sensitivity and empirical cost-service trade-off." |
| "Proving static calibration fails" | Present in Section 4.3 | **REPLACED** | Replaced with "The experiment showed severe empirical coverage degradation when the evaluation distribution differed." |
| "Non-exchangeable temporal shift" | Present in Section 4.3 | **REPLACED** | Replaced with "The observed behavior was inconsistent with the stability required for historical calibration distributions to remain representative." |
| "Zero-Leakage Proof" | Present in Section 4.2 | **QUALIFIED** | Replaced with "proof of temporal filtration isolation under lag $\ge 7$ protocol." |

---

## 5. Audit Conclusion

- **Total Numerical Discrepancies Identified**: 3 (Bootstrap CI metric distinction clarified, residual-scale ratio metric corrected, coverage precision aligned).
- **Total Terminology Adjustments**: 6 (All unhedged universal and causal phrasing removed).
- **Unresolved Values**: **0**.
- **Experimental Integrity**: All frozen experimental artifacts (EXP-011 through EXP-020) remain **100% intact, untouched, and un-modified**.
