# SECTION 4 CLAIM AUDIT & SCIENTIFIC INTEGRITY GOVERNANCE (RECONCILED)

This document provides a line-by-line claim audit of all empirical observations, interpretations, and generalization boundaries presented in **Section 4 (Results and Discussion)** of the journal extension manuscript following the **JOURNAL-005A Numerical Reconciliation Audit**. Every assertion is verified against our frozen experimental artifacts (EXP-011 through EXP-020) and audited against the writing governance rules defined in `CLAIM_BOUNDARIES.md`.

---

## 1. Master Results Claim Audit Table

| Claim ID | Section | Claim Text / Excerpt | Classification | Source Experiment & Artifact | Epistemological Boundary Check |
| :---: | :---: | :--- | :---: | :--- | :---: |
| **R-01** | 4.2 | *"Direct XGBoost validation MAE remains stable across horizons h=1..7 (MAE: 6.0285 to 6.0518, Aggregate: 6.0388)."* | Observation | EXP-011, `forecasting_results.csv` | **PASSED** (Directly supported by 2016 validation metrics) |
| **R-02** | 4.2 | *"Decoupling horizon estimation avoids the compounding error propagation inherent in iterative recursive forecasting."* | Interpretation | EXP-011, `forecasting_results.csv` | **PASSED** (Qualitative operational interpretation) |
| **R-03** | 4.2 | *"Validation metrics reflect the decision-time-safe lag >= 7 protocol and cannot be directly compared to recursive single-step error metrics from prior conference literature."* | Generalization Boundary | EXP-011, `METHODOLOGY_FREEZE.md` | **PASSED** (Explicit protocol distinction preserved) |
| **R-04** | 4.3 | *"Raw rolling conformal calibration was leakage-safe, but empirical 2017 coverage fell substantially below nominal (17.90%–29.37% at nominal 90%)."* | Observation | EXP-013, `uncertainty_results.csv` | **PASSED** (Mandatory negative finding retained) |
| **R-05** | 4.3 | *"The experiment showed severe empirical coverage degradation when the 2017 evaluation distribution differed from the historical calibration regime."* | Interpretation | EXP-013, `uncertainty_results.csv` | **PASSED** (Reconciled empirical framing without theoretical overreach) |
| **R-06** | 4.4 | *"Residual standard deviation expanded by 3.55x to 4.30x and MAE expanded by 3.76x to 4.58x between historical calibration and 2017 test periods."* | Observation | EXP-014, `uncertainty_failure_diagnosis.md` | **PASSED** (Corrected from 'variance expansion' to exact scale metrics) |
| **R-07** | 4.4 | *"Lower and upper violation rates at L=1, 90% were nearly identical (35.31% lower vs 35.32% upper, total violations 70.63%)."* | Observation | EXP-014, `uncertainty_results.csv` | **PASSED** (Demonstrates bilateral dispersion surge) |
| **R-08** | 4.4 | *"This diagnosis is an empirical finding for the evaluated retail dataset; the observed 2017 behavior is inconsistent with the stability required for historical calibration distributions to remain representative."* | Generalization Boundary | EXP-014, Section 4.4 | **PASSED** (Proper epistemological qualification) |
| **R-09** | 4.5 | *"Trailing scale normalization narrowed intervals by 14%–17% but led to worse empirical 2017 coverage (15.23%–25.54% at 90% nominal)."* | Observation | EXP-015, `uncertainty_results.csv` | **PASSED** (Mandatory negative finding retained) |
| **R-10** | 4.5 | *"Scale normalization alone did not restore empirical coverage under forward distribution shift."* | Interpretation | EXP-015, `FINAL_SCIENTIFIC_VERDICT.md` | **PASSED** (Honest failure reporting) |
| **R-11** | 4.6 | *"Forecast-conditional calibration improved 2017 empirical coverage (18.91%–31.98%) and lowered Winkler loss across all horizons (p < 10^-27)."* | Observation | EXP-016/018, `uncertainty_results.csv` | **PASSED** (Directly supported by test metrics) |
| **R-12** | 4.6 | *"Absolute coverage remained far below nominal 90%; forecast conditioning represents an empirical improvement in interval scoring behavior, not a restoration of nominal coverage guarantees."* | Generalization Boundary | EXP-016/018, Section 4.6 | **PASSED** (No claim of restored nominal guarantees) |
| **R-13** | 4.6 | *"At L=1, 90%, High-demand coverage rose from 28.03% to 37.30%, while Low-demand coverage dropped from 26.57% to 22.04%."* | Observation | EXP-016, `demand_regime_results.csv` | **PASSED** (Mandatory negative sliced finding retained) |
| **R-14** | 4.7 | *"Calibration neighborhood sweep established a broad stability plateau across K in [30, 100] on 2016 validation data."* | Observation | EXP-018, `docs/local_conditional_robustness_018.md` | **PASSED** (Validation Winkler loss within 0.67%) |
| **R-15** | 4.7 | *"Frozen parameter K*=100 retains all W=100 observations within the rolling window and was verified on 2017 test set (<0.1% diff from K=50)."* | Observation | EXP-018, `uncertainty_results.csv` | **PASSED** (No misleading KNN claims; full rolling window) |
| **R-16** | 4.8 | *"Conformal replenishment improved demand-weighted fill rate by +7.29 to +10.64 percentage points across all evaluated lead times."* | Observation | EXP-019, `inventory_results.csv` | **PASSED** (Directly verified across 500 series) |
| **R-17** | 4.9 | *"Conformal safety stock reduced unfulfilled customer demand by 771k to 1.14M units annually (20.03% to 29.57% shortage reduction)."* | Observation | EXP-019, `inventory_results.csv` | **PASSED** (Directly supported lost-sales metrics) |
| **R-18** | 4.9 | *"In the evaluated discrete-event simulation, the uncertainty-aware policy produced higher fill rates across evaluated lead times."* | Interpretation | EXP-019, Section 4.9 | **PASSED** (Proper simulation qualification used) |
| **R-19** | 4.10 | *"For L=1, conformal replenishment achieved lower total simulated operating cost in the evaluated scenario (-10.83%, -$790,834) at baseline penalty p=$1.00."* | Observation | EXP-019, `inventory_results.csv` | **PASSED** (Reconciled non-universal terminology) |
| **R-20** | 4.10 | *"For multi-day lead times L >= 3 at p=$1.00, holding cost increases exceeded shortage cost reductions, resulting in higher total simulated operating costs."* | Observation | EXP-019, `inventory_results.csv` | **PASSED** (Mandatory cost trade-off retained) |
| **R-21** | 4.10 | *"Under high shortage penalties (p >= $2.00), conformal replenishment achieved lower total operating costs across L=1, 3, 5."* | Observation | EXP-019, `cost_sensitivity_results.csv` | **PASSED** (Penalty-dependent cost preference verified) |
| **R-22** | 4.11 | *"Tuning nominal coverage beta in {80%, 90%, 95%} allows inventory planners to adjust the empirical trade-off between fill rate and inventory investment."* | Observation | EXP-019, `inventory_results.csv` | **PASSED** (Replaced 'strict monotonic controllability' with empirical trade-off) |
| **R-23** | 4.12 | *"The portfolio demand-weighted fill-rate improvement at L=1 is +10.64 pp, while the mean series-level paired fill-rate improvement is +10.36 pp (95% CI: [+10.19 pp, +10.55 pp])."* | Observation | EXP-019, `statistical_results.csv` | **PASSED** (Exact mathematical reconciliation verified) |
| **R-24** | 4.12 | *"Paired two-sided Wilcoxon signed-rank tests confirmed statistical significance (p < 10^-75 for service gains across all horizons)."* | Observation | EXP-019, `statistical_results.csv` | **PASSED** (Exact p-values reproduced) |
| **R-25** | 4.14 | *"The journal extension substantially expands the preliminary conference baseline across 14 distinct methodological dimensions."* | Implementation Fact | Table 9, `CONFERENCE_VS_JOURNAL.md` | **PASSED** (Methodological extension clearly established) |

---

## 2. Forbidden Terminology & Governance Audit (All Passed)

1. **"Causal" / "Causally"**: Replaced all instances asserting that forecasting methods "caused" real-world sales. Method is described as "decision-time-safe" or "causal feature protocol" (referring strictly to temporal filtration isolation).
2. **"Proved" / "Proving"**: Replaced with "demonstrated," "showed," or "diagnosed."
3. **"Guaranteed" / "Guarantees"**: Qualified to refer only to theoretical conformal properties under exchangeability, accompanied by the empirical finding that coverage guarantees failed under forward shift.
4. **"Universally Cheaper" / "Win-Win" / "Dominance"**: Completely removed. Replaced with "lower total simulated cost in the evaluated scenario," "cost-preferred under high shortage penalties," or "empirical service-cost trade-off."
5. **"Variance Expansion"**: Replaced with exact underlying metrics: "residual standard deviation expansion (3.55x to 4.30x)" and "mean absolute error expansion (3.76x to 4.58x)."
6. **"Strict Monotonic Controllability"**: Replaced with "Uncertainty-level sensitivity and empirical cost-service trade-off."
