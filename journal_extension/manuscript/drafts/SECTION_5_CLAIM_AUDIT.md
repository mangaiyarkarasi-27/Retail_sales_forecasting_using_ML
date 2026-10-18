# SECTION 5 CLAIM AUDIT & EPISTEMOLOGICAL BOUNDARY GOVERNANCE

This document provides a line-by-line claim audit of all summary statements, empirical findings, research contributions, practical implications, limitations, and future directions presented in **Section 5 (Conclusion and Future Directions)** of the journal extension manuscript. Every assertion is verified against our frozen experimental artifacts (EXP-011 through EXP-020) and audited against the writing governance rules defined in `CLAIM_BOUNDARIES.md`.

---

## 1. Master Section 5 Claim Audit Table

| Claim ID | Section | Claim Text / Excerpt | Classification | Verification Source | Epistemological Boundary Check |
| :---: | :---: | :--- | :---: | :--- | :---: |
| **C-01** | 5.1 | *"The methodology integrates decision-time-safe direct multi-horizon forecasting, OOS residual pooling, forecast-conditional conformal calibration, shift diagnostics, robustness auditing, and discrete-event replenishment simulation."* | Directly Supported | EXP-011 to EXP-020, `METHODOLOGY_FREEZE.md` | **PASSED** (Accurately summarizes completed pipeline) |
| **C-02** | 5.2 (RQ1) | *"Seven direct XGBoost models achieved stable validation accuracy across h=1..7 under the minimum lag >= 7 protocol (Aggregate MAE: 6.0388, RMSE: 7.8638)."* | Directly Supported | EXP-011, `forecasting_results.csv` | **PASSED** (Verified exact validation metrics) |
| **C-03** | 5.2 (RQ1) | *"Decoupling horizon estimation eliminated compounding error accumulation while satisfying 10/10 automated leakage checks."* | Empirical Interpretation | EXP-011, `test_feature_leakage.py` | **PASSED** (Qualitative operational interpretation) |
| **C-04** | 5.2 (RQ2) | *"Raw rolling conformal calibration empirical coverage fell to 17.90%–29.37% at nominal 90% confidence on the 2017 test set."* | Directly Supported | EXP-013, `uncertainty_results.csv` | **PASSED** (Mandatory negative result retained) |
| **C-05** | 5.2 (RQ2) | *"Diagnostics identified a 3.55x to 4.30x residual standard deviation expansion (and 3.76x to 4.58x MAE expansion) alongside quarterly seasonal shifts."* | Directly Supported | EXP-014, `uncertainty_failure_diagnosis.md` | **PASSED** (Exact scale expansion metrics verified) |
| **C-06** | 5.2 (RQ2) | *"Forecast-conditional calibration improved empirical coverage (18.91%–31.98%) and lowered Winkler loss (409.66 at L=1, p < 10^-27), but did not restore nominal 90% coverage."* | Directly Supported | EXP-016/018, `uncertainty_results.csv` | **PASSED** (No claim of restored nominal guarantees) |
| **C-07** | 5.2 (RQ3) | *"Uncertainty-aware replenishment increased portfolio demand-weighted fill rate by +7.29 to +10.64 pp across lead times L in {1, 3, 5, 7} (p < 10^-75)."* | Directly Supported | EXP-019, `inventory_results.csv` | **PASSED** (Verified demand-weighted fill rates) |
| **C-08** | 5.2 (RQ3) | *"Mean series-level paired fill rate improved by +10.36 pp at L=1 with 1,000-bootstrap 95% CI [+10.19 pp, +10.55 pp]."* | Directly Supported | EXP-019, `statistical_results.csv` | **PASSED** (Reconciled bootstrap CI verified) |
| **C-09** | 5.2 (RQ3) | *"Conformal safety buffers reduced lost sales by 20.03% to 29.57%, preventing up to 1,139,914 units of unmet demand annually."* | Directly Supported | EXP-019, `inventory_results.csv` | **PASSED** (Exact lost-sales reduction reproduced) |
| **C-10** | 5.2 (RQ4) | *"For L=1 at p=$1.00, conformal replenishment achieved lower total simulated operating cost (-10.83%, saving $790,834, p=6.65e-55)."* | Directly Supported | EXP-019, `inventory_results.csv` | **PASSED** (Exact cost reduction verified) |
| **C-11** | 5.2 (RQ4) | *"For L >= 3 at p=$1.00, holding costs exceeded shortage savings; under higher penalties (p >= $2.00), conformal policy achieved lower total costs across multi-day horizons."* | Directly Supported | EXP-019, `cost_sensitivity_results.csv` | **PASSED** (Trade-off and sensitivity verified) |
| **C-12** | 5.3 | *"Contribution 1: Decision-time-safe direct multi-horizon forecasting protocol preventing future target leakage."* | Directly Supported | EXP-011, `JOURNAL_CONTRIBUTIONS.md` | **PASSED** (Grounded methodological contribution) |
| **C-13** | 5.3 | *"Contribution 2: Empirical study of forecast-conditional conformal calibration under forward distribution shift without claiming theoretical guarantees."* | Directly Supported | EXP-013-018, `JOURNAL_CONTRIBUTIONS.md` | **PASSED** (Properly qualified empirical contribution) |
| **C-14** | 5.3 | *"Contribution 3: Calibration window robustness characterization identifying broad plateau K in [30, 100] and freezing K*=100."* | Directly Supported | EXP-018, `local_conditional_robustness_018.md` | **PASSED** (No misleading KNN nomenclature) |
| **C-15** | 5.3 | *"Contribution 4: Operational translation of forecast uncertainty into replenishment buffers within discrete-event simulation."* | Directly Supported | EXP-019, `JOURNAL_CONTRIBUTIONS.md` | **PASSED** (Grounded operational contribution) |
| **C-16** | 5.3 | *"Contribution 5: Portfolio-level cost-service characterization and inferential statistical testing across lead times and penalties."* | Directly Supported | EXP-019, `JOURNAL_CONTRIBUTIONS.md` | **PASSED** (Grounded economic contribution) |
| **C-17** | 5.4 | *"Managerial implications: point accuracy is insufficient; calibration must be monitored; uncertainty buffers should be evaluated operationally; lead time and penalties dictate cost preference."* | Empirical Interpretation | Section 5.4 | **PASSED** (Properly qualified managerial takeaways) |
| **C-18** | 5.5.1 | *"Dataset limitation: evaluation conducted on single 500-series retail benchmark dataset."* | Limitation | Section 5.5.1, `CLAIM_BOUNDARIES.md` | **PASSED** (Explicit dataset boundary declared) |
| **C-19** | 5.5.2 | *"Simulation limitation: controlled simulation assuming unconstrained capacity, fixed lead times, and lost sales without backorders."* | Limitation | Section 5.5.2, `CLAIM_BOUNDARIES.md` | **PASSED** (Explicit simulation boundary declared) |
| **C-20** | 5.5.3 | *"Demand model limitation: forecasting layer uses XGBoost; does not establish universal superiority over deep neural or foundation models."* | Limitation | Section 5.5.3, `CLAIM_BOUNDARIES.md` | **PASSED** (Explicit model scope declared) |
| **C-21** | 5.5.4 | *"Uncertainty limitation: conformal calibration does not maintain nominal coverage guarantees under distribution shift."* | Limitation | Section 5.5.4, `CLAIM_BOUNDARIES.md` | **PASSED** (Explicit UQ boundary declared) |
| **C-22** | 5.5.5 | *"Inventory policy limitation: evaluates order-up-to policy; does not solve general multi-echelon stochastic optimization."* | Limitation | Section 5.5.5, `CLAIM_BOUNDARIES.md` | **PASSED** (Explicit policy scope declared) |
| **C-23** | 5.5.6 | *"Economic cost limitation: uses fixed linear parameters ($K=$10, h=$0.20, p); real businesses may exhibit nonlinear costs."* | Limitation | Section 5.5.6, `CLAIM_BOUNDARIES.md` | **PASSED** (Explicit cost boundary declared) |
| **C-24** | 5.5.7 | *"Generalization limitation: results reflect specific evaluated setting; external validation is required."* | Limitation | Section 5.5.7, `CLAIM_BOUNDARIES.md` | **PASSED** (Explicit external validity boundary declared) |
| **C-25** | 5.6 | *"Future directions: adaptive calibration under drift, multi-dataset benchmarking, alternative probabilistic models, realistic supply chain dynamics, multi-echelon pooling, real-world field validation."* | Future Hypothesis | Section 5.6 | **PASSED** (Properly framed prospective extensions) |

---

## 2. Forbidden Terminology Search & Verification Audit

A strict search for unhedged, causal, and universal terminology was conducted across the Section 5 text:

1. **"Causal" / "Causally"**: Zero occurrences claiming forecasting causes customer demand. The term appears strictly in "strictly causal feature engineering protocol," referring to temporal filtration isolation ($\tau \ge 7$ days).
2. **"Prove" / "Proven" / "Proving"**: Zero unhedged claims. All statements refer to empirical demonstrations within the evaluated setting.
3. **"Guarantee" / "Guaranteed"**: Used strictly to state that theoretical finite-sample coverage guarantees require exchangeability and failed under forward temporal shift.
4. **"Optimal" / "Optimality"**: Zero claims of global policy optimality. Section 5.5.5 explicitly declares that the study does not solve general stochastic inventory optimization.
5. **"Universally" / "Always" / "Dominant" / "Best" / "Superior"**: Zero unhedged occurrences. Replaced with context-specific empirical descriptions ("cost-preferred under high shortage penalty settings," "lower total simulated cost in the evaluated scenario").
6. **"Strict Monotonic Controllability"**: Zero occurrences. Replaced with "operational responsiveness" and "empirical cost-service trade-off."
