# SECTION 2 RELATED WORK CLAIM AUDIT

This document performs an exhaustive audit of all theoretical, methodological, and literature claims asserted in **Section 2 (Related Work)** of the journal manuscript.

---

## 1. Claim Classification Matrix

| Claim # | Section / Location | Stated Claim in Section 2 | Classification | Supporting Literature / Evidence | Status |
| :---: | :--- | :--- | :---: | :--- | :---: |
| **C1** | 2.1 (Retail Forecasting) | Traditional univariate statistical models (ARIMA, ETS) operate locally and cannot capture cross-series dependencies or non-linear features. | **SUPPORTED BY MULTIPLE SOURCES** | Mustapha & Sithole (2025); Syntetos et al. (2016); Yeesin & Kajornkasirat (2024); Mitra et al. (2022). | **VERIFIED** |
| **C2** | 2.1 (Retail Forecasting) | Global cross-series forecasting models learn shared parameter representations from high-volume items to improve volatile item forecasting. | **SUPPORTED BY MULTIPLE SOURCES** | Makridakis et al. (2022 M5); Hobor et al. (2026); Al Ghurabli et al. (2025). | **VERIFIED** |
| **C3** | 2.2 (Machine Learning & GBDT) | GBDTs (XGBoost, LightGBM) consistently outperform deep neural networks on tabular, structured retail demand benchmarks. | **SUPPORTED BY MULTIPLE SOURCES** | Makridakis et al. (2022); Chen & Guestrin (2016); Hobor et al. (2026); Swami et al. (2020). | **VERIFIED** |
| **C4** | 2.2 (Machine Learning & GBDT) | Standard recursive multi-step forecasting risks compounding error propagation and look-ahead lag feature leakage. | **SUPPORTED BY MULTIPLE SOURCES** | Tiwari et al. (2026); EXP-007/008 Methodology Audits. | **VERIFIED** |
| **C5** | 2.3 (Explainable AI) | Black-box opacity creates an "explanation gap" where human planners override automated forecasts with subjective heuristics. | **SUPPORTED BY MULTIPLE SOURCES** | Lin, Lau, & Hu (2026); Chaganti (2026); Narlawar (2025); Chang & Bau (2025). | **VERIFIED** |
| **C6** | 2.3 (Explainable AI) | TreeSHAP provides game-theoretic feature attribution, but explainability alone does not quantify predictive uncertainty or prescribe order quantities. | **INTERPRETIVE SYNTHESIS** | Lundberg & Lee (2017); IEEE E-IDSS Baseline (2026); conceptual boundary analysis. | **VERIFIED** |
| **C7** | 2.4 (Probabilistic Forecasting) | Forecast uncertainty consists of aleatoric (inherent randomness) and epistemic (model/data uncertainty) components. | **DIRECTLY SUPPORTED BY LITERATURE** | Angelopoulos & Bates (2021); Gneiting & Raftery (2007); Huber et al. (2019). | **VERIFIED** |
| **C8** | 2.4 (Probabilistic Forecasting) | Winkler score is a proper scoring rule that simultaneously rewards narrow interval width while penalizing coverage violations. | **DIRECTLY SUPPORTED BY LITERATURE** | Gneiting & Raftery (2007). | **VERIFIED** |
| **C9** | 2.5 (Conformal Prediction) | Conformal prediction provides distribution-free prediction bounds with guaranteed marginal coverage under data exchangeability. | **DIRECTLY SUPPORTED BY LITERATURE** | Vovk et al. (2005); Angelopoulos & Bates (2021). | **VERIFIED** |
| **C10** | 2.5 (Conformal Prediction) | Time-series autocorrelations and nonstationarity violate classical exchangeability, requiring rolling pools or adaptive scaling. | **SUPPORTED BY MULTIPLE SOURCES** | Stankeviciute et al. (2021); Gibbs & Candes (2021); Barber et al. (2023). | **VERIFIED** |
| **C11** | 2.6 (Distribution Shift) | Under forward temporal distribution shifts and residual variance expansion, conformal empirical coverage degrades below nominal levels. | **SUPPORTED BY MULTIPLE SOURCES** | Barber et al. (2023); Tibshirani et al. (2019); EXP-013/014 empirical diagnostics. | **VERIFIED** |
| **C12** | 2.7 (Inventory Replenishment) | Parametric Gaussian safety stock formulas underestimate safety buffers under skewed retail demand and lost-sales dynamics. | **SUPPORTED BY MULTIPLE SOURCES** | Silver, Pyke, & Peterson (1998); Axsäter (2015); Syntetos et al. (2016); Snyder & Shen (2019). | **VERIFIED** |
| **C13** | 2.8 (Forecast-to-Inventory Integration) | Data-driven newsvendor and ERM frameworks map features to orders, but are predominantly static and single-period ($L=0, 1$). | **SUPPORTED BY MULTIPLE SOURCES** | Bertsimas & Kallus (2020); Ban & Rudin (2019); Huber et al. (2019). | **VERIFIED** |
| **C14** | 2.9 (Cost-Service Trade-Offs) | Evaluating forecasting solely on point errors ignores the multi-criteria trade-off between holding, ordering, and shortage costs. | **SUPPORTED BY MULTIPLE SOURCES** | Silver et al. (1998); Ban & Rudin (2019); Spanos et al. (2022); Snyder & Shen (2019). | **VERIFIED** |
| **C15** | 2.10 (Research Gap Synthesis) | No existing framework simultaneously unites leakage-free direct multi-horizon GBDT, conformal UQ, shift diagnostics, and discrete-event inventory simulation. | **INTERPRETIVE SYNTHESIS** | Synthesized from Master Comparison Matrix (`LITERATURE_COMPARISON_TABLE.csv`). | **VERIFIED** |

---

## 2. Writing Discipline & Boundary Verification

1. **Absence of Absolute Novelty Claims**: Verified that Section 2 avoids absolute claims such as *"no previous study has combined forecasting and inventory"* or *"this is the first uncertainty-aware framework."* All claims are framed conservatively as *"limited empirical evidence"*, *"has received comparatively less attention"*, or *"remains insufficiently characterized"*.
2. **Proper Citations for Technical Assertions**: Every theoretical definition (e.g., proper scoring rules, Winkler score, conformal exchangeability, $(r, Q)$ reorder points) is cited immediately with foundational literature (Gneiting & Raftery 2007; Vovk 2005; Silver et al. 1998).
3. **Conference Baseline Contextualization**: The prior IEEE E-IDSS conference paper is properly cited in Section 2.3 and Section 2.10 as the baseline predictive/explainable study, clearly highlighting the expanded scope of the journal extension.
