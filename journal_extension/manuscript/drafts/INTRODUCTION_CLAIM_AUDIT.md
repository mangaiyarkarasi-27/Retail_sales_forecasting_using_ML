# INTRODUCTION SECTION CLAIM AUDIT

This document performs an exhaustive audit of all factual, theoretical, methodological, and empirical claims asserted in **Section 1 (Introduction)** of the journal manuscript.

---

## 1. Audit Summary Matrix

| Claim # | Paragraph / Location | Stated Claim in Introduction | Classification | Supporting Evidence / Boundary Condition | Status |
| :---: | :--- | :--- | :---: | :--- | :---: |
| **C1** | Para 1 (Context) | Retail demand exhibits substantial heterogeneity, seasonality, promotional spikes, and intermittency across store-item series. | **LITERATURE-SUPPORTED** | Mitra et al. (2022); Venkiteela (2025); Tiwari & Prasad (2025). | **VERIFIED** |
| **C2** | Para 1 (Context) | Inaccurate forecasts propagate into operational stockouts (lost sales) or overstocking (holding cost/waste). | **LITERATURE-SUPPORTED** | Spanos et al. (2022); Malik et al. (2024); Mustapha & Sithole (2025). | **VERIFIED** |
| **C3** | Para 2 (Point vs. Risk) | Symmetric point-forecast metrics (MAE, RMSE) treat over- and under-estimation equally, whereas retail supply chain losses are asymmetric. | **LITERATURE-SUPPORTED** | Ban & Rudin (2019); Bertsimas & Kallus (2020); Gneiting & Raftery (2007). | **VERIFIED** |
| **C4** | Para 3 (Machine Learning) | Global GBDTs (XGBoost, LightGBM) outperform classical statistical methods on structured multi-item retail sales benchmarks. | **LITERATURE-SUPPORTED** | Makridakis et al. (2022 M5); Chen & Guestrin (2016); Hobor et al. (2026). | **VERIFIED** |
| **C5** | Para 3 (Machine Learning) | Recursive multi-step forecasting can suffer from compounding error propagation and look-ahead lag feature leakage. | **LITERATURE-SUPPORTED** | Tiwari et al. (2026); EXP-007/008 Methodology Audits. | **VERIFIED** |
| **C6** | Para 4 (Explainability) | E-IDSS baseline introduced XGBoost with TreeSHAP to address the enterprise explanation gap and reduce override rates. | **LITERATURE-SUPPORTED** | IEEE E-IDSS Conference Paper (2026); Lundberg & Lee (2017); Lin et al. (2026); Narlawar (2025). | **VERIFIED** |
| **C7** | Para 4 (Explainability) | Explainability describes historical feature importance but does not quantify predictive uncertainty or determine replenishment quantities. | **INTERPRETIVE** | Conceptual boundary between descriptive attribution (SHAP) and prescriptive decision models. | **VERIFIED** |
| **C8** | Para 5 (Conformal UQ) | Conformal prediction constructs distribution-free prediction bounds from historical residuals without Gaussian i.i.d. assumptions. | **LITERATURE-SUPPORTED** | Vovk et al. (2005); Angelopoulos & Bates (2021); Stankeviciute et al. (2021). | **VERIFIED** |
| **C9** | Para 5 (Conformal UQ) | Standard conformal prediction finite-sample coverage guarantees depend on data exchangeability. | **LITERATURE-SUPPORTED** | Vovk et al. (2005); Barber et al. (2023). | **VERIFIED** |
| **C10** | Para 6 (Distribution Shift) | Temporal distribution shifts (variance expansion, nonstationarity) cause conformal empirical coverage to fall below nominal targets. | **DIRECTLY SUPPORTED** | EXP-013/014: 2017 empirical coverage (15.5%–34.7%) under 3.55x–4.58x test-year variance expansion; Barber et al. (2023). | **VERIFIED** |
| **C11** | Para 7 (Inventory Decision) | Continuous-review reorder points must cover expected lead-time demand plus safety stock buffers: $ROP_{t,L} = \hat{D}_{t,L} + SS_{t,L}$. | **LITERATURE-SUPPORTED** | Silver, Pyke, & Peterson (1998); Axsäter (2015); Syntetos et al. (2016). | **VERIFIED** |
| **C12** | Section 1.2 (Gaps) | Four connected research gaps connect forecast accuracy, distribution shift, uncertainty translation, and cost-service trade-offs. | **INTERPRETIVE** | Synthesized from literature matrix (JOURNAL-001 / `RESEARCH_GAP.md`). | **VERIFIED** |
| **C13** | Section 1.3 (Extension) | Journal extension adds direct multi-horizon models, OOS residuals, conformal calibration, discrete-event simulation, and lead-time analysis. | **DIRECTLY SUPPORTED** | Methodological comparison matrix verified against codebase and EXP-011 through EXP-020. | **VERIFIED** |
| **C14** | Section 1.5 (Contribution 1) | Direct multi-horizon XGBoost with lag $\ge 7$ features ensures decision-time feature availability and stable validation MAE (6.0388). | **DIRECTLY SUPPORTED** | EXP-011: 7 global models, 10/10 leakage checks passed, validation MAE 6.0388. | **VERIFIED** |
| **C15** | Section 1.5 (Contribution 2) | Rolling forecast-conditional scale normalization reduces Winkler score ($p < 0.02$) and boosts high-demand coverage by +8.8 pp. | **DIRECTLY SUPPORTED** | EXP-016/017/018: Mean Winkler loss 2198.81 vs 2244.38; high-regime coverage 28.0% $\to$ 36.8%. | **VERIFIED** |
| **C16** | Section 1.5 (Contribution 3) | Coverage degradation is driven by 3.55x–4.58x variance expansion; calibration window shows stability plateau across $K \in [30, 100]$. | **DIRECTLY SUPPORTED** | EXP-014 Location-Scale Diagnostics; EXP-018 K-Sensitivity Audit. | **VERIFIED** |
| **C17** | Section 1.5 (Contribution 4) | Conformal safety stock improves portfolio fill rate (+7.28 to +10.64 pp, $p < 10^{-75}$) and cuts lost sales by 20.0% to 29.6% (saving up to 1.14M units). | **DIRECTLY SUPPORTED** | EXP-019: Simulation across 500 series $\times$ 4 lead times; paired Wilcoxon $p < 10^{-75}$. | **VERIFIED** |
| **C18** | Section 1.5 (Contribution 5) | Uncertainty-aware replenishment achieves win-win at $L=1$ (-10.8% cost, $-\$790.8\text{k}$) and an efficient frontier at $L \ge 3$ dominant when $p \ge \$2.00$. | **REQUIRES QUALIFICATION** | EXP-019: Qualified that at $L \ge 3$ and $p=\$1.00$, holding costs exceed shortage savings, creating a trade-off frontier. | **VERIFIED & QUALIFIED** |

---

## 2. Epistemological Governance Verification

1. **No Overclaiming of Coverage**: Verified that Section 1 explicitly states that conformal empirical coverage degrades under temporal distribution shift, and avoids any claim of guaranteed finite-sample calibration on the 2017 test set.
2. **No Overclaiming of Economic Superiority**: Verified that Section 1 states that cost reductions are unconditional only at $L=1$, while for $L \ge 3$ uncertainty awareness establishes an efficient frontier dependent on shortage penalty levels.
3. **No Overclaiming of Stockout Elimination**: Verified that Section 1 describes fill rate gains and lost sales reduction rather than complete stockout elimination.
4. **Causality & Terminology Discipline**: Verified that Section 1 uses *"decision-time-safe / leakage-free forecasting"* rather than *"causal forecasting"*, and *"rolling forecast-conditional conformal calibration"* rather than *"KNN local conformal prediction"*.
