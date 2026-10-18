# LITERATURE GAP-TO-CONTRIBUTION MAPPING

This document maps the identified research gaps directly to the verified empirical evidence, methodological responses, and completed experiments from EXP-011 through EXP-020.

---

## 1. Master Gap-to-Contribution Matrix

| Research Gap | Literature Evidence | Methodological & Scientific Response | Completed Experiment | Supported Claim Strength |
| :--- | :--- | :--- | :---: | :---: |
| **Gap 1: Forecast Accuracy vs. Operational Decision Quality** | Makridakis et al. (2022); Chen & Guestrin (2016); Hobor et al. (2026); Ban & Rudin (2019) emphasize point predictive metrics (MAE, RMSE, WAPE) without connecting them to downstream multi-horizon inventory replenishment outcomes. | Develop a strictly causal, direct multi-horizon GBDT architecture ($M_1,\dots,M_7$) with lag $\ge 7$ protocol, evaluated directly inside a multi-period discrete-event inventory simulation engine. | **EXP-011** (Forecasting Validation) + **EXP-019** (Inventory Simulation) | **DIRECTLY SUPPORTED** (Validation MAE 6.0388; fill rate gains +7.3 to +10.6 pp over point-forecast baseline). |
| **Gap 2: Uncertainty Calibration Under Distribution Shift** | Vovk et al. (2005); Angelopoulos & Bates (2021); Barber et al. (2023); Tibshirani et al. (2019) prove theoretical conformal bounds under exchangeability, but demonstrate that forward nonstationarity and variance expansion degrade empirical coverage below nominal levels. | Perform location-scale diagnostic decomposition of 2017 distribution shift (3.55x–4.58x variance expansion); implement rolling forecast-conditional scale normalization ($s_{r,L} = \frac{|y_{r,L} - \hat{D}_{r,L}|}{\hat{D}_{r,L} + \epsilon}$) with $W=100, K^*=100$. | **EXP-013/014** (Shift Diagnostics) + **EXP-016/017/018** (Conformal Calibration & Freezing) | **DIRECTLY SUPPORTED** (Winkler score reduced to 2198.81, $p < 0.02$; high-demand coverage boosted +8.8 pp; coverage degradation empirically characterized). |
| **Gap 3: Uncertainty-to-Inventory Translation Across Multi-Day Lead Times** | Silver et al. (1998); Syntetos et al. (2016); Bertsimas & Kallus (2020); Stankeviciute et al. (2021) isolate probabilistic forecasting from continuous-review $(r, Q)$ control or confine optimization to single-period static newsvendor settings. | Couple multi-horizon forecast-conditional conformal bounds directly into continuous-review reorder points ($ROP_{t,L} = \hat{D}_{t,L} + SS_{t,L}^{(\beta)}$), simulating physical pipeline tracking and lost sales across lead times $L \in \{1, 3, 5, 7\}$. | **EXP-019** (Multi-Lead-Time Replenishment Simulation) | **DIRECTLY SUPPORTED** (Shortage reduced by 20.0% to 29.6%, saving up to 1.14M units across 500 store-item series; paired Wilcoxon $p < 10^{-75}$). |
| **Gap 4: Portfolio-Level Cost–Service Efficient Frontier** | Huber et al. (2019); Spanos et al. (2022); Snyder & Shen (2019) evaluate inventory costs under fixed cost parameters or isolated service metrics without characterizing multi-lead-time sensitivity to asymmetric shortage penalties. | Evaluate total operational costs across an exhaustive grid ($L \in \{1, 3, 5, 7\}$, $p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$, $\beta \in \{80\%, 90\%, 95\%\}$) with paired Wilcoxon tests and 1,000 bootstrap CIs across 500 series. | **EXP-019** (Cost Sensitivity & Frontier Characterization) | **DIRECTLY SUPPORTED** (Unconditional win-win at $L=1$, $-\$790.8\text{k}$ total cost; cost-service efficient frontier at $L \ge 3$ dominant when $p \ge \$2.00$). |

---

## 2. Epistemological Integrity Assurance

1. **Every Proposed Contribution is Mapped to an Actual Experiment**: No theoretical claim is asserted without an explicit experimental mapping (EXP-011 through EXP-019).
2. **No Overclaiming of Novelty**: The paper acknowledges existing foundational works (XGBoost, TreeSHAP, Conformal Prediction, Newsvendor ERM) and specifies that the core contribution is the **end-to-end integration and empirical evaluation** of these components across a large retail portfolio under distribution shift.
3. **No Unqualified Generalization**: Findings are strictly bound to the 500 store-item benchmark series under discrete-event simulation assumptions (lost sales, deterministic lead times, unconstrained supply).
