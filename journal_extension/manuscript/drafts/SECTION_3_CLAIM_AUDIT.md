# SECTION 3 CLAIM AUDIT & EPISTEMOLOGICAL BOUNDARY VERIFICATION

This document provides a line-by-line audit of all methodological statements, mathematical formulations, and experimental designs presented in **Section 3 (Methodology)** of the journal extension manuscript. Every assertion is verified against our code implementations, frozen experimental artifacts, and the strict claim boundaries defined in `CLAIM_BOUNDARIES.md`.

---

## 1. Master Methodological Claim Audit Table

| Claim ID | Section | Claim Text / Excerpt | Classification | Verification Source | Boundary Compliance Check |
| :---: | :---: | :--- | :---: | :--- | :---: |
| **M-01** | 3.1 | *"The methodology integrates four distinct functional layers: causal forecasting, rolling conformal calibration, distribution-shift diagnostics, and discrete-event inventory simulation."* | Implementation Fact | `journal_extension/src/experiments/run_full_journal_pipeline.py` | **PASSED** (Accurately describes modular system architecture) |
| **M-02** | 3.1 | *"Zero 2017 observations are used in model training, feature selection, hyperparameter tuning, or calibration parameter estimation."* | Implementation Fact | `METHODOLOGY_FREEZE.md`, config files | **PASSED** (Strict chronological separation verified) |
| **M-03** | 3.2 | *"Cumulative lead-time demand $D_{i,t,L} = \sum_{k=1}^L y_{i,t+k}$ represents total sales over the forward delivery interval."* | Implementation Fact | Eq (1), `direct_multihorizon.py` | **PASSED** (Standard operations research formulation) |
| **M-04** | 3.3 | *"All dynamic autoregressive and rolling features are extracted strictly from historical sales realized at or before day $t-7$."* | Directly Supported | `decision_time_features.py`, `test_feature_leakage.py` | **PASSED** (Verified minimum lag $\ge 7$ days) |
| **M-05** | 3.3 | *"Temporal separation $\Delta \tau = (t+h) - (t-7) = h+7 \ge 8 > 0$ guarantees zero feature leakage into the forecast execution window."* | Directly Supported | Eq (2), mathematical proof in Section 3.3 | **PASSED** (Strictly non-overlapping window) |
| **M-06** | 3.4 | *"Decoupled direct multi-horizon architecture trains 7 independent global regression models $M_1 \dots M_7$ to prevent recursive error accumulation."* | Implementation Fact | Eq (4), `direct_multihorizon.py` | **PASSED** (7 independent XGBoost regressors) |
| **M-07** | 3.4 | *"Cumulative point forecast is obtained by summing point predictions of direct horizon models: $\hat{D}_{i,t,L} = \sum_{k=1}^L M_k(\mathbf{x}_{i,t})$."* | Implementation Fact | Eq (7), `direct_multihorizon.py` | **PASSED** (Exact summation formula) |
| **M-08** | 3.5 | *"Residual calibration pool is populated via rolling-origin TimeSeriesSplit across 2013–2016 development records."* | Implementation Fact | `rolling_conformal.py`, $N=608,165$ | **PASSED** (Zero test partition access during pooling) |
| **M-09** | 3.5 | *"Historical residuals require $r \le t - L - 1$ to ensure complete evaluation before entering the calibration set."* | Implementation Fact | Eq (9), `filter_eligible_residuals` | **PASSED** (Strict decision-time eligibility condition) |
| **M-10** | 3.6 | *"Scale-normalized nonconformity score $s_{i,r,L} = |y - \hat{D}| / (\hat{D} + \epsilon)$ normalizes errors by forecast magnitude."* | Directly Supported | Eq (10), `scale_adaptive_conformal.py` | **PASSED** (Resolves heteroskedastic scale mismatch) |
| **M-11** | 3.6 | *"Rolling conformal calibration extracts empirical quantile from sliding window $W=100$ with finite-sample correction."* | Directly Supported | Eq (11)-(13), `conformal_calibrator.py` | **PASSED** (Frozen $W=100, K^*=100$; no KNN misnomer used) |
| **M-12** | 3.6 | *"Uncertainty-derived safety stock $SS_{i,t,L}^{(\beta)} = q_{i,t,L}^{(\beta)} \cdot \hat{D}_{i,t,L}$ scales proportionally with demand and volatility."* | Directly Supported | Eq (15), `safety_stock.py` | **PASSED** (Mathematically and empirically consistent) |
| **M-13** | 3.7 | *"Residual variance expansion ratio $\rho_{\sigma^2}(h)$ and quarterly signed bias diagnose nonstationary temporal distribution shift."* | Directly Supported | Eq (16)-(20), `run_uncertainty_diagnostics.py` | **PASSED** (Empirically verified 3.55x-4.58x variance expansion) |
| **M-14** | 3.8 | *"Policy A sets $ROP = \hat{D}$ with $SS=0$; Policy B sets $ROP = \hat{D} + SS^{(\beta)} = \text{Upper}^{(\beta)}$."* | Implementation Fact | Eq (22)-(24), `reorder_point.py` | **PASSED** (Exact policy formulations) |
| **M-15** | 3.9 | *"Discrete-event simulation engine executes an 8-step daily timing sequence with lost-sales dynamics."* | Qualified | `inventory_simulator.py`, Section 3.9 | **PASSED** (Explicitly qualified as lost-sales simulation) |
| **M-16** | 3.9 | *"Initial inventory $OH_{i,1} = L \cdot \mu_i^{\text{pre-2017}}$ is derived strictly from 2013–2016 historical sales."* | Implementation Fact | Eq (26), `_initialize_simulation` | **PASSED** (Zero 2017 initialization data used) |
| **M-17** | 3.9 | *"Final $L$ calendar days of test year are excluded from metric aggregation to prevent unfulfilled pipeline truncation artifacts."* | Implementation Fact | Section 3.9, `evaluate_portfolio_metrics` | **PASSED** (Pre-specified boundary safeguard) |
| **M-18** | 3.10 | *"Operational cost model evaluates holding ($h=\$0.20$), ordering ($K_{\text{order}}=\$10$), and shortage ($p=\$1.00$) costs."* | Qualified | Eq (27)-(30), `_settle_daily_costs` | **PASSED** (Explicitly qualified as simulation cost parameters) |
| **M-19** | 3.11 | *"Demand-weighted Fill Rate, Day-Based Stockout Rate, OTIF Service Level, and Winkler Score evaluate system performance."* | Implementation Fact | Eq (31)-(37), `policy_evaluation.py` | **PASSED** (Standard operations and scoring formulations) |
| **M-20** | 3.12 | *"Paired two-sided Wilcoxon signed-rank tests and 1,000-iteration paired bootstrap CIs evaluate statistical significance."* | Directly Supported | Eq (38)-(40), `statistical_results.csv` | **PASSED** (Non-parametric paired evaluation on 500 series) |

---

## 2. Epistemological Boundary & Negative Constraint Audit

We specifically confirm that Section 3 adheres strictly to all negative constraints and prohibited phrases:

1. **No Claims of Guaranteed Coverage Under Distribution Shift**:
   - Section 3.6 and 3.7 explicitly clarify that theoretical finite-sample conformal guarantees require data exchangeability \cite{vovk2005algorithmic}.
   - The text explicitly notes that retail nonstationarity violates exchangeability, and frames $\beta \in \{80\%, 90\%, 95\%\}$ as a **nominal managerial risk-tuning parameter** rather than guaranteed empirical coverage.
2. **No Misnomer of "K-Nearest Neighbors" for Frozen Calibration**:
   - Because the frozen optimal parameter $K^*=100$ equals the sliding window $W=100$, the methodology is correctly designated as **"Rolling Forecast-Conditional Conformal Calibration"** or **"Scale-Normalized Conformal Calibration"**.
   - No misleading KNN terminology is introduced.
3. **Lost Sales & Single-Echelon Boundary Qualification**:
   - Section 3.9 explicitly states that unmet customer demand is treated as **lost sales** (permanently dropped without backlogging).
   - Section 3.8 and 3.9 explicitly state that the inventory control model operates on a **single-echelon retail store setting with deterministic delivery lead times**.
4. **Simulation Cost Parameter Qualification**:
   - Section 3.10 explicitly states that financial metrics are derived from a **standardized commercial cost model** with fixed ordering ($K_{\text{order}}=\$10$), daily holding ($h=\$0.20$), and unit shortage penalties ($p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$), rather than proprietary enterprise accounting ledgers.
5. **No Claims of Universal Cost Dominance in Methodology**:
   - Section 3.8 and 3.10 introduce the replenishment policies objectively without claiming that Policy B is universally cheaper across all lead times and penalty structures.
