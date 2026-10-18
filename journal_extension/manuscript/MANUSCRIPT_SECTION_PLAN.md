# DETAILED JOURNAL MANUSCRIPT SECTION PLAN

This document provides the section-by-section drafting plan for the journal manuscript, specifying target word counts, mathematical formulations, figures, tables, and narrative flow.

---

## Section 1: Introduction (~1,200 words)

- **1.1 Context & Background**:
  - Growth of data-driven retail supply chains, high-dimensional store-item demand volatility, and the imperative for automated replenishment.
- **1.2 Problem Motivation**:
  - The disconnect between symmetric point-forecast optimization (MAE/RMSE) and asymmetric downstream inventory replenishment risks.
  - The limitation of Gaussian safety stock assumptions in skewed, intermittent retail sales.
- **1.3 The Four Connected Research Gaps**:
  - (1) Accuracy vs. Decision Quality; (2) Conformal UQ under Distribution Shift; (3) Uncertainty-to-Inventory Translation; (4) Portfolio Cost-Service Trade-offs.
  - Integration paragraph connecting all four gaps.
- **1.4 Research Questions**:
  - Explicit statement of RQ1, RQ2, RQ3, RQ4.
- **1.5 Summary of Primary Contributions**:
  - Presentation of the 5 defensible contributions.
- **1.6 Paper Organization**:
  - Roadmap of subsequent sections.

---

## Section 2: Related Work (~1,800 words)

- **2.1 Retail Demand Forecasting**:
  - Evolution from ARIMA/ETS to GBDTs (XGBoost, LightGBM) and deep learning (DeepAR, TFT).
  - Multi-horizon forecasting: recursive autoregression vs. direct multi-model strategies.
- **2.2 Explainable AI (XAI) in Supply Chain Decision Support**:
  - Post-hoc interpretability (TreeSHAP, LIME); resolving the "explanation gap" in enterprise DSS.
- **2.3 Probabilistic Forecasting & Conformal Uncertainty Quantification**:
  - Distribution-free conformal prediction foundations (Vovk, Angelopoulos & Bates).
  - Time-series conformal prediction, rolling-origin residual pooling, and nonconformity measures.
- **2.4 Nonstationarity, Covariate Shift & Conformal Coverage Limitations**:
  - Violations of exchangeability under macroeconomic drift and variance explosion (Barber et al., Tibshirani et al.).
- **2.5 Inventory Replenishment & Decision Intelligence**:
  - Classical continuous-review $(s, S)$ and $(r, Q)$ policies; data-driven newsvendor models (Bertsimas & Kallus, Ban & Rudin).
  - Integration of machine-learning prediction intervals into safety stock sizing.
- **2.6 Literature Synthesis & Gap Summary**:
  - Overview table derived from `LITERATURE_GAP_MATRIX.csv`.

---

## Section 3: Methodology (~2,500 words)

- **3.1 End-to-End Decision Framework**:
  - Flow diagram of the complete decision-support architecture.
- **3.2 Mathematical Formulation of Retail Demand**:
  - Definition of store-item series $y_{i,t}$ ($i=1,\dots,N$, $t=1,\dots,T$).
- **3.3 Strictly Causal Feature Engineering (Zero-Leakage Protocol)**:
  - Definition of the 16 decision-time features; formal proof of lag $\ge 7$ temporal isolation.
- **3.4 Direct Multi-Horizon Gradient Boosting Architecture**:
  - Mathematical formulation of decoupled global models $M_1,\dots,M_7$: $\hat{y}_{i,t+h} = M_h(\mathbf{x}_{i,t})$.
  - Multi-day cumulative lead-time demand forecast: $\hat{D}_{t,L} = \sum_{k=1}^L \hat{y}_{t+k}$.
- **3.5 Rolling-Origin Out-of-Sample (OOS) Residual Pool**:
  - TimeSeriesSplit cross-validation over 2013–2016 development data; residual eligibility condition: $r \le t - L - 1$.
- **3.6 Rolling Forecast-Conditional Conformal Calibration**:
  - Scale-normalized nonconformity score: $s_{r,L} = \frac{|y_{r,L} - \hat{D}_{r,L}|}{\hat{D}_{r,L} + \epsilon}$.
  - Conformal quantile extraction over sliding window $W=100$: $q_{t,L}^{(\beta)} = \text{Quantile}_{\lceil (W+1)\beta \rceil / W}(\{s_{r,L}\}_{r \in \mathcal{C}_{t,L}})$.
  - Non-parametric upper bound and safety stock: $\text{Upper}_{t,L}^{(\beta)} = \hat{D}_{t,L} \cdot (1 + q_{t,L}^{(\beta)})$, $SS_{t,L}^{(\beta)} = q_{t,L}^{(\beta)} \cdot \hat{D}_{t,L}$.
- **3.7 Distribution Shift Diagnostic Framework**:
  - Location-scale decomposition; quarterly signed bias; upper/lower violation symmetry equations.
- **3.8 Inventory Replenishment Policy Formulation**:
  - Policy A (Deterministic): $ROP_{t,L} = \hat{D}_{t,L}$, $SS_{t,L} = 0$.
  - Policy B (Conformal Uncertainty-Aware): $ROP_{t,L} = \hat{D}_{t,L} + SS_{t,L}^{(\beta)}$.
- **3.9 Discrete-Event Simulation Engine & Daily Event Timing Sequence**:
  - Formal 8-step execution loop: (1) Receipts $\to$ (2) IP $\to$ (3) Forecast/ROP $\to$ (4) Order $\to$ (5) Dispatch $\to$ (6) Demand $\to$ (7) Fulfillment/Lost Sales $\to$ (8) Cost Settlement.
- **3.10 Cost Accounting & Financial Model**:
  - Holding cost $h \cdot OH_t^{\text{close}}$, ordering cost $K_{\text{order}} \cdot \mathbb{I}(Q_t > 0)$, shortage cost $p \cdot \text{Shortage}_t$, total cost $C_{\text{total}}$.
- **3.11 Statistical Inference Protocol**:
  - Paired two-sided Wilcoxon signed-rank test formulation; 1,000-iteration paired bootstrap CI formulation.

---

## Section 4: Experimental Design (~1,000 words)

- **4.1 Benchmark Dataset Characteristics**:
  - Kaggle Store Item Demand dataset (500 series, 10 stores $\times$ 50 items, 913,000 daily observations).
- **4.2 Chronological Partitioning & Integrity Governance**:
  - Training (2013–2015), Validation (2016), Untouched Test (2017). Zero 2017 tuning constraint.
- **4.3 Parameter Grid Specification**:
  - Lead times $L \in \{1, 3, 5, 7\}$; Nominal coverage $\beta \in \{80\%, 90\%, 95\%\}$; Shortage penalties $p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$; Fixed parameters $K_{\text{order}}=\$10.00, h=\$0.20/\text{unit/day}$.
- **4.4 Initialization & Boundary Safeguards**:
  - Pre-2017 initial inventory $OH_1 = L \cdot \mu_i^{\text{pre-2017}}$; zero pipeline $OO_0=0$; final $L$ days excluded from aggregation.
- **4.5 Metric Definitions**:
  - Fill Rate (demand-weighted), Stockout Rate, Service Level, Shortage Units, Average Ending Inventory, Total Orders, Total Cost, Winkler Score.

---

## Section 5: Experimental Results (~3,000 words)

- **5.1 Direct Multi-Horizon Forecasting Validation**:
  - Presentation of Table `forecasting_results.csv` ($M_1,\dots,M_7$ MAE/RMSE on 2016).
- **5.2 Conformal Uncertainty Evaluation on 2017 Out-of-Sample Test Partition**:
  - Presentation of Table `uncertainty_results.csv` (Raw Conformal vs. Scale-Normalized vs. Forecast-Conditional).
- **5.3 Empirical Diagnosis of 2017 Distribution Shift**:
  - Presentation of 3.55x–4.58x variance expansion and quarterly bias swings.
- **5.4 Robustness & Calibration Window Analysis**:
  - Performance plateau across $K \in [10, 100]$ confirming stability of $W=100$.
- **5.5 Portfolio Replenishment & Service Level Performance**:
  - Presentation of Table `inventory_results.csv` across $L \in \{1, 3, 5, 7\}$ ($p=\$1.00$, nominal 90%).
  - Detailed analysis of fill rate improvements (+7.28 to +10.64 pp) and stockout rate drops.
  - Figure references: `fill_rate_by_lead_time.png`, `stockout_rate_by_lead_time.png`.
- **5.6 Shortage & Lost Sales Reduction**:
  - Up to 1.14M units of lost sales prevented; Figure reference: `shortage_units_by_lead_time.png`.
- **5.7 Inventory Investment & Ordering Dynamics**:
  - Average inventory scaling with lead time; order consolidation savings ($-\$53.5\text{k}$ at $L=1$).
  - Figure references: `average_inventory_by_lead_time.png`, `cost_breakdown_bars.png`.
- **5.8 Total Operational Cost Dynamics & Win-Win at $L=1$**:
  - Total cost reduction at $L=1$ (-10.8%, net saving $\$790,834$, $p = 6.65 \times 10^{-55}$).
  - Figure reference: `total_cost_by_lead_time.png`.
- **5.9 Asymmetric Shortage Penalty Sensitivity**:
  - Presentation of Table `cost_sensitivity_results.csv` ($p=\$0.60$ to $\$4.00$).
  - Characterization of the Cost-Service Efficient Frontier; Figure references: `cost_service_frontier.png`, `shortage_penalty_sensitivity.png`.
- **5.10 Disaggregated Performance by Demand Volume Regimes**:
  - Presentation of Table `demand_regime_results.csv` (Medium vs. High volume tiers); Figure reference: `demand_regime_comparison.png`.
- **5.11 Paired Statistical Significance & Bootstrap Confidence Intervals**:
  - Presentation of Table `statistical_results.csv` (Wilcoxon $p$-values $< 10^{-75}$, Bootstrap 95% CIs).
  - Figure references: `fill_rate_improvement_distribution.png`, `cost_difference_distribution.png`.

---

## Section 6: Discussion (~1,800 words)

- **6.1 The Disconnect Between Forecast Accuracy and Decision Quality**:
  - Why high $R^2$ models starve supply chains under deterministic replenishment.
- **6.2 Conformal Coverage Breakdown Under Nonstationary Covariate Shift**:
  - Why theoretical exchangeability fails in retail, and why scale-normalized margins remain operationally effective despite lower empirical coverage.
- **6.3 Dual Economic Regimes: Single-Day Win-Win vs. Multi-Day Trade-Off Frontier**:
  - Mathematical explanation of why $L=1$ reduces cost unconditionally while $L \ge 3$ creates a trade-off frontier.
- **6.4 Comparison with Prior Retail & Prescriptive Analytics Literature**:
  - Contextualization against M5 findings, Newsvendor ERM, and parametric safety stock models.

---

## Section 7: Managerial Implications & Decision Intelligence (~1,000 words)

- **7.1 Overcoming the "Risk-Neutrality Trap" in Retail Planning**:
  - Shifting from mean point forecasts to risk-buffered replenishment.
- **7.2 Nominal Coverage as an Actionable Managerial Control Knob**:
  - How supply chain executives can tune $\beta \in [0.80, 0.95]$ to navigate corporate service-level agreements.
- **7.3 Decision-Support Integration with Explainable AI**:
  - Unifying SHAP feature drivers with quantitative conformal reorder points on executive dashboards.

---

## Section 8: Limitations & Future Research Directions (~800 words)

- **8.1 Empirical & Dataset Boundaries**:
  - Single public benchmark; absence of promotional/price elasticities.
- **8.2 Conformal Non-Exchangeability & Adaptive Drift Tracking**:
  - Need for forward-looking volatility forecasting layers.
- **8.3 Multi-Echelon & Stochastic Lead-Time Extensions**:
  - Upstream distribution center pooling, random lead-time arrivals, and capacity constraints.
- **8.4 End-to-End Asymmetric Loss Training**:
  - Direct pinball/newsvendor loss integration into multi-horizon gradient boosting.

---

## Section 9: Conclusion (~500 words)

- **9.1 Summary of Contributions & Core Takeaways**:
  - Concluding synthesis of multi-horizon forecasting, conformal uncertainty, and inventory simulation.
- **9.2 Final Concluding Statement**:
  - The transformative role of decision-aware uncertainty quantification in modern retail operations.

---

## Total Estimated Manuscript Length: ~13,600 words (Standard Full Journal Paper Length)
