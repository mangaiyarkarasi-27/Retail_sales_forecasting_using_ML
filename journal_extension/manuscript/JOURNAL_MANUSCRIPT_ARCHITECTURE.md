# JOURNAL MANUSCRIPT ARCHITECTURE & CONCEPTUAL POSITIONING

**Target Manuscript Title**:  
*Bridging Multi-Horizon Retail Demand Forecasting and Inventory Decisions via Rolling Forecast-Conditional Conformal Uncertainty Calibration*

**Working Paper Code**: JOURNAL-001  
**Extension Base**: IEEE E-IDSS Conference Baseline (*An Explainable Intelligent Decision Support System for Retail Sales Forecasting Using Machine Learning and Decision Intelligence*)  
**Experimental Foundation**: EXP-011 through EXP-020 (Frozen)

---

## 1. Executive Summary & Narrative Arc

Retail supply chains face a fundamental operational challenge: **machine learning point forecasts optimize for symmetric expected accuracy (e.g., MAE/RMSE), yet downstream inventory replenishment decisions are inherently asymmetric, risk-sensitive, and multi-horizon in nature.**

This journal manuscript presents an end-to-end methodological and empirical framework that spans the entire decision-support chain:
$$\text{Data} \longrightarrow \text{Leakage-Free Multi-Horizon Forecast} \longrightarrow \text{Forecast-Conditional Conformal Uncertainty} \longrightarrow \text{Uncertainty-Aware Replenishment} \longrightarrow \text{Cost-Service Efficient Frontier}$$

### The Core Conceptual Progression:
1. **The Operational Disconnect**: High point-forecast accuracy ($R^2 > 0.93$, MAE $\approx 6.04$) does not prevent supply chain starvation; deterministic point-forecast replenishment suffers from 21% to 67% stockout rates across lead times because it lacks safety buffers for downside demand variance.
2. **The Methodological Challenge**: Standard parametric safety stock assumes Gaussian i.i.d. errors, while raw conformal prediction assumes sample exchangeability. In nonstationary retail environments, forward distribution shift (a 3.5x–4.5x residual variance expansion) leads to empirical coverage degradation.
3. **The Proposed Solution**: A **direct multi-horizon gradient boosting architecture** ($h=1,\dots,7$) paired with **rolling forecast-conditional conformal calibration** ($W=100, K^*=100$), dynamically scaling non-parametric uncertainty margins to forecasted demand magnitude without look-ahead feature leakage.
4. **The Empirical Finding**: When converted into inventory reorder points ($ROP_{t,L} = \hat{D}_{t,L} + SS_{t,L}$), conformal uncertainty delivers **statistically significant portfolio-wide fill rate gains (+7.28 to +10.64 pp, $p < 10^{-75}$)** and reduces lost sales by **20.0% to 29.6% (saving up to 1.14M units)** across 500 store-item series.
5. **The Economic Characterization**: For single-day lead times ($L=1$), uncertainty awareness is an **unconditional win-win** (+10.6 pp fill rate, -10.8% total cost, saving $\$790,834$). For multi-day lead times ($L \ge 3$), it establishes an actionable **cost-service efficient frontier**, achieving total cost dominance whenever shortage penalties exceed $\$2.00/\text{unit}$.

---

## 2. Complete 9-Section Manuscript Structure

```
========================================================================================
SECTION 1: INTRODUCTION
  1.1 Background & Context: Retail sales forecasting & supply chain volatility
  1.2 Problem Motivation: The gap between point accuracy and operational decision-making
  1.3 Four Connected Research Gaps
  1.4 Research Questions (RQ1 to RQ4)
  1.5 Summary of Five Defensible Contributions
  1.6 Paper Organization
----------------------------------------------------------------------------------------
SECTION 2: RELATED WORK
  2.1 Retail Demand Forecasting: Classical, GBDT, and deep learning architectures
  2.2 Multi-Horizon Forecasting: Recursive vs. direct forecasting methodologies
  2.3 Explainable AI (XAI) in Supply Chain Decision Support Systems
  2.4 Probabilistic Forecasting & Conformal Uncertainty Quantification
  2.5 Nonstationarity, Covariate Shift, and Coverage Breakdown in Time Series
  2.6 Inventory Replenishment & Safety Stock under Forecast Uncertainty
  2.7 Synthesis: The Unified Forecast-to-Decision Gap
----------------------------------------------------------------------------------------
SECTION 3: METHODOLOGY
  3.1 Framework Overview & Decision-Time Event Flow
  3.2 Store-Item Problem Formulation & Dataset Slicing
  3.3 Causal Decision-Time Feature Engineering (Zero-Leakage Lag >= 7 Protocol)
  3.4 Direct Multi-Horizon Gradient Boosting Regressors (M1 to M7)
  3.5 Rolling-Origin Out-of-Sample (OOS) Residual Pool Generation
  3.6 Rolling Forecast-Conditional Conformal Uncertainty Calibration
  3.7 Distribution-Shift Diagnostic Framework (Location-Scale Decomposition)
  3.8 Continuous-Review Reorder Point Inventory Policy (Policy A vs. Policy B)
  3.9 Discrete-Event Inventory Simulation Engine & Lost-Sales Accounting
  3.10 Portfolio-Level Cost Functions & Metrics (Holding, Ordering, Shortage)
  3.11 Statistical Inference Protocol (Paired Wilcoxon Signed-Rank & Bootstrap CIs)
----------------------------------------------------------------------------------------
SECTION 4: EXPERIMENTAL DESIGN
  4.1 Benchmark Dataset Characteristics (10 Stores x 50 Items = 500 Series)
  4.2 Chronological Partitioning (2013-2015 Train, 2016 Val, 2017 Untouched Test)
  4.3 Lead-Time Configurations (L in {1, 3, 5, 7} Days)
  4.4 Nominal Uncertainty Coverage Targets (80%, 90%, 95%)
  4.5 Shortage Penalty Sensitivity Grid (p in {$0.60, $1.00, $2.00, $4.00})
  4.6 Operational Service & Cost Performance Metrics Definitions
----------------------------------------------------------------------------------------
SECTION 5: EXPERIMENTAL RESULTS
  5.1 Direct Multi-Horizon Forecasting Performance (h=1 to h=7 Validation)
  5.2 Empirical Conformal Uncertainty Evaluation on 2017 Out-of-Sample Test Set
  5.3 Diagnosis of 2017 Temporal Distribution Shift and Variance Expansion
  5.4 Comparative Evaluation of Conformal Uncertainty Quantification Methods
  5.5 Robustness Analysis of Calibration Neighborhood Size (K in [10, 100])
  5.6 Portfolio-Level Inventory Replenishment & Fill Rate Results
  5.7 Stockout Frequency & Lost Sales Reduction Analysis
  5.8 Total Operational Cost Dynamics & Component Cost Breakdown
  5.9 Sensitivity to Asymmetric Shortage Penalties
  5.10 Disaggregated Performance by Frozen Demand Volume Regimes
  5.11 Paired Statistical Significance Tests & 95% Bootstrap Confidence Intervals
----------------------------------------------------------------------------------------
SECTION 6: DISCUSSION
  6.1 Interpretation of Multi-Horizon Forecasting Accuracy vs. Decision Quality
  6.2 Understanding Conformal Coverage Degradation Under Temporal Covariate Shift
  6.3 Why Empirical Uncertainty Still Yields Superior Operational Replenishment
  6.4 The Dual Cost-Service Dynamics: Win-Win at L=1 vs. Trade-Off Frontier at L>=3
  6.5 Comparison with Previous Retail and Supply Chain Forecasting Literature
----------------------------------------------------------------------------------------
SECTION 7: MANAGERIAL IMPLICATIONS & DECISION INTELLIGENCE
  7.1 Avoiding the "Risk-Neutrality Trap" of Pure Point Forecasts
  7.2 Conformal Prediction Parameters as Actionable Managerial Control Knobs
  7.3 Navigating the Cost-Service Frontier in Asymmetric Shortage Environments
  7.4 Integrating XAI Explanations with Quantitative Uncertainty Buffers
----------------------------------------------------------------------------------------
SECTION 8: LIMITATIONS & FUTURE RESEARCH DIRECTIONS
  8.1 Empirical Boundaries: Single Benchmark Dataset & Lost-Sales Assumptions
  8.2 Conformal Non-Exchangeability Under Macroeconomic Trend Shift
  8.3 Multi-Echelon, Stochastic Lead Times, and Supplier Capacity Constraints
  8.4 End-to-End Asymmetric Pinball Loss Training Opportunities
----------------------------------------------------------------------------------------
SECTION 9: CONCLUSION
  9.1 Summary of Findings
  9.2 Final Research Takeaway
========================================================================================
```

---

## 3. Core Positioning Statement

> **"This paper establishes that high point-forecast accuracy is an insufficient condition for effective retail replenishment, and demonstrates that integrating rolling forecast-conditional conformal uncertainty into multi-horizon inventory decisions delivers statistically significant fill rate improvements (+7.3 to +10.6 pp) and prevents up to 1.14 million units of lost sales, while establishing a quantifiable cost-service efficient frontier across lead times and penalty regimes."**
