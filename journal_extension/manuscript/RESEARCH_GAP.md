# RESEARCH GAPS & THEORETICAL POSITIONING

## 1. Context & Motivation

Demand forecasting in retail operations has experienced rapid technological advancement, progressing from classical linear time-series methods (e.g., ARIMA, Exponential Smoothing) to complex non-linear machine learning models (e.g., Extreme Gradient Boosting [XGBoost], LightGBM) and deep neural architectures (e.g., DeepAR, Temporal Fusion Transformers). 

In parallel, research in supply chain management has investigated inventory optimization, safety stock sizing, and multi-echelon replenishment policies. 

However, **these literature streams have evolved largely in silos**. Forecasting research predominantly evaluates models using symmetric statistical loss functions (MAE, RMSE, WAPE), while inventory research typically evaluates replenishment policies under stylized parametric demand distributions (e.g., Gaussian, Poisson) or assumes perfect knowledge of demand parameters. 

To bridge this fundamental divide, this study articulates and investigates **four connected research gaps**.

---

## 2. The Four Connected Research Gaps

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GAP 1: Forecast Accuracy vs. Operational Decision Quality                             │
│ "Predictive accuracy alone does not establish replenishment decision superiority."    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GAP 2: Forecast Uncertainty Under Temporal Distribution Shift                          │
│ "Conformal exchangeability fails under temporal shift; empirical coverage degrades."    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GAP 3: Uncertainty-to-Inventory Translation Across Multi-Day Lead Times                │
│ "Limited evidence on how forecast uncertainty scales into dynamic lead-time buffers." │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ GAP 4: Portfolio-Level Cost–Service Efficient Frontier                                │
│ "Uncertainty awareness creates complex cost-service trade-offs, not universal wins."   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Gap 1 — Forecast Accuracy versus Operational Decision Quality
* **Literature Context**: Modern retail forecasting literature focuses intensively on minimizing point-forecast errors (MAE, RMSE, RMSSE, WAPE) across large-scale benchmarks (e.g., M5 competition, Kaggle benchmarks).
* **Deficiency**: A forecast that minimizes symmetric error can still produce catastrophic operational stockouts because supply chain losses are fundamentally asymmetric—stockouts cause permanent lost sales, customer defection, and SLA penalties, whereas over-forecasting incurs holding costs. Minimal empirical research evaluates whether leakage-free machine learning forecasts translate into better replenishment outcomes when simulated at the item level.
* **Formal Gap Statement**:
  > *Limited empirical evidence connects leakage-free multi-horizon retail forecasting performance with downstream replenishment outcomes using a common, decision-level evaluation framework across large store-item portfolios.*

---

### Gap 2 — Forecast Uncertainty Calibration Under Distribution Shift
* **Literature Context**: Conformal prediction (CP) and distribution-free uncertainty quantification have gained widespread popularity due to their theoretical finite-sample coverage guarantees.
* **Deficiency**: Standard conformal prediction assumes exchangeability (or i.i.d. data). In nonstationary retail environments characterized by macroeconomic trend shifts, promotional volatility, and annual scale expansion, exchangeability is routinely violated. When residual variance explodes (e.g., 3.5x–4.5x expansion in out-of-sample periods), empirical coverage collapses below nominal targets.
* **Formal Gap Statement**:
  > *Limited evidence exists on how forecast-conditional uncertainty calibration behaves under forward retail demand distribution shifts, and how empirical uncertainty margins should be structured and interpreted when empirical coverage deviates substantially from nominal confidence targets.*

---

### Gap 3 — Operational Translation of Forecast Uncertainty into Replenishment Buffers
* **Literature Context**: Classical inventory theory relies on Gaussian safety stock formulas ($SS = z_{\alpha} \cdot \sigma_L$), while machine learning literature focuses on quantile regression and interval generation.
* **Deficiency**: Existing studies rarely integrate non-parametric, scale-normalized conformal prediction intervals directly into multi-period discrete-event reorder point ($ROP$) mechanisms across varying lead times ($L \in \{1, 3, 5, 7\}$) in a lost-sales environment.
* **Formal Gap Statement**:
  > *The operational translation of multi-horizon forecast uncertainty into dynamic replenishment buffers remains insufficiently characterized, particularly regarding how safety margins scale across varying replenishment lead times and non-linear demand profiles.*

---

### Gap 4 — Portfolio-Level Cost–Service Trade-Offs under Asymmetric Shortage Costs
* **Literature Context**: Inventory studies frequently evaluate total cost under fixed holding and backorder penalty parameters, or assess fill rates without modeling holding cost penalties.
* **Deficiency**: Uncertainty-aware safety buffers inherently increase on-hand inventory holding costs. In real-world retail, the economic return of holding safety stock depends heavily on the ratio of holding cost ($h$) to shortage penalty ($p$) and lead time ($L$). Comprehensive portfolio-level studies evaluating this trade-off across hundreds of series are scarce.
* **Formal Gap Statement**:
  > *There remains limited portfolio-level empirical evidence showing how uncertainty-aware replenishment affects customer service, lost sales volume, inventory investment, and total operating cost across varying lead-time constraints and asymmetric shortage-penalty regimes.*

---

## 3. The Unified Research Gap Paragraph (Journal Ready)

> **"Existing research has substantially advanced machine-learning demand forecasting, explainable artificial intelligence, conformal uncertainty quantification, and inventory replenishment optimization; however, these research streams remain only partially connected in retail decision-support research. Forecasting studies commonly emphasize point predictive accuracy, uncertainty studies emphasize interval calibration under exchangeability, and inventory studies evaluate replenishment policies under stylized parametric demand assumptions. Consequently, there is limited empirical evidence on how leakage-free multi-horizon retail forecasts and forecast-conditional uncertainty estimates behave under forward distribution shift, and how this uncertainty translates into operational replenishment performance across different lead times and shortage-cost regimes. In particular, the empirical relationship among conformal uncertainty calibration, customer fill rate, lost sales prevention, inventory investment, and total simulated cost remains insufficiently characterized at the store-item portfolio level."**

---

## 4. Literature Categorization & Critical Synthesis

| Literature Stream | Current State of the Art | Critical Limitations | How Current Study Addresses the Gap |
| :--- | :--- | :--- | :--- |
| **A. Retail Demand Forecasting** | GBDTs (XGBoost, LightGBM) and deep learning models achieve high accuracy on benchmark sales datasets. | Frequently evaluate only point forecasts; many recursive implementations suffer from look-ahead lag feature leakage. | Implements a strictly causal, leakage-free direct multi-horizon architecture ($M_1,\dots,M_7$) with lag $\ge 7$ protocol. |
| **B. Explainable AI (XAI)** | TreeSHAP and LIME provide post-hoc feature importance and local attribution scores. | Feature importance alone is non-prescriptive and does not establish risk-adjusted order quantities. | Connects model explainability with quantitative conformal uncertainty bounds and simulation-based decision evaluation. |
| **C. Probabilistic & Conformal UQ** | Conformal prediction constructs distribution-free prediction intervals with theoretical coverage guarantees. | Assumes exchangeability; collapses under temporal variance expansion and covariate shift; lacks retail scale adaptation. | Diagnoses 2017 distribution shift (3.5x-4.5x variance expansion) and implements rolling forecast-conditional scaling. |
| **D. Inventory Replenishment** | Continuous-review $(s, S)$ and $(r, Q)$ policies using parametric Gaussian safety stock assumptions. | Parametric assumptions fail on skewed, intermittent retail demand; rarely driven by machine-learning prediction intervals. | Evaluates non-parametric conformal uncertainty margins directly as safety stock ($SS_{t,L} = q_{t,L} \cdot \hat{D}_{t,L}$) under lost sales. |
| **E. Forecast-to-Decision Integration** | Emerging "predict-then-optimize" and "end-to-end" decision intelligence frameworks. | Often tested on small synthetic datasets or single-period newsvendor models without multi-day lead-time pipeline tracking. | Evaluates full discrete-event inventory simulation across 500 series $\times$ 4 lead times $\times$ 4 shortage penalties on 2017 test partition. |
