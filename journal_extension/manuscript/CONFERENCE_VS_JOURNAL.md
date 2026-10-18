# CONFERENCE VS. JOURNAL EXTENSION: METHODOLOGICAL COMPARISON

This document establishes the precise structural, methodological, and empirical mapping between the original conference paper (**E-IDSS**) and the comprehensive journal extension.

---

## 1. Context & Relationship

- **Conference Paper**: *An Explainable Intelligent Decision Support System (E-IDSS) for Retail Sales Forecasting Using Machine Learning and Decision Intelligence* (Published/Presented at IEEE Conference).
  - **Focus**: Established the core retail forecasting framework using XGBoost, TreeSHAP feature explainability, a Multi-Criteria Decision Analysis (MCDA) risk scoring engine, and a 9-page Streamlit operational dashboard.
- **Journal Extension**: Substantially expands the conference work from a point-predictive and heuristic scoring framework into a **theoretically grounded, leakage-free, conformal uncertainty-driven, and discrete-event inventory decision-support system**.

---

## 2. Comprehensive Methodological Comparison Matrix

| Component | Conference Baseline (E-IDSS) | Journal Extension | Scientific Rationale for Extension |
| :--- | :--- | :--- | :--- |
| **1. Primary Research Problem** | Explainable point demand forecasting & heuristic priority scoring. | End-to-end translation of forecast uncertainty into multi-horizon inventory decisions. | Point accuracy does not prevent stockouts; uncertainty quantification is required for safety stock. |
| **2. Forecasting Architecture** | Single-step global XGBoost regression model. | 7 Global Direct Multi-Horizon XGBoost models ($M_1, \dots, M_7$). | Decouples forecast horizons ($h=1..7$) without recursive error propagation or target leakage. |
| **3. Feature Temporal Safety** | Standard time-series lags and rolling averages. | Strictly causal decision-time feature protocol (all lags $\ge 7$ days). | Eliminates autoregressive look-ahead feature leakage across all 7 evaluation horizons. |
| **4. Feature Explainability** | TreeSHAP local attributions and global summary plots. | Preserved and integrated with quantitative uncertainty margins. | Explains feature drivers while conformal quantiles provide actionable risk bounds. |
| **5. Calibration Residuals** | In-sample / basic residual standard deviations. | Rolling-Origin Out-of-Sample (OOS) calibration residual pool ($N=608,165$). | Prevents in-sample overfitting bias from artificially narrowing nonconformity quantiles. |
| **6. Uncertainty Quantification** | Heuristic variance / MCDA Risk Score. | Rolling Forecast-Conditional Conformal Calibration ($W=100, K^*=100, \epsilon=10^{-4}$). | Distribution-free, scale-normalized non-parametric upper prediction bounds. |
| **7. Distribution Shift Analysis** | Not characterized (assumed stationary environment). | Explicit location-scale diagnostic decomposition of 2017 test shift. | Diagnoses why conformal empirical coverage degrades under 3.5x–4.5x variance expansion. |
| **8. Calibration Robustness** | Fixed single heuristic score. | Full calibration neighborhood sensitivity audit ($K \in [10, 100]$). | Establishes a broad stability plateau across calibration window sizes. |
| **9. Inventory Replenishment** | Heuristic categorical recommendations (Increase/Maintain/Reduce). | Continuous-review order-up-to policy ($ROP_{t,L} = \hat{D}_{t,L} + SS_{t,L}$). | Provides exact daily quantitative order quantities ($Q_t$) for supply chain execution. |
| **10. Simulation Engine** | No physical inventory simulation executed. | 8-step daily discrete-event simulation with Lost Sales dynamics. | Simulates daily pipeline arrivals, inventory position, and unmet customer demand. |
| **11. Lead-Time Evaluation** | Undefined / single-period implicit assumption. | Multi-lead-time evaluation ($L \in \{1, 3, 5, 7\}$ days). | Captures lead-time demand risk accumulation across short and long replenishment pipelines. |
| **12. Financial Cost Model** | No cost accounting or financial settlement. | Comprehensive operational cost model ($K_{\text{order}}=\$10, h=\$0.20/\text{day}, p=\$1.00$). | Quantifies the financial balance between holding, ordering, and shortage penalties. |
| **13. Shortage Sensitivity** | Fixed heuristic assumption. | Sensitivity sweep across shortage penalties ($p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$). | Reveals critical penalty thresholds where uncertainty awareness becomes cost-dominant. |
| **14. Statistical Evaluation** | Point error metrics (MAE, RMSE, $R^2$). | Paired Wilcoxon signed-rank tests ($p < 10^{-75}$) + 1,000 paired bootstrap CIs. | Establishes portfolio-wide statistical significance across 500 heterogeneous series. |

---

## 3. Preservation of Conference Baseline

The journal paper explicitly cites and builds upon the conference paper:
- **Baseline Models Preserved**: The conference XGBoost baseline and TreeSHAP explainer are cited as the initial predictive foundation.
- **Dataset Consistency**: Both studies evaluate the 500 store-item benchmark series (10 stores $\times$ 50 items).
- **No Revisionist History**: The conference validation MAE (5.8615 under 1-day lag) is transparently discussed in comparison to the strictly leakage-free multi-horizon validation MAE (6.0388 with lag $\ge 7$), highlighting how eliminating look-ahead feature leakage produces scientifically sound operational models.
