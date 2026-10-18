# Visual Audit Report (VISUAL_AUDIT.md)

## Overview
This visual audit evaluates every figure and table in the journal manuscript against the scientific visualization guidelines, research question mapping, originality, citation requirements, and non-redundancy principles.

, 

## 1. Figure Audit Matrix

| ID | Label / Reference | Figure Title / Subject | Purpose | RQ Supported | Original / External | Citation Required | Citation Present | Redundant? | Keep / Remove |
|, |, |, |, |, |, |, |, |, |, |
| **Fig. 1** | `fig:framework` | End-to-End Journal Extension Framework | System architecture showing distinction between conference baseline and journal extensions | All RQs (RQ1--RQ4) | Original | No (Author Framework) | N/A (Original) | No | **KEEP** |
| **Fig. 2** | `fig:direct_architecture` | Decision-Time Temporal Availability Protocol | Illustrates $\tau \ge 7$ days lag constraint enforcing temporal isolation | RQ1 | Original | No (Author Protocol) | N/A (Original) | No | **KEEP** |
| **Fig. 3** | `fig:multi_horizon_workflow` | Decoupled Direct vs. Recursive Forecasting | Compares 7 direct estimators $M_1\dots M_7$ against recursive feedback | RQ1 | Original | No (Author Design) | N/A (Original) | No | **KEEP** |
| **Fig. 4** | `fig:shift_diagnostics` | Residual Scale Expansion & Distribution Shift | Demonstrates $3.55\times\dots 4.30\times$ residual scale expansion under shift | RQ2 | Original (EXP-014) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Fig. 5** | `fig:coverage_width` | Conformal Coverage vs. Nominal Targets | Illustrates coverage degradation under nonstationary variance expansion | RQ2 | Original (EXP-015/016) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Fig. 6** | `fig:winkler_comparison` | Mean Winkler Interval Score Comparison | Evaluates Winkler loss across Raw, Scale-Normalized, and Forecast-Conditional CP | RQ2 | Original (EXP-017) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Fig. 7** | `fig:k_sensitivity` | Validation Winkler Loss vs. Neighborhood Size $K$ | Demonstrates calibration stability plateau across $K \in [10, 100]$ | RQ2 | Original (EXP-018) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Fig. 8** | `fig:service_comparison` | Inventory Service Level & Lost Sales Mitigation | Shows $+7.29$ to $+10.64$ pp fill rate gains and lost sales reduction | RQ3 | Original (EXP-019) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Fig. 9** | `fig:cost_surface` | Economic Cost Preference Surface | Maps cost preference crossover frontier across lead times and penalty $p$ | RQ4 | Original (EXP-019) | No (Author Results) | N/A (Original) | No | **KEEP** |

, 

## 2. Table Audit Matrix

| ID | Label / Reference | Table Title / Subject | Purpose | RQ Supported | Original / External | Citation Required | Citation Present | Redundant? | Keep / Remove |
|, |, |, |, |, |, |, |, |, |, |
| **Tab. 1** | `tab:conference_vs_journal` | Conference Baseline vs. Journal Extension | Structural mapping of 10 extension dimensions | All RQs | Original | Yes (Cite Conference [e_idss_ieee]) | Yes (`\citep{e_idss_ieee}`) | No | **KEEP** |
| **Tab. 2** | `tab:partitions` | Dataset Partitioning & Calibration Pool | Experimental split (Train, Val, OOS Test holdout) | Setup | Original | No (Author Setup) | N/A (Original) | No | **KEEP** |
| **Tab. 3** | `tab:forecasting_results` | Direct Multi-Horizon Forecasting Accuracy | Benchmark results (MAE, RMSE, WAPE, SMAPE) | RQ1 | Original (EXP-011) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Tab. 4** | `tab:conformal_results` | Conformal Uncertainty Calibration | Empirical coverage, Winkler loss, top-decile performance | RQ2 | Original (EXP-013--EXP-018) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Tab. 5** | `tab:shift_diagnostics` | Distribution Shift Diagnostics | Two-sample KS test statistics, std dev ratios, MAE ratios | RQ2 | Original (EXP-014) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Tab. 6** | `tab:k_sensitivity` | Calibration Neighborhood Sensitivity $K$ | Winkler loss sweep across $K \in [10, 100]$ | RQ2 | Original (EXP-018) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Tab. 7** | `tab:inventory_service` | Replenishment Service Levels & Stockouts | Portfolio fill rate, bootstrap CIs, lost sales units | RQ3 | Original (EXP-019) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Tab. 8** | `tab:inventory_cost` | Portfolio Simulated Annual Operating Cost | Detailed cost breakdown (Ordering, Holding, Shortage) | RQ4 | Original (EXP-019) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Tab. 9** | `tab:shortage_sensitivity` | Shortage Penalty Sensitivity Matrix | Net cost delta ($\Delta TC$) across penalty regimes $p$ | RQ4 | Original (EXP-019) | No (Author Results) | N/A (Original) | No | **KEEP** |
| **Tab. 10** | `tab:rq_findings` | Summary of RQs & Empirical Findings | Concise synthesis connecting RQs to findings | All RQs (RQ1--RQ4) | Original | No (Author Synthesis) | N/A (Original) | No | **KEEP** |
| **Tab. A1**| `tab:app_notation` | Mathematical Notation Summary | Appendix notation table | Appendix A | Original | No (Author Notation) | N/A (Original) | No | **KEEP** |
| **Tab. A2**| `tab:app_features` | Feature Engineering Space Breakdown | Appendix 45-dimension feature list | Appendix B | Original | No (Author Feature List)| N/A (Original) | No | **KEEP** |
| **Tab. A4**| `tab:app_computational`| Computational Complexity & Runtime | Appendix hardware & execution wall-clock time | Appendix D | Original | No (Author Benchmark) | N/A (Original) | No | **KEEP** |

, 

## 3. Checklist Compliance Verification

- [x] **Scientific Purpose**: Every visual directly supports specific RQs or methodological components.
- [x] **No Decorative Graphics**: 0 stock photos, 0 generic internet figures, 0 dashboard screenshots.
- [x] **Zero Redundancy**: Tables provide exact numerical values; figures visualize structural trade-offs.
- [x] **In-Text Citation**: 100% of figures and tables are cited in the main narrative prior to appearance using `\autoref` or `\ref`.
- [x] **Source Attribution**: Original author-generated experimental results are clearly identified in captions ("Source: Authors' experimental results").
- [x] **Exact Numerical Consistency**: All numbers in tables and figures exactly match frozen experimental artifacts EXP-011 through EXP-019.
- [x] **Page Layout Fit**: All tables fit 100% within `\textwidth` with zero margin overflows.
