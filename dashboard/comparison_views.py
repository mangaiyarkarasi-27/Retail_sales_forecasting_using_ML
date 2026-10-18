"""
dashboard/comparison_views.py
------------------------------
Renders the side-by-side structural and empirical comparison between
Phase 1 (Conference Baseline: IEEE E-IDSS) and Phase 2 (Scopus Journal Extension).
"""

import os
import streamlit as st
import pandas as pd


def render_phase_comparison(project_root: str, journal_root: str, helpers: dict):
    st.subheader("Structural & Empirical Extension: Conference Baseline vs. Journal Extension")
    
    st.markdown("""
    This view highlights the substantial research and methodological extension from the preliminary
    **Conference Baseline publication** to the completed **Scopus-targeted Journal Extension**.
    """)
    
    # High-level comparison KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card info">
            <div class="metric-lbl">Phase 1: Conference Scope</div>
            <div class="metric-val">Single-Horizon E-IDSS</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.3rem;'>
                Single-step point prediction, heuristic uncertainty (RMSE), static rule recommendation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card success">
            <div class="metric-lbl">Phase 2: Journal Scope</div>
            <div class="metric-val">Direct Multi-Horizon + Conformal</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.3rem;'>
                7 Decoupled XGBoost models, causal lag constraint (τ ≥ 7), rolling forecast-conditional conformal bounds.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card success">
            <div class="metric-lbl">Decision Support Impact</div>
            <div class="metric-val">+10.64 pp Fill Rate</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.3rem;'>
                1.14M units lost sales prevented at L=1, economic cost crossover at p ≥ $2.00.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Table 1: Methodological and Empirical Extension Matrix")
    st.markdown("""
    The table below presents the formal 10-dimension comparison between the published conference paper and the journal extension.
    """)

    comparison_data = [
        {
            "Dimension": "1. Problem Formulation",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Point forecasting + Heuristic MCDA risk scoring",
            "Phase 2: Journal Extension (Scopus Manuscript)": "End-to-end decision-support: multi-horizon forecasting, distribution-free uncertainty calibration, discrete-event replenishment simulation"
        },
        {
            "Dimension": "2. Information Boundary",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Standard lag features (lags 7, 14, 30) without explicit decision-time isolation verification",
            "Phase 2: Journal Extension (Scopus Manuscript)": "Strict causal temporal protocol enforcing minimum 7-day lag constraint (τ ≥ 7 days); verified by 10-check automated leakage audit"
        },
        {
            "Dimension": "3. Multi-Horizon Architecture",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Single-step point estimator (XGBoost v2)",
            "Phase 2: Journal Extension (Scopus Manuscript)": "7 Decoupled Direct XGBoost regressors (M_1 … M_7), completely avoiding recursive error propagation across lead times"
        },
        {
            "Dimension": "4. Uncertainty Quantification",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Heuristic static RMSE interval (Gaussian assumption)",
            "Phase 2: Journal Extension (Scopus Manuscript)": "Non-parametric rolling-origin Conformal Prediction with finite-sample coverage guarantees"
        },
        {
            "Dimension": "5. Scale Adaptation under Shift",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Static homoskedastic variance assumption across all items",
            "Phase 2: Journal Extension (Scopus Manuscript)": "Forecast-conditional normalization s = |e| / (Dhat + ε) scaling uncertainty to forecasted demand volume"
        },
        {
            "Dimension": "6. Distribution Diagnostics",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Standard residual RMSE metric without drift or scale analysis",
            "Phase 2: Journal Extension (Scopus Manuscript)": "Two-sample location-scale decomposition: empirical diagnosis of 3.55x–4.30x residual scale expansion and seasonal swings"
        },
        {
            "Dimension": "7. Hyperparameter Robustness",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Fixed hyperparameter defaults (n_est=50, depth=8)",
            "Phase 2: Journal Extension (Scopus Manuscript)": "Exhaustive validation sensitivity sweep across calibration neighborhood sizes K ∈ [10, 100]; stability plateau K ∈ [30, 100]"
        },
        {
            "Dimension": "8. Downstream Inventory Layer",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Heuristic ±5% growth threshold recommendations (Increase/Reduce/Maintain)",
            "Phase 2: Journal Extension (Scopus Manuscript)": "Discrete-event continuous-review replenishment simulation under lost-sales dynamics across 365 daily events"
        },
        {
            "Dimension": "9. Operational Evaluation",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Forecast error metrics only (MAE, RMSE, R²)",
            "Phase 2: Journal Extension (Scopus Manuscript)": "Fill rate, stockout frequency, shortage units, ordering cost, holding cost, shortage cost evaluated across L ∈ {1, 3, 5, 7}"
        },
        {
            "Dimension": "10. Statistical & Audit Rigor",
            "Phase 1: Conference Baseline (IEEE E-IDSS)": "Point comparison of test MAE/RMSE",
            "Phase 2: Journal Extension (Scopus Manuscript)": "Paired Wilcoxon signed-rank tests (p < 10^-75), 1,000-sample bootstrap CIs, 100% frozen reproducibility audit (EXP-011–020)"
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Narrative on Key Findings
    st.markdown("### Key Scientific Takeaways")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("""
        **What Held from Conference Baseline:**
        - Global Gradient Boosted Decision Trees (XGBoost) consistently outperform linear and naive baselines on retail demand data.
        - Autoregressive calendar and lag features provide strong predictive signals.
        - The Kaggle Store Item Demand dataset (500 store-item series, 2013–2017) provides a robust, heterogeneous testbed.
        """)
    with col_b:
        st.success("""
        **What the Journal Extension Discovered:**
        - **Recursive Leakage Vulnerability:** Standard multi-step recursive forecasting propagates errors and risks data leakage; direct decoupled estimators eliminate this.
        - **Conformal Exchangeability Breakdown:** Under real-world demand shift, standard conformal coverage degrades severely (from 90% to ~29%), driven by a 3.55x–4.30x test-year residual scale expansion.
        - **Forecast-Conditional Calibration:** Conditioning nonconformity scores on forecast scale restores high-demand protection (+8.8 pp top-decile coverage) and minimizes Winkler interval loss.
        - **Economic Frontier:** Conformal safety stock unconditionally cuts shortage units (-20% to -30%) and dominates operating costs whenever shortage penalty p ≥ $2.00.
        """)
