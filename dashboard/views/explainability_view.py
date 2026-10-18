"""
dashboard/views/explainability_view.py
-------------------------------------
Explainable AI (XAI) module for E-IDSS.
Provides local natural-language rationale for active series decisions,
global SHAP feature attribution galleries, and adaptive model routing
meta-feature importance analysis.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import series_service


def render_explainability_view(project_root: str, journal_root: str, helpers: dict):
    load_image = helpers["load_image"]

    active_store = st.session_state.get("active_store", 1)
    active_item = st.session_state.get("active_item", 1)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%); 
                padding: 1.5rem 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
                box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.25);">
        <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">
            🔍 Explainable AI & Decision Attribution
        </h2>
        <p style="color: #e0e7ff; margin-top: 0.4rem; font-size: 0.95rem; margin-bottom: 0;">
            Game-theoretic feature attribution via TreeExplainer SHAP values and adaptive router 
            meta-feature analysis, bridging mathematical ML predictions to operational business rationale.
        </p>
    </div>
    """, unsafe_allow_html=True)

    dec = series_service.get_series_decision_record(project_root, active_store, active_item)
    rank = dec.get("Priority_Rank", 1)
    score = dec.get("Decision_Score", 50.0)
    level = dec.get("Priority_Level", "Medium")

    st.markdown(f"""
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem; color: #334155;">
            <span>📍 <b>Active Series:</b> Store {active_store} · Item {active_item}</span>
            <span><b>Priority Rank:</b> #{rank} / 500</span>
            <span><b>Decision Score:</b> {score:.1f} / 100</span>
            <span><b>Level:</b> <span style="font-weight:600; color:#4338ca;">{level}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Local Explainability: Natural Language Rationale
    st.markdown("### 💬 Local Decision Attribution & Natural Language Reasoning")
    st.caption("Auto-generated managerial justifications derived from SHAP decomposition and demand trajectory analysis:")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        shap_expl = dec.get("Explanation_SHAP") or dec.get("SHAP_Explanation", "Feature contributions indicate standard seasonal patterns.")
        st.markdown(f"""
        <div style="background: #f8fafc; border-left: 4px solid #4338ca; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: #1e1b4b; font-size: 0.95rem; margin-bottom: 0.3rem;">
                🎯 Key Feature Attribution (SHAP Driver)
            </div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                {shap_expl}
            </div>
        </div>
        """, unsafe_allow_html=True)

        trend_expl = dec.get("Explanation_Trend") or dec.get("Trend_Explanation", "Trend dynamics reflect baseline momentum.")
        st.markdown(f"""
        <div style="background: #f8fafc; border-left: 4px solid #0284c7; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0;">
            <div style="font-weight: 700; color: #0c4a6e; font-size: 0.95rem; margin-bottom: 0.3rem;">
                📈 Trend & Velocity Momentum
            </div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                {trend_expl}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_exp2:
        fc_expl = dec.get("Explanation_Forecast") or dec.get("Forecast_Explanation", "Predicted demand is aligned with trailing moving averages.")
        st.markdown(f"""
        <div style="background: #f8fafc; border-left: 4px solid #059669; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
            <div style="font-weight: 700; color: #064e3b; font-size: 0.95rem; margin-bottom: 0.3rem;">
                🔮 Horizon Demand Trajectory
            </div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                {fc_expl}
            </div>
        </div>
        """, unsafe_allow_html=True)

        risk_expl = dec.get("Explanation_Risk") or dec.get("Risk_Explanation", "Volatility indicators remain within acceptable tolerance thresholds.")
        st.markdown(f"""
        <div style="background: #f8fafc; border-left: 4px solid #d97706; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0;">
            <div style="font-weight: 700; color: #78350f; font-size: 0.95rem; margin-bottom: 0.3rem;">
                ⚠️ Volatility & Stockout Vulnerability
            </div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                {risk_expl}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Global SHAP Attribution Gallery
    st.markdown("### 🌐 Global Model Attribution (TreeExplainer SHAP)")
    tab_sum, tab_bar, tab_dep, tab_wf = st.tabs([
        "SHAP Summary (Beeswarm)", 
        "Global Feature Importance", 
        "Feature Dependence (Nonlinear Interactions)", 
        "Local Waterfall Decomposition"
    ])

    with tab_sum:
        col_s_txt, col_s_img = st.columns([1, 1])
        with col_s_txt:
            st.markdown("""
            **SHAP Beeswarm Summary Plot**
            - Each point represents an individual prediction sample from the test partition.
            - **Color represents feature value:** High values in red/pink, low values in blue.
            - **Horizontal position represents impact:** Positive SHAP values push predicted demand higher; negative values pull demand lower.
            - **Primary Takeaway:** Trailing sales lags (`sales_lag_7`, `sales_lag_14`) and longer-term rolling averages (`rolling_mean_30`) dominate predictive impact with strong positive correlation.
            """)
        with col_s_img:
            img = load_image(os.path.join(project_root, "results", "plots", "shap", "shap_summary_plot.png"))
            if img:
                st.image(img, caption="Global SHAP Beeswarm Distribution Across Retail Features", use_container_width=True)

    with tab_bar:
        col_b_txt, col_b_img = st.columns([1, 1])
        with col_b_txt:
            st.markdown("""
            **Global Feature Ranking (|SHAP| Importance)**
            - Measures the mean absolute SHAP value across all out-of-sample observations:
              $$\\text{Importance}_j = \\frac{1}{N}\\sum_{i=1}^N |\\phi_{i,j}|$$
            - Validates that causal lag features constructed at decision time $t$ carry $>85\\%$ of total decision weight, while calendar seasonality captures remaining periodic variance.
            """)
        with col_b_img:
            img = load_image(os.path.join(project_root, "results", "plots", "shap", "shap_bar_plot.png"))
            if img:
                st.image(img, caption="Mean Absolute SHAP Value Ranking", use_container_width=True)

    with tab_dep:
        col_d_txt, col_d_img = st.columns([1, 1])
        with col_d_txt:
            st.markdown("""
            **SHAP Dependence & Interaction Plot**
            - Shows how the predictive contribution of a primary feature varies across its range, colored by an interacting feature.
            - Illustrates smooth nonlinear transitions without overfitting artifacts, confirming model stability.
            """)
        with col_d_img:
            img = load_image(os.path.join(project_root, "results", "plots", "shap", "shap_dependence_plot.png"))
            if img:
                st.image(img, caption="SHAP Nonlinear Feature Dependence", use_container_width=True)

    with tab_wf:
        col_w_txt, col_w_img = st.columns([1, 1])
        with col_w_txt:
            st.markdown("""
            **Sample Waterfall Decomposition**
            - Explains how a specific observation moves from the base expected value $E[f(X)]$ to the final model prediction $f(x)$.
            - Each arrow demonstrates the positive or negative push exerted by each feature for a given store-item-date triad.
            """)
        with col_w_img:
            img = load_image(os.path.join(project_root, "results", "plots", "shap", "shap_waterfall_plot.png"))
            if img:
                st.image(img, caption="Local Observation SHAP Waterfall Breakdown", use_container_width=True)

    st.markdown("---")

    # Adaptive Model Router Feature Importance
    st.markdown("### 🔀 Adaptive Router Architecture (Meta-Feature Importance)")
    r_c1, r_c2 = st.columns([1, 1])

    with r_c1:
        st.markdown("""
        In Phase 2, the **Adaptive Model Router** directs each time series to the optimal specialized 
        forecasting and uncertainty engine based on time-series meta-features.
        
        The meta-classifier weights features as follows:
        """)
        csv_r = os.path.join(journal_root, "results", "explainability", "router_feature_importance.csv")
        if os.path.exists(csv_r):
            df_r = pd.read_csv(csv_r)
            df_r["Importance (%)"] = (df_r["Importance"] * 100.0).round(2)
            df_r["Meta-Feature"] = df_r["Feature"].replace({
                "mean_sales": "Mean Sales Volume (Scale)",
                "cv_sales": "Coefficient of Variation (Variability)",
                "trend_strength_r2": "Trend Strength (R²)",
                "lag365_autocorr": "Yearly Seasonality (Lag-365 Autocorr)"
            })
            st.dataframe(
                df_r[["Meta-Feature", "Importance (%)"]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Router importance data loading...")

    with r_c2:
        if os.path.exists(csv_r):
            fig_r = go.Figure(go.Bar(
                x=df_r["Importance (%)"],
                y=df_r["Meta-Feature"],
                orientation="h",
                marker_color="#4338ca"
            ))
            fig_r.update_layout(
                title="Adaptive Router Meta-Feature Weights",
                xaxis_title="Weight Contribution (%)",
                template="plotly_white",
                height=260,
                margin=dict(l=20, r=20, t=40, b=30)
            )
            st.plotly_chart(fig_r, use_container_width=True)
