"""
dashboard/views/home_view.py
----------------------------
Executive and operational home view for E-IDSS.
Provides a clear, modern landing interface with series context selection,
platform capabilities overview, key operational KPIs, and quick-action shortcuts.
"""

import os
import streamlit as st
import pandas as pd
import series_service


def render_home_view(project_root: str, journal_root: str, helpers: dict):
    load_image = helpers["load_image"]

    # Header Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #38bdf8 100%); 
                padding: 1.8rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.5rem; 
                box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);">
        <h1 style="color: white; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">
            E-IDSS | Intelligent Retail Decision Support System
        </h1>
        <p style="color: #e0f2fe; margin-top: 0.5rem; font-size: 1.05rem; font-weight: 400; max-width: 900px; line-height: 1.5;">
            An integrated software application combining decision-time-safe multi-horizon sales forecasting, 
            game-theoretic explainability, distribution-free conformal uncertainty calibration, and 
            discrete-event continuous-review replenishment simulation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Active Series State
    active_store = st.session_state.get("active_store", 1)
    active_item = st.session_state.get("active_item", 1)

    # Series Profile Snapshot
    prof = series_service.get_series_profile(journal_root, active_store, active_item)
    dec = series_service.get_series_decision_record(project_root, active_store, active_item)

    # Active Retail Series Workspace Banner
    st.markdown("### 🎯 Active Retail Series Workspace")
    sbc_cat = prof.get("sbc_category", "Smooth")
    mean_sales = prof.get("mean_sales", 0.0)
    cv_sales = prof.get("cv_sales", 0.0)
    dec_score = dec.get("Decision_Score", 50.0)
    risk_lvl = dec.get("Risk_Level", "Low Risk")
    priority_rank = dec.get("Priority_Rank", 1)
    action_text = dec.get("Operational_Action", "Maintain optimal replenishment")

    st.markdown(f"""
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.1rem 1.4rem; box-shadow: 0 2px 4px rgba(0,0,0,0.03); margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 800; color: #0f172a; font-size: 1.15rem;">
                📍 Store {active_store} · Item {active_item} Live Workspace
            </span>
            <span style="background: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 9999px; font-size: 0.82rem; font-weight: 700;">
                Portfolio Rank #{priority_rank} / 500
            </span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; font-size: 0.88rem; color: #334155; margin-bottom: 0.6rem;">
            <div><b>Demand Category:</b> <span style="color:#0284c7; font-weight:600;">{sbc_cat}</span></div>
            <div><b>Mean Velocity:</b> {mean_sales:.1f} units/day</div>
            <div><b>Demand CV:</b> {cv_sales:.2f}</div>
            <div><b>MCDA Score:</b> <span style="font-weight:700; color:#0f766e;">{dec_score:.1f} / 100</span> ({risk_lvl})</div>
        </div>
        <div style="font-size: 0.85rem; color: #475569; background: #f8fafc; padding: 0.5rem 0.8rem; border-radius: 6px; border-left: 3px solid #0284c7;">
            <b>Operational Recommendation:</b> <span style="color:#0f766e; font-weight:600;">{action_text}</span>
            <span style="float: right; color: #64748b; font-size: 0.78rem;">To switch series, use the sidebar selectors or Decision Priority ledger</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Workflow Actions
    st.markdown("### ⚡ Quick Navigation Shortcuts")
    btn_c1, btn_c2, btn_c3, btn_c4, btn_c5 = st.columns(5)
    with btn_c1:
        if st.button("📈 Demand Forecasting", use_container_width=True):
            st.session_state["nav_selection"] = "2. Demand Forecasting"
            st.rerun()
    with btn_c2:
        if st.button("📦 Inventory Intelligence", use_container_width=True):
            st.session_state["nav_selection"] = "3. Inventory Intelligence"
            st.rerun()
    with btn_c3:
        if st.button("🔍 Explainable AI", use_container_width=True):
            st.session_state["nav_selection"] = "4. Explainable AI"
            st.rerun()
    with btn_c4:
        if st.button("🛡️ Risk & Uncertainty", use_container_width=True):
            st.session_state["nav_selection"] = "5. Uncertainty and Risk Analysis"
            st.rerun()
    with btn_c5:
        if st.button("⚖️ Decision Priority", use_container_width=True):
            st.session_state["nav_selection"] = "6. Decision Support and Recommendations"
            st.rerun()

    st.markdown("---")

    # Enterprise Capabilities Grid
    st.markdown("### 🏛️ Platform Core Capabilities")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card success">
            <div class="metric-lbl">Direct Multi-Horizon Forecasting</div>
            <div class="metric-val">Horizons 1 to 7 Days</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height: 1.4;'>
                7 Decoupled global XGBoost models trained strictly on pre-forecast causal features (τ ≥ 7 days), 
                eliminating recursive error accumulation and look-ahead data leakage.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card info">
            <div class="metric-lbl">Discrete-Event Replenishment</div>
            <div class="metric-val">Continuous-Review Policy</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height: 1.4;'>
                Daily event-driven inventory simulation under lost-sales dynamics, converting conformal 
                uncertainty bounds into dynamic safety stocks across lead times L ∈ {1, 3, 5, 7}.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card warning">
            <div class="metric-lbl">Conformal Calibration & Shift</div>
            <div class="metric-val">Distribution-Free Bounds</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height: 1.4;'>
                Rolling forecast-conditional nonconformity scores scaled to demand volume, diagnosed against 
                a 3.55x–4.30x residual variance explosion on non-exchangeable test data.
            </p>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("""
        <div class="metric-card info">
            <div class="metric-lbl">Explainable AI (SHAP)</div>
            <div class="metric-val">Local & Global Attribution</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height: 1.4;'>
                Game-theoretic TreeExplainer SHAP values providing natural-language justifications for forecasts, 
                plus feature importance analysis for adaptive model routing.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="metric-card success">
            <div class="metric-lbl">Multi-Criteria Decision Engine</div>
            <div class="metric-val">0–100 MCDA Prioritization</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height: 1.4;'>
                Ranks all 500 store-item combinations by combining demand growth, trend momentum, 
                SHAP impact, seasonal spikes, and forecasting residual uncertainty.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div class="metric-card success">
            <div class="metric-lbl">Economic Frontier & Audit</div>
            <div class="metric-val">Service-Cost Optimization</div>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height: 1.4;'>
                Identifies unit penalty crossover points (p ≥ $2.00) where conformal replenishment dominates costs, 
                backed by 100% frozen reproducibility audits and Wilcoxon tests (p < 10⁻⁷⁵).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset & System Information Area
    st.markdown("### 📊 System Information & Benchmark Scope")
    col_inf1, col_inf2 = st.columns([1, 1])
    with col_inf1:
        st.markdown("""
        - **Dataset:** Kaggle Store Item Demand Forecasting Benchmark
        - **Total Records:** 913,000 daily observations across 5 years (2013-01-01 to 2017-12-31)
        - **Scale:** 10 Retail Stores × 50 Product Items = 500 Distinct Time Series
        - **Longitudinal Partitioning:**
          - *Training Split (2013–2015):* 547,500 rows for model fitting
          - *Validation Split (2016):* 183,000 rows for calibration neighborhood & hyperparameter sweeps
          - *Untouched Out-of-Sample Test (2017):* 182,500 rows for realistic temporal distribution shift evaluation
        """)
    with col_inf2:
        split_img = load_image(os.path.join(project_root, "results", "plots", "timeseries_split_visualization.png"))
        if split_img:
            st.image(split_img, caption="Longitudinal Chronological Split & Calibration Boundary", use_container_width=True)
