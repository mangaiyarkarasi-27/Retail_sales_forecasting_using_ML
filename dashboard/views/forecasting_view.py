"""
dashboard/views/forecasting_view.py
----------------------------------
Interactive Demand Forecasting module for E-IDSS.
Provides historical demand visualization, real-time direct multi-horizon
forecast vector generation (h=1..7), SBC demand profiling, and causal
feature architecture analysis for the active retail series.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import series_service


def render_forecasting_view(project_root: str, journal_root: str, helpers: dict):
    load_image = helpers["load_image"]

    active_store = st.session_state.get("active_store", 1)
    active_item = st.session_state.get("active_item", 1)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
                padding: 1.5rem 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;">
        <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">
            📈 Multi-Horizon Demand Forecasting
        </h2>
        <p style="color: #94a3b8; margin-top: 0.4rem; font-size: 0.95rem; margin-bottom: 0;">
            Direct decoupled multi-horizon forecasting architecture (M₁...M₇) operating strictly on 
            pre-forecast causal features (τ ≥ 7 days) to eliminate recursive error cascading.
        </p>
    </div>
    """, unsafe_allow_html=True)

    prof = series_service.get_series_profile(journal_root, active_store, active_item)
    sbc_category = prof.get("sbc_category", "Smooth")
    mean_sales = prof.get("mean_sales", 0.0)
    cv_sales = prof.get("cv_sales", 0.0)
    trend_r2 = prof.get("trend_r2", 0.0)

    st.markdown(f"""
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem; color: #334155;">
            <span>📍 <b>Active Series:</b> Store {active_store} · Item {active_item}</span>
            <span><b>SBC Category:</b> <span style="color:#0284c7; font-weight:600;">{sbc_category}</span></span>
            <span><b>Mean Velocity:</b> {mean_sales:.1f} units/day</span>
            <span><b>CV:</b> {cv_sales:.2f}</span>
            <span><b>Trend R²:</b> {trend_r2:.3f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Historical Series Data
    history_df = series_service.get_series_history(project_root, active_store, active_item)
    if history_df.empty:
        st.warning(f"No historical records found for Store {active_store}, Item {active_item}.")
        return

    # Horizon Range Controls
    st.markdown("### 📊 Interactive Time Series & Live 7-Day Forecast Vector")
    time_col1, time_col2 = st.columns([2, 1])
    with time_col1:
        range_choice = st.radio(
            "Historical Display Window:",
            options=["Last 90 Days", "Last 180 Days", "Last 365 Days", "Complete History (2013–2017)"],
            horizontal=True,
            index=1
        )
    with time_col2:
        st.caption("Forecasting Horizon: **h = 1 to 7 Days Ahead** (Decoupled Direct Inference)")

    if range_choice == "Last 90 Days":
        plot_df = history_df.tail(90).copy()
    elif range_choice == "Last 180 Days":
        plot_df = history_df.tail(180).copy()
    elif range_choice == "Last 365 Days":
        plot_df = history_df.tail(365).copy()
    else:
        plot_df = history_df.copy()

    # Generate genuine live multi-horizon forecast vector
    fc_vector_df = series_service.generate_multihorizon_forecast(journal_root, project_root, active_store, active_item)

    # Plot with Plotly
    fig = go.Figure()

    # Historical Line
    fig.add_trace(go.Scatter(
        x=plot_df["date"],
        y=plot_df["sales"],
        mode="lines",
        name="Historical Sales",
        line=dict(color="#1e293b", width=2),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual Sales:</b> %{y} units<extra></extra>"
    ))

    # Rolling 7-day average for context
    if len(plot_df) >= 7:
        plot_df["rolling_7"] = plot_df["sales"].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=plot_df["date"],
            y=plot_df["rolling_7"],
            mode="lines",
            name="7-Day Moving Avg",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>7-Day MA:</b> %{y:.1f} units<extra></extra>"
        ))

    # Connect forecast vector to last historical point
    if not fc_vector_df.empty:
        last_date = plot_df["date"].iloc[-1]
        last_val = plot_df["sales"].iloc[-1]
        fc_dates = [last_date] + list(fc_vector_df["target_date"])
        fc_vals = [last_val] + list(fc_vector_df["predicted_sales"])

        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=fc_vals,
            mode="lines+markers",
            name="Multi-Horizon Forecast (h=1..7)",
            line=dict(color="#2563eb", width=3, dash="dash"),
            marker=dict(size=8, color="#2563eb", symbol="diamond"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecast ŷ:</b> %{y:.1f} units<extra></extra>"
        ))

    fig.update_layout(
        title=f"Store {active_store} · Item {active_item}: Historical Daily Demand & Direct Forecast Vector",
        xaxis_title="Date",
        yaxis_title="Daily Units Sold",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast Breakdown Table & Lead-Time Demand
    st.markdown("### 📋 Multi-Horizon Prediction Vector Details")
    c_tbl, c_arch = st.columns([3, 2])

    with c_tbl:
        if not fc_vector_df.empty:
            display_fc = fc_vector_df.copy()
            display_fc["target_date_str"] = display_fc["target_date"].dt.strftime("%Y-%m-%d")
            display_table = pd.DataFrame({
                "Horizon": [f"h = {row['horizon']}" for _, row in display_fc.iterrows()],
                "Target Date": display_fc["target_date_str"],
                "Weekday": display_fc["target_day_name"],
                "Model ID": [f"M_{row['horizon']} (XGBoost)" for _, row in display_fc.iterrows()],
                "Point Forecast ŷ (Units)": display_fc["predicted_sales"],
                "Cumulative Demand D_L (Units)": display_fc["cumulative_demand"]
            })
            st.dataframe(
                display_table,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Generating multi-horizon vector...")

    with c_arch:
        st.markdown("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; height: 100%;">
            <h4 style="margin-top: 0; color: #1e293b; font-size: 1rem;">🛡️ Decision-Time-Safe Multi-Horizon Design</h4>
            <ul style="font-size: 0.85rem; color: #475569; padding-left: 1.2rem; line-height: 1.5; margin-bottom: 0;">
                <li><b>No Recursive Leakage:</b> Standard autoregressive models feed noisy predictions back as future lag inputs, causing error compounding.</li>
                <li><b>Direct Formulation:</b> Each horizon <i>h ∈ {1..7}</i> is predicted by a dedicated independent model <i>M_h(X_t)</i> trained directly on target <i>y_{t+h}</i>.</li>
                <li><b>Lag Cutoff Constraint:</b> All feature transformations strictly respect <code>τ ≥ 7 days</code> (e.g., <code>sales_lag_7</code>, <code>rolling_mean_7</code>).</li>
                <li><b>Immediate Execution:</b> All 7 lead-time predictions are resolved simultaneously at decision time <i>t</i> with zero iterative latency.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Model Performance & Benchmark Context
    st.markdown("### 🏆 Horizon-by-Horizon Validation Performance")
    col_perf1, col_perf2 = st.columns(2)

    with col_perf1:
        st.markdown("""
        The direct multi-horizon models were validated on out-of-sample data under chronological partitioning.
        Because all models observe identical causal history at <i>t</i>, accuracy remains robust across the entire 7-day planning window:
        """)
        
        # Load validation metrics from CSV if available
        val_csv = os.path.join(journal_root, "results", "forecasting", "direct_multihorizon_validation.csv")
        if os.path.exists(val_csv):
            val_df = pd.read_csv(val_csv)
            val_df["Horizon"] = [f"h = {h}" for h in val_df["horizon"]]
            val_df["MAE (Units)"] = val_df["MAE"].round(3)
            val_df["RMSE (Units)"] = val_df["RMSE"].round(3)
            val_df["Inference Latency"] = (val_df["prediction_time"] * 1000).round(1).astype(str) + " ms"
            
            st.dataframe(
                val_df[["Horizon", "MAE (Units)", "RMSE (Units)", "Inference Latency"]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.write("Validation metrics loading...")

    with col_perf2:
        pred_plot = load_image(os.path.join(project_root, "results", "plots", "actual_vs_predicted_v2.png"))
        if pred_plot:
            st.image(pred_plot, caption="Actual vs Predicted Demand: Tuned XGBoost Validation", use_container_width=True)
