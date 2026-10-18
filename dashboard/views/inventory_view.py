"""
dashboard/views/inventory_view.py
--------------------------------
Inventory Intelligence & Replenishment Optimization module for E-IDSS.
Provides interactive continuous-review (s, S) replenishment parameter calculation,
series-specific discrete-event simulation performance outcomes, and 500-series
portfolio-level service-cost frontiers and economic sensitivity analysis.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import series_service


def render_inventory_view(project_root: str, journal_root: str, helpers: dict):
    load_image = helpers["load_image"]

    active_store = st.session_state.get("active_store", 1)
    active_item = st.session_state.get("active_item", 1)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #065f46 0%, #047857 50%, #059669 100%); 
                padding: 1.5rem 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
                box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.25);">
        <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">
            📦 Inventory Intelligence & Replenishment
        </h2>
        <p style="color: #d1fae5; margin-top: 0.4rem; font-size: 0.95rem; margin-bottom: 0;">
            Continuous-review (s, S) replenishment policy driven by conformal uncertainty bounds, 
            empirically evaluated under daily discrete-event lost-sales inventory simulation across 500 retail series.
        </p>
    </div>
    """, unsafe_allow_html=True)

    prof = series_service.get_series_profile(journal_root, active_store, active_item)
    sbc_cat = prof.get("sbc_category", "Smooth")
    mean_sales = prof.get("mean_sales", 0.0)

    st.markdown(f"""
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.88rem; color: #334155;">
            <span>📍 <b>Active Series:</b> Store {active_store} · Item {active_item}</span>
            <span><b>Demand Profile:</b> <span style="color:#059669; font-weight:600;">{sbc_cat}</span></span>
            <span><b>Mean Velocity:</b> {mean_sales:.1f} units/day</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Replenishment Calculator
    st.markdown("### ⚙️ Series Replenishment Parameters & Operating Policy")
    calc_c1, calc_c2, calc_c3, calc_c4 = st.columns(4)

    with calc_c1:
        lead_time = st.selectbox(
            "Lead Time (L in days)",
            options=[1, 3, 5, 7],
            index=0,
            key="inv_lt_select"
        )

    with calc_c2:
        shortage_penalty = st.selectbox(
            "Shortage Penalty ($/unit)",
            options=[0.60, 1.00, 1.50, 2.00, 3.00, 4.00],
            index=1,
            format_func=lambda x: f"${x:.2f} / unit",
            key="inv_p_select"
        )

    with calc_c3:
        st.metric("Holding Cost", "$0.10 / unit-day")

    with calc_c4:
        st.metric("Fixed Order Cost", "$2.00 / order")

    # Generate multi-horizon cumulative forecast
    fc_vector = series_service.generate_multihorizon_forecast(journal_root, project_root, active_store, active_item)
    if not fc_vector.empty and len(fc_vector) >= lead_time:
        cum_demand = float(fc_vector[fc_vector["horizon"] == lead_time]["cumulative_demand"].iloc[0])
    else:
        cum_demand = round(mean_sales * lead_time, 2)

    # Uncertainty metrics
    unc_record = series_service.get_series_uncertainty_metrics(journal_root, active_store, active_item, lead_time=lead_time, nominal=0.90)
    conformal_width = unc_record.get("mean_interval_width", mean_sales * 0.5 * np.sqrt(lead_time))
    conformal_safety_stock = round(float(conformal_width / 2.0), 1)
    reorder_point = round(cum_demand + conformal_safety_stock, 1)
    order_quantity = round(max(30.0, float(cum_demand * 1.5)), 0)

    # Policy Display Cards
    card1, card2, card3, card4 = st.columns(4)
    with card1:
        st.markdown(f"""
        <div class="metric-card info">
            <div class="metric-lbl">Lead-Time Forecast (D̂_L)</div>
            <div class="metric-val">{cum_demand:.1f} units</div>
            <small style='color:#64748b;'>Cumulative demand over L={lead_time}d</small>
        </div>
        """, unsafe_allow_html=True)

    with card2:
        st.markdown(f"""
        <div class="metric-card warning">
            <div class="metric-lbl">Conformal Safety Stock (SS)</div>
            <div class="metric-val">+{conformal_safety_stock:.1f} units</div>
            <small style='color:#64748b;'>90% Coverage Conformal Buffer</small>
        </div>
        """, unsafe_allow_html=True)

    with card3:
        st.markdown(f"""
        <div class="metric-card success">
            <div class="metric-lbl">Reorder Point (ROP = s)</div>
            <div class="metric-val">{reorder_point:.1f} units</div>
            <small style='color:#64748b;'>Trigger order when I + IP ≤ s</small>
        </div>
        """, unsafe_allow_html=True)

    with card4:
        st.markdown(f"""
        <div class="metric-card success">
            <div class="metric-lbl">Order Target Level (S)</div>
            <div class="metric-val">{reorder_point + order_quantity:.1f} units</div>
            <small style='color:#64748b;'>Order-up-to target level</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Discrete-Event Simulation Empirical Record for this series
    st.markdown("### 🧪 Empirical Discrete-Event Simulation Outcome (Active Series)")
    sim_res = series_service.get_series_simulation_results(
        journal_root, active_store, active_item, lead_time=lead_time, shortage_penalty=shortage_penalty
    )

    if sim_res:
        det_fr = sim_res["det_fill_rate"] * 100.0
        conf_fr = sim_res["conf_fill_rate"] * 100.0
        delta_fr = sim_res["delta_fill_rate_pp"]
        shortage_prevented = sim_res["units_prevented"]
        cost_delta = sim_res["cost_delta"]

        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        with res_c1:
            st.metric("Fill Rate (Deterministic)", f"{det_fr:.1f}%")
        with res_c2:
            st.metric("Fill Rate (Conformal)", f"{conf_fr:.1f}%", delta=f"+{delta_fr:.1f}% pp")
        with res_c3:
            st.metric("Stockout Units Prevented", f"{shortage_prevented:,.0f} units")
        with res_c4:
            st.metric("Net Cost Delta", f"${cost_delta:+,.2f}", delta_color="inverse")

        # Visual Cost Breakdown Comparison
        sim_col1, sim_col2 = st.columns([3, 2])
        with sim_col1:
            cost_fig = go.Figure()
            cost_fig.add_trace(go.Bar(
                name="Deterministic Policy (Point Forecast Only)",
                x=["Holding Cost", "Shortage Penalty", "Ordering Cost", "Total Cost"],
                y=[sim_res["det_holding_cost"], sim_res["det_shortage_cost"], sim_res["det_ordering_cost"], sim_res["det_total_cost"]],
                marker_color="#94a3b8"
            ))
            cost_fig.add_trace(go.Bar(
                name="Conformal Policy (Uncertainty-Aware Buffer)",
                x=["Holding Cost", "Shortage Penalty", "Ordering Cost", "Total Cost"],
                y=[sim_res["conf_holding_cost"], sim_res["conf_shortage_cost"], sim_res["conf_ordering_cost"], sim_res["conf_total_cost"]],
                marker_color="#059669"
            ))
            cost_fig.update_layout(
                title=f"Annual Cost Breakdown: Store {active_store} · Item {active_item} (L={lead_time}d, p=${shortage_penalty:.2f})",
                yaxis_title="Annual Operating Cost ($)",
                barmode="group",
                template="plotly_white",
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(cost_fig, use_container_width=True)

        with sim_col2:
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 1.1rem; height: 100%;">
                <h4 style="color: #166534; margin-top: 0; font-size: 0.95rem;">💡 Policy Mechanics & Trade-Off</h4>
                <p style="font-size: 0.85rem; color: #15803d; line-height: 1.5;">
                    Under the <b>Deterministic baseline</b>, the replenishment trigger relies solely on expected point demand <i>D̂_L</i>.
                    When demand spikes or residuals expand, stock is quickly exhausted, resulting in <b>{sim_res['det_shortage_units']:,.0f} lost sales units</b>.
                </p>
                <p style="font-size: 0.85rem; color: #15803d; line-height: 1.5; margin-bottom: 0;">
                    The <b>Conformal Uncertainty policy</b> dynamically sizes safety stock using distribution-free prediction bounds.
                    This lifts customer service from <b>{det_fr:.1f}% to {conf_fr:.1f}%</b>, successfully avoiding 
                    <b>{shortage_prevented:,.0f} lost-sales events</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Simulation outcome for this specific parameter combination is loading...")

    st.markdown("---")

    # Portfolio-Level Inventory Intelligence (500 Series Evaluation)
    st.markdown("### 🌐 Portfolio-Level Inventory Evaluation (500 Retail Series)")
    tab_lt, tab_pen, tab_cat = st.tabs([
        "Lead Time Sensitivity (Table 7)", 
        "Shortage Penalty & Cost Crossover (Table 9)", 
        "Business Category Diagnostics (Table 8)"
    ])

    with tab_lt:
        col_lt1, col_lt2 = st.columns([3, 2])
        with col_lt1:
            st.markdown("""
            **Table 7: Multi-Lead-Time Fill Rate & Shortage Performance (Portfolio Scale)**  
            Demand-weighted and series-level fill rates across all 500 store-item series under continuous-review simulation:
            """)
            lt_csv = os.path.join(journal_root, "results", "inventory", "exp_019", "lead_time_comparison.csv")
            if os.path.exists(lt_csv):
                lt_df = pd.read_csv(lt_csv)
                # Filter for baseline penalty $1.00
                lt_df = lt_df[np.isclose(lt_df["shortage_penalty"], 1.0)]
                display_lt = []
                for lt_val in [1, 3, 5, 7]:
                    det_sub = lt_df[(lt_df["lead_time"] == lt_val) & (lt_df["policy"] == "Deterministic")].iloc[0]
                    conf_sub = lt_df[(lt_df["lead_time"] == lt_val) & (lt_df["policy"] == "Conformal_Uncertainty")].iloc[0]
                    display_lt.append({
                        "Lead Time": f"L = {lt_val} Days",
                        "Det. Demand Fill Rate": f"{det_sub['fill_rate']*100:.1f}%",
                        "Conf. Demand Fill Rate": f"{conf_sub['fill_rate']*100:.1f}%",
                        "Fill Rate Gain": f"+{(conf_sub['fill_rate'] - det_sub['fill_rate'])*100:.1f}% pp",
                        "Det. Shortage Units": f"{det_sub['shortage_units']:,.0f}",
                        "Conf. Shortage Units": f"{conf_sub['shortage_units']:,.0f}",
                        "Units Prevented": f"{det_sub['shortage_units'] - conf_sub['shortage_units']:,.0f}"
                    })
                st.dataframe(pd.DataFrame(display_lt), hide_index=True, use_container_width=True)

        with col_lt2:
            fr_plot = load_image(os.path.join(journal_root, "results", "inventory", "exp_019", "fill_rate_by_lead_time.png"))
            if fr_plot:
                st.image(fr_plot, caption="Portfolio Fill Rate Improvement Across Lead Times L ∈ {1, 3, 5, 7}", use_container_width=True)

    with tab_pen:
        col_pen1, col_pen2 = st.columns([3, 2])
        with col_pen1:
            st.markdown("""
            **Table 9: Economic Frontier & Unit Shortage Penalty Sensitivity**  
            Demonstrating the empirical cost crossover at $p \\ge \\$2.00$:
            """)
            st.markdown("""
            > [!IMPORTANT]
            > **Honest Scientific Disclosure:** When stockout penalty is low ($p \\le \\$1.00$), holding additional inventory 
            > costs more than the saved penalty. However, once stockout consequences exceed **$p \\ge \\$2.00 / unit**, 
            > conformal uncertainty replenishment achieves decisive net operating cost savings (saving up to **-$1,564,888** at $p=\\$4.00$).
            """)

            pen_csv = os.path.join(journal_root, "results", "inventory", "exp_019", "shortage_penalty_sensitivity.csv")
            if os.path.exists(pen_csv):
                pen_df = pd.read_csv(pen_csv)
                # Show L=3 as representative
                pen_sub = pen_df[pen_df["lead_time"] == 3]
                display_pen = []
                for p_val in sorted(pen_sub["shortage_penalty"].unique()):
                    d_r = pen_sub[(pen_sub["shortage_penalty"] == p_val) & (pen_sub["policy"] == "Deterministic")].iloc[0]
                    c_r = pen_sub[(pen_sub["shortage_penalty"] == p_val) & (pen_sub["policy"] == "Conformal_Uncertainty")].iloc[0]
                    delta_cost = c_r["total_cost"] - d_r["total_cost"]
                    display_pen.append({
                        "Unit Penalty (p)": f"${p_val:.2f} / unit",
                        "Det. Operating Cost": f"${d_r['total_cost']:,.0f}",
                        "Conf. Operating Cost": f"${c_r['total_cost']:,.0f}",
                        "Net Cost Delta": f"${delta_cost:+,.0f}",
                        "Cost Dominance": "Conformal Cost-Optimal" if delta_cost < 0 else "Deterministic Lower Cost"
                    })
                st.dataframe(pd.DataFrame(display_pen), hide_index=True, use_container_width=True)

        with col_pen2:
            pen_plot = load_image(os.path.join(journal_root, "results", "inventory", "exp_019", "shortage_penalty_sensitivity.png"))
            if pen_plot:
                st.image(pen_plot, caption="Economic Crossover: Total Operating Cost vs Shortage Penalty", use_container_width=True)

    with tab_cat:
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("""
            **Table 8: Cost Breakdown by Business Demand Category (SBC Classification)**  
            Conformal uncertainty delivers targeted protection across diverse demand profiles:
            - **Smooth Demand:** Steady velocity, moderate variance, baseline safety buffers sufficient.
            - **Erratic Demand:** High demand variability; conformal intervals widen dynamically to prevent catastrophic stockouts.
            - **Intermittent & Lumpy:** Zero-inflated demand patterns; adaptive nonconformity scores prevent severe overstocking while safeguarding service.
            """)
        with col_c2:
            cost_bar_plot = load_image(os.path.join(journal_root, "results", "inventory", "exp_019", "cost_breakdown_bars.png"))
            if cost_bar_plot:
                st.image(cost_bar_plot, caption="Operating Cost Components by Demand Category", use_container_width=True)
