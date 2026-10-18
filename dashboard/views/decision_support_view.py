"""
dashboard/views/decision_support_view.py
---------------------------------------
Decision Support & Priority Recommendations module for E-IDSS.
Provides multi-criteria portfolio ranking (MCDA), priority quadrant analysis,
operational replenishment action cards, and cross-application series synchronization.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import series_service


def render_decision_support_view(project_root: str, journal_root: str, helpers: dict):
    load_image = helpers["load_image"]

    active_store = st.session_state.get("active_store", 1)
    active_item = st.session_state.get("active_item", 1)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f766e 0%, #0d9488 50%, #14b8a6 100%); 
                padding: 1.5rem 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
                box-shadow: 0 10px 25px -5px rgba(13, 148, 136, 0.25);">
        <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">
            ⚖️ Decision Support & Priority Recommendations
        </h2>
        <p style="color: #ccfbf1; margin-top: 0.4rem; font-size: 0.95rem; margin-bottom: 0;">
            Multi-Criteria Decision Analysis (MCDA) synthesizing multi-horizon forecasts, 
            demand growth trends, SHAP feature impact, and forecast risk into prescriptive actions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load complete priority ranking dataset
    ranking_path = os.path.join(project_root, "results", "decision_intelligence", "priority_ranking.csv")
    if not os.path.exists(ranking_path):
        st.error("Priority ranking dataset not found.")
        return

    ranking_df = pd.read_csv(ranking_path)

    # Active Series Focus Card
    active_row = ranking_df[(ranking_df["Store"] == active_store) & (ranking_df["Item"] == active_item)]
    if not active_row.empty:
        r = active_row.iloc[0]
        rank_val = r["Priority_Rank"]
        score_val = r["Decision_Score"]
        prio_lvl = r["Priority_Level"]
        risk_lvl = r["Risk_Level"]
        biz_cat = r["Business_Category"]
        action_txt = r["Operational_Action"]
        shap_txt = r.get("Explanation_SHAP") or r.get("SHAP_Explanation", "")

        st.markdown(f"""
        <div style="background: white; border: 2px solid #0d9488; border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <span style="font-size: 1.15rem; font-weight: 800; color: #134e4a;">
                    🎯 Active Series Focus: Store {active_store} · Item {active_item}
                </span>
                <span style="background: #ccfbf1; color: #0f766e; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">
                    Portfolio Priority Rank #{rank_val} of 500
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 0.8rem; font-size: 0.85rem;">
                <div><b>Decision Score:</b> <span style="font-size:1.05rem; font-weight:700; color:#0d9488;">{score_val:.1f} / 100</span></div>
                <div><b>Priority Level:</b> <span style="font-weight:600;">{prio_lvl}</span></div>
                <div><b>Business Category:</b> <span style="font-weight:600;">{biz_cat}</span></div>
                <div><b>Risk Category:</b> <span style="font-weight:600;">{risk_lvl}</span></div>
            </div>
            <div style="background: #f0fdfa; border-left: 4px solid #0d9488; padding: 0.75rem 1rem; border-radius: 0 6px 6px 0;">
                <b style="color: #134e4a;">Prescribed Operational Action:</b> {action_txt}<br>
                <small style="color: #0f766e;"><b>Attribution:</b> {shap_txt}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Series Switcher / Synchronization
    st.markdown("### 🔄 Set Active Workspace Series from Portfolio")
    sync_c1, sync_c2 = st.columns([3, 1])
    with sync_c1:
        # Create a user-friendly selection list sorted by Rank
        sorted_ranking = ranking_df.sort_values("Priority_Rank").copy()
        options_list = [
            f"Rank #{row['Priority_Rank']} | Store {row['Store']} - Item {row['Item']} | Score: {row['Decision_Score']:.1f} | {row['Business_Category']}"
            for _, row in sorted_ranking.iterrows()
        ]
        # Current index
        current_opt_prefix = f"Store {active_store} - Item {active_item} "
        curr_idx = 0
        for i, opt in enumerate(options_list):
            if f"Store {active_store} - Item {active_item} " in opt:
                curr_idx = i
                break

        selected_option = st.selectbox(
            "Select Series from Portfolio Ranking:",
            options=options_list,
            index=curr_idx,
            key="prio_series_sync_select"
        )

    with sync_c2:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        if st.button("📌 Set as Active Series", use_container_width=True):
            # Parse Store and Item from selected option
            # format: "Rank #... | Store X - Item Y | ..."
            parts = selected_option.split("|")[1].strip()
            s_num = int(parts.split("Store")[1].split("-")[0].strip())
            i_num = int(parts.split("Item")[1].strip())
            st.session_state["active_store"] = s_num
            st.session_state["active_item"] = i_num
            st.session_state["sb_global_store"] = s_num
            st.session_state["sb_global_item"] = i_num
            st.success(f"Workspace synchronized to Store {s_num}, Item {i_num}!")
            st.rerun()

    st.markdown("---")

    # Visual Decision Analytics: Quadrant & Store Breakdown
    st.markdown("### 📊 Decision Space & Portfolio Distribution")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Scatter: Decision Score vs Risk Score
        fig_scatter = px.scatter(
            ranking_df,
            x="Risk_Score",
            y="Decision_Score",
            color="Priority_Level",
            size="Historical_Average",
            hover_data=["Store", "Item", "Business_Category", "Priority_Rank"],
            color_discrete_map={
                "High Priority": "#ef4444",
                "Medium Priority": "#f59e0b",
                "Low Priority": "#10b981"
            },
            title="MCDA Prioritization Quadrant: Decision Score vs Risk Score",
            labels={"Risk_Score": "Forecast Volatility / Risk Score", "Decision_Score": "Composite Decision Score"}
        )
        fig_scatter.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=40, r=30, t=50, b=40)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with viz_col2:
        # High Priority Count by Store
        hp_by_store = ranking_df[ranking_df["Priority_Level"] == "High Priority"].groupby("Store").size().reset_index(name="High_Priority_Count")
        fig_store = px.bar(
            hp_by_store,
            x="Store",
            y="High_Priority_Count",
            title="Distribution of High-Priority Series Across Stores",
            labels={"Store": "Retail Store ID", "High_Priority_Count": "Count of High-Priority Items"},
            color="High_Priority_Count",
            color_continuous_scale="Teal"
        )
        fig_store.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=40, r=30, t=50, b=40)
        )
        st.plotly_chart(fig_store, use_container_width=True)

    st.markdown("---")

    # Complete Interactive Filterable Ranking Table
    st.markdown("### 📋 Complete 500-Series Multi-Criteria Priority Ledger")
    
    # Filter controls
    f_c1, f_c2, f_c3 = st.columns(3)
    with f_c1:
        store_filter = st.selectbox("Filter Store:", options=["All Stores"] + [f"Store {s}" for s in range(1, 11)], index=0)
    with f_c2:
        prio_filter = st.selectbox("Filter Priority Level:", options=["All Priorities", "High Priority", "Medium Priority", "Low Priority"], index=0)
    with f_c3:
        cat_filter = st.selectbox(
            "Filter Category:", 
            options=["All Categories"] + sorted(list(ranking_df["Business_Category"].unique())), 
            index=0
        )

    filtered_df = ranking_df.copy()
    if store_filter != "All Stores":
        s_val = int(store_filter.split(" ")[1])
        filtered_df = filtered_df[filtered_df["Store"] == s_val]
    if prio_filter != "All Priorities":
        filtered_df = filtered_df[filtered_df["Priority_Level"] == prio_filter]
    if cat_filter != "All Categories":
        filtered_df = filtered_df[filtered_df["Business_Category"] == cat_filter]

    display_cols = [
        "Priority_Rank", "Store", "Item", "Decision_Score", "Priority_Level", 
        "Business_Category", "Risk_Level", "Operational_Action"
    ]
    st.dataframe(
        filtered_df[display_cols].sort_values("Priority_Rank"),
        hide_index=True,
        use_container_width=True
    )

    # Export Download Button
    csv_bytes = ranking_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Full 500-Series Priority Rankings (CSV)",
        data=csv_bytes,
        file_name="e_idss_500_series_priority_rankings.csv",
        mime="text/csv"
    )
