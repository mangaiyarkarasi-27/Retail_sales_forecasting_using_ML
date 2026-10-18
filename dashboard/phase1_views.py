"""
dashboard/phase1_views.py
-------------------------
Contains all 9 original pages for Phase 1 (Conference Baseline: IEEE E-IDSS).
Preserves the exact existing functionality, charts, tables, and narrative from the conference project.
"""

import os
import streamlit as st
import pandas as pd
from PIL import Image


def render_phase1_page(page_name: str, project_root: str, helpers: dict):
    load_csv = helpers["load_csv"]
    load_txt = helpers["load_txt"]
    load_image = helpers["load_image"]

    train_path = os.path.join(project_root, "data", "processed_train.csv")
    recs_path = os.path.join(project_root, "results", "recommendations", "recommendation_summary.csv")
    priority_path = os.path.join(project_root, "results", "decision_intelligence", "priority_ranking.csv")
    comparison_path = os.path.join(project_root, "results", "metrics", "model_comparison_before_after.csv")
    exe_report_path = os.path.join(project_root, "results", "decision_intelligence", "executive_decision_report.txt")
    plots_dir = os.path.join(project_root, "results", "plots")
    di_dir = os.path.join(project_root, "results", "decision_intelligence")

    # ----------------- PAGE 1: HOME -----------------
    if page_name == "1. Home":
        st.subheader("Project Overview & Dashboard Home (Phase 1 Baseline)")
        
        st.markdown("""
        Welcome to the portal for **An Explainable Intelligent Decision Support System for Retail Sales Forecasting Using Machine Learning and Explainable Analytics** (IEEE Conference Baseline).
        
        This system bridges the gap between raw statistical forecasts and corporate inventory optimization. Traditional models output demand numbers and stop; our system evaluates growth dynamics, local feature impacts (SHAP), and forecasting residuals to rank opportunities, flag risks, and justify replenishment actions.
        """)
        
        # KPI row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Total Records Analyzed</div>
                <div class="metric-val">913,000 Sales Rows</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Scope / Dimensions</div>
                <div class="metric-val">10 Stores & 50 Items</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Best Model Selected</div>
                <div class="metric-val">XGBoost Regressor (v2)</div>
            </div>
            """, unsafe_allow_html=True)
            
        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Model R² Score (v2)</div>
                <div class="metric-val">0.9332</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Model MAE (v2)</div>
                <div class="metric-val">6.2698 Units</div>
            </div>
            """, unsafe_allow_html=True)
        with col6:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Model RMSE (v2)</div>
                <div class="metric-val">8.1708 Units</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Project Objective")
        st.info("""
        **Primary Goal**: To build a research-grade, explainable intelligent decision support system (E-IDSS) capable of:
        1. Prioritizing stocking levels across supply chain structures.
        2. Quantifying operational business risks dynamically.
        3. Generating natural explanations of decisions using game-theoretic SHAP values.
        4. Producing executive-ready reporting suitable for retail integrations.
        """)
        
        st.markdown("### System Architecture")
        st.markdown("""
        - **Step 1: Data Preprocessing**: Standardizes raw data, splits time components.
        - **Step 2: EDA**: Details monthly/yearly trends, seasonality, and heatmaps.
        - **Step 3: Machine Learning Forecasting**: Trains Random Forest & XGBoost with advanced time-series lags and rolling window statistics.
        - **Step 4: Explainable AI (SHAP)**: Leverages TreeExplainer to break down local and global predictions.
        - **Step 5: Operational Recommendation Engine**: Outputs basic inventory rules (lags/rolling thresholds).
        - **Step 6: Decision Intelligence Framework**: Formulates Decision and Risk scores, Priority rankings, and Business category matrices.
        """)

    # ----------------- PAGE 2: DATASET OVERVIEW -----------------
    elif page_name == "2. Dataset Overview":
        st.subheader("Dataset Structure & Statistics")
        
        if os.path.exists(train_path):
            with st.spinner("Loading dataset overview..."):
                df_processed = load_csv(train_path)
                
            if df_processed is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Metadata & Shape")
                    st.write(f"- **Row Count**: {df_processed.shape[0]:,}")
                    st.write(f"- **Column Count**: {df_processed.shape[1]}")
                    st.write(f"- **Date Range**: {df_processed['date'].min()} to {df_processed['date'].max()}")
                    st.write(f"- **Unique Stores**: {df_processed['store'].nunique()} stores")
                    st.write(f"- **Unique Items**: {df_processed['item'].nunique()} items")
                
                with col2:
                    st.markdown("#### Missing Value Summary")
                    nulls = df_processed.isnull().sum().to_frame("Missing Count")
                    st.dataframe(nulls, use_container_width=True)
                    
                st.markdown("#### Feature List & Definitions")
                features_desc = {
                    "date": "Transaction timestamp (datetime format).",
                    "store": "Categorical Store ID (1-10).",
                    "item": "Categorical Product Item ID (1-50).",
                    "sales": "Actual units sold (Target Variable).",
                    "year/month/day/weekday/weekofyear/quarter": "Engineered calendar date variables.",
                    "is_weekend": "Binary flag (1 = Saturday/Sunday, 0 = Weekday).",
                    "month_start/month_end": "Binary flags representing calendar month boundaries.",
                    "sales_lag_7/14/30": "Shifted sales values representing sales exactly 7, 14, and 30 days ago.",
                    "rolling_mean_7/30": "Rolling averages representing demand trends, calculated per store-item combination."
                }
                st.dataframe(pd.DataFrame(list(features_desc.items()), columns=["Feature Name", "Definition"]), use_container_width=True)
                
                st.markdown("#### Statistical Description")
                st.dataframe(df_processed.describe(), use_container_width=True)
                
                st.markdown("#### Sample Dataset Preview")
                st.dataframe(df_processed.head(100), use_container_width=True)
                
                csv = df_processed.head(1000).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Sample Processed CSV (Top 1000 Rows)", data=csv, file_name="processed_train_sample.csv", mime="text/csv")
        else:
            st.warning(f"Cleaned training file not found at {train_path}.")

    # ----------------- PAGE 3: EDA ANALYTICS -----------------
    elif page_name == "3. EDA Analytics":
        st.subheader("Exploratory Data Analysis Plots")
        
        tab_temp, tab_dist, tab_stats = st.tabs([
            "📅 Temporal Trends", 
            "🏪 Store & Item Distributions", 
            "📊 Statistical Analytics"
        ])
        
        with tab_temp:
            col1, col2 = st.columns(2)
            with col1:
                img = load_image(os.path.join(plots_dir, "overall_sales_trend.png"))
                if img:
                    st.image(img, caption="Overall Sales Trend Over Time", use_container_width=True)
                    st.markdown("**Interpretation**: Displays a distinct yearly upward growth trend overlaid with consistent annual seasonality peaks (occurring in summer).")
                else:
                    st.info("overall_sales_trend.png not found.")
                    
            with col2:
                img = load_image(os.path.join(plots_dir, "monthly_sales_trend.png"))
                if img:
                    st.image(img, caption="Monthly Sales Trend", use_container_width=True)
                    st.markdown("**Interpretation**: Illustrates peak monthly demand during July, with January and December exhibiting the lowest baseline sales volumes.")
                else:
                    st.info("monthly_sales_trend.png not found.")
                    
            st.markdown("---")
            img = load_image(os.path.join(plots_dir, "yearly_sales_trend.png"))
            if img:
                st.image(img, caption="Yearly Sales Trend", use_container_width=True)
                st.markdown("**Interpretation**: Confirms year-over-year increases in cumulative sales volume, indicating organic business expansion.")
                
        with tab_dist:
            col1, col2 = st.columns(2)
            with col1:
                img = load_image(os.path.join(plots_dir, "store_sales_distribution.png"))
                if img:
                    st.image(img, caption="Store-wise Sales Distribution", use_container_width=True)
                    st.markdown("**Interpretation**: Box plot displaying sales variability across the 10 stores. Stores 2 and 8 are top-performing hubs, while Store 5 shows lower median demand.")
                else:
                    st.info("store_sales_distribution.png not found.")
                    
            with col2:
                img = load_image(os.path.join(plots_dir, "item_sales_distribution.png"))
                if img:
                    st.image(img, caption="Item-wise Sales Distribution", use_container_width=True)
                    st.markdown("**Interpretation**: Shows variance in sales quantities across the 50 product items, illustrating that product catalog demand varies systematically.")
                else:
                    st.info("item_sales_distribution.png not found.")
                    
            st.markdown("---")
            img = load_image(os.path.join(plots_dir, "top_10_items.png"))
            if img:
                st.image(img, caption="Top 10 Best Selling Items", use_container_width=True)
                st.markdown("**Interpretation**: Pinpoints the highest-revenue products. Item 15 is the highest selling item overall, followed closely by Item 28.")
                
        with tab_stats:
            col1, col2 = st.columns(2)
            with col1:
                img = load_image(os.path.join(plots_dir, "correlation_heatmap.png"))
                if img:
                    st.image(img, caption="Feature Correlation Heatmap", use_container_width=True)
                    st.markdown("**Interpretation**: Shows correlation relationships. Rolling averages and lag variables exhibit extremely strong linear correlation with sales, indicating high forecasting importance.")
                else:
                    st.info("correlation_heatmap.png not found.")
                    
            with col2:
                img = load_image(os.path.join(plots_dir, "sales_distribution_histogram.png"))
                if img:
                    st.image(img, caption="Sales Distribution Histogram", use_container_width=True)
                    st.markdown("**Interpretation**: Shows the frequency curve of sales counts. The sales volume exhibits a right-skewed distribution centered around 40-50 units per day.")
                else:
                    st.info("sales_distribution_histogram.png not found.")

    # ----------------- PAGE 4: FORECASTING PERFORMANCE -----------------
    elif page_name == "4. Forecasting Performance":
        st.subheader("Model Performance Evaluation (v2 vs v1)")
        
        st.markdown("""
        Advanced feature engineering (lags, rolling statistics, calendar indicators) was introduced in **Forecasting v2** to replace base calendar-only forecasting features.
        This change drastically improved the system's performance, as outlined in the comparative summary below:
        """)
        
        if os.path.exists(comparison_path):
            df_comp = load_csv(comparison_path)
            if df_comp is not None:
                st.dataframe(df_comp, use_container_width=True)
                
                # Display best model in clean cards
                st.markdown("### Best Model KPI Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class="metric-card success">
                        <div class="metric-lbl">Best Selected Model</div>
                        <div class="metric-val">XGBoost Regressor (v2)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="metric-card success">
                        <div class="metric-lbl">Best Model R² Score</div>
                        <div class="metric-val">0.9332</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div class="metric-card success">
                        <div class="metric-lbl">Best Model MAE</div>
                        <div class="metric-val">6.2698 Units</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Model comparison data not found.")
            
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            img = load_image(os.path.join(plots_dir, "actual_vs_predicted_v2.png"))
            if img:
                st.image(img, caption="Actual vs Predicted Sales Plot (v2)", use_container_width=True)
            else:
                st.info("actual_vs_predicted_v2.png not found.")
                
        with col_plot2:
            img = load_image(os.path.join(plots_dir, "feature_importance.png"))
            if img:
                st.image(img, caption="Model Feature Importances (v2)", use_container_width=True)
            else:
                st.info("feature_importance.png not found.")

    # ----------------- PAGE 5: SHAP EXPLAINABILITY -----------------
    elif page_name == "5. SHAP Explainability":
        st.subheader("SHAP Interpretability Diagnostics")
        
        st.markdown("Understand the global drivers of sales predictions and the local features backing individual nodes.")
        
        col_shap1, col_shap2 = st.columns(2)
        with col_shap1:
            img = load_image(os.path.join(plots_dir, "shap", "shap_summary_plot.png"))
            if img:
                st.image(img, caption="SHAP Summary Plot (Beeswarm)", use_container_width=True)
                st.markdown("**Beeswarm Interpretation**: Shows impact distribution. Features like `sales_lag_7` and `rolling_mean_30` are primary drivers. Higher values of lag sales heavily push forecasted sales upwards.")
            else:
                st.info("shap_summary_plot.png not found.")
                
        with col_shap2:
            img = load_image(os.path.join(plots_dir, "shap", "shap_bar_plot.png"))
            if img:
                st.image(img, caption="SHAP Bar Plot (Global Importance)", use_container_width=True)
                st.markdown("**Global Bar Interpretation**: Ranks absolute impact. `sales_lag_7` exerts the highest mean absolute SHAP impact, verifying the importance of weekly cycles.")
            else:
                st.info("shap_bar_plot.png not found.")
                
        st.markdown("---")
        col_shap3, col_shap4 = st.columns(2)
        with col_shap3:
            img = load_image(os.path.join(plots_dir, "shap", "shap_dependence_plot.png"))
            if img:
                st.image(img, caption="SHAP Dependence Plot (sales_lag_7)", use_container_width=True)
                st.markdown("**Dependence Interpretation**: Shows relation between `sales_lag_7` and its impact. There is a strong linear relationship; higher past sales directly escalate predicted output.")
            else:
                st.info("shap_dependence_plot.png not found.")
                
        with col_shap4:
            img = load_image(os.path.join(plots_dir, "shap", "shap_waterfall_plot.png"))
            if img:
                st.image(img, caption="SHAP Waterfall Plot (Local Forecast Justification)", use_container_width=True)
                st.markdown("**Waterfall Interpretation**: Explains a single node's forecast. It shows how the base expectation ($E[f(x)]$ units) shifts to the final prediction ($f(x)$ units) due to the presence of specific local features.")
            else:
                st.info("shap_waterfall_plot.png not found.")
                
        st.markdown("### Model Decisions Business Report")
        shap_txt = load_txt(os.path.join(project_root, "results", "metrics", "shap_report.txt"))
        with st.expander("📄 Click to View Full SHAP Text Explanations"):
            st.markdown(f'<div class="report-block">{shap_txt}</div>', unsafe_allow_html=True)

    # ----------------- PAGE 6: RECOMMENDATION ENGINE -----------------
    elif page_name == "6. Recommendation Engine":
        st.subheader("Operational Inventory Recommendations")
        
        if os.path.exists(recs_path):
            recs_df = load_csv(recs_path)
            if recs_df is not None:
                # Action counts
                increase_cnt = (recs_df["Primary_Action"] == "Increase inventory levels").sum()
                reduce_cnt = (recs_df["Primary_Action"] == "Reduce procurement volume").sum()
                maintain_cnt = (recs_df["Primary_Action"] == "Maintain optimal inventory").sum()
                
                # KPI Cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card success">
                        <div class="metric-lbl">Increase Inventory</div>
                        <div class="metric-val">{increase_cnt} Nodes</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card critical">
                        <div class="metric-lbl">Reduce Procurement</div>
                        <div class="metric-val">{reduce_cnt} Nodes</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card info">
                        <div class="metric-lbl">Maintain Inventory</div>
                        <div class="metric-val">{maintain_cnt} Nodes</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                col_chart, col_explain = st.columns([2, 1])
                with col_chart:
                    img = load_image(os.path.join(project_root, "results", "recommendations", "recommendation_distribution.png"))
                    if img:
                        st.image(img, caption="Operational Action Distribution", use_container_width=True)
                    else:
                        st.info("recommendation_distribution.png not found.")
                with col_explain:
                    st.markdown("#### Recommendation Rules (±5% Thresholds)")
                    st.info("""
                    - **Increase Inventory**: Triggered when predicted demand growth is > 5% above the historical baseline average.
                    - **Reduce Procurement**: Triggered when forecasted demand falls < -5% compared to the baseline.
                    - **Maintain Optimal**: Standard replacement buffer when predicted demand remains within the ±5% baseline interval.
                    """)
                    
                # Filters
                st.markdown("### Filter Recommendations")
                store_filter = st.selectbox("Select Store", ["All"] + sorted(recs_df["Store"].unique().tolist()))
                item_filter = st.selectbox("Select Item", ["All"] + sorted(recs_df["Item"].unique().tolist()))
                
                filtered_recs = recs_df.copy()
                if store_filter != "All":
                    filtered_recs = filtered_recs[filtered_recs["Store"] == int(store_filter)]
                if item_filter != "All":
                    filtered_recs = filtered_recs[filtered_recs["Item"] == int(item_filter)]
                    
                st.dataframe(filtered_recs, use_container_width=True)
                
                csv = filtered_recs.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Filtered Recommendations CSV", data=csv, file_name="filtered_recommendations.csv", mime="text/csv", use_container_width=True)
        else:
            st.warning("Recommendation summary file not found.")

    # ----------------- PAGE 7: DECISION INTELLIGENCE FRAMEWORK -----------------
    elif page_name == "7. Decision Intelligence Framework":
        st.subheader("Decision Intelligence & Multi-Criteria Prioritization (E-IDSS)")
        
        st.markdown("""
        The E-IDSS framework scores nodes on a 0-100 scale using Decision Scores and Risk Scores, establishing operational priority classes.
        """)
        
        col_di1, col_di2 = st.columns(2)
        with col_di1:
            img = load_image(os.path.join(di_dir, "decision_score_distribution.png"))
            if img:
                st.image(img, caption="E-IDSS Decision Score Distribution", use_container_width=True)
            else:
                st.info("decision_score_distribution.png not found.")
                
        with col_di2:
            img = load_image(os.path.join(di_dir, "risk_score_distribution.png"))
            if img:
                st.image(img, caption="E-IDSS Risk Score Distribution", use_container_width=True)
            else:
                st.info("risk_score_distribution.png not found.")
                
        st.markdown("---")
        col_di3, col_di4 = st.columns(2)
        with col_di3:
            img = load_image(os.path.join(di_dir, "priority_matrix.png"))
            if img:
                st.image(img, caption="Operational Priority Levels", use_container_width=True)
            else:
                st.info("priority_matrix.png not found.")
                
        with col_di4:
            img = load_image(os.path.join(di_dir, "growth_vs_risk_scatter.png"))
            if img:
                st.image(img, caption="Opportunity Score vs. Risk Score Scatter Plot", use_container_width=True)
            else:
                st.info("growth_vs_risk_scatter.png not found.")
                
        st.markdown("---")
        img = load_image(os.path.join(di_dir, "store_performance_heatmap.png"))
        if img:
            st.image(img, caption="Store-Item Performance Heatmap (Items 1-15)", use_container_width=True)
            
        st.markdown("### Priority Rankings & Business Classifications")
        if os.path.exists(priority_path):
            p_df = load_csv(priority_path)
            if p_df is not None:
                # Multi-filters
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    priority_filter = st.selectbox("Priority Level", ["All"] + p_df["Priority_Level"].unique().tolist())
                with col_f2:
                    cat_filter = st.selectbox("Business Category", ["All"] + p_df["Business_Category"].unique().tolist())
                    
                filtered_p = p_df.copy()
                if priority_filter != "All":
                    filtered_p = filtered_p[filtered_p["Priority_Level"] == priority_filter]
                if cat_filter != "All":
                    filtered_p = filtered_p[filtered_p["Business_Category"] == cat_filter]
                    
                st.dataframe(filtered_p, use_container_width=True)
                
                csv = filtered_p.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Priority Rankings", data=csv, file_name="priority_rankings.csv", mime="text/csv")
        else:
            st.warning("Priority rankings CSV file not found.")

    # ----------------- PAGE 8: EXECUTIVE SUMMARY -----------------
    elif page_name == "8. Executive Summary":
        st.subheader("Executive Decision Support Analytics")
        
        report_text = load_txt(exe_report_path)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Average Decision Score</div>
                <div class="metric-val">44.86 / 100</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card warning">
                <div class="metric-lbl">Average Risk Score</div>
                <div class="metric-val">41.11 / 100</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Growth Opportunities</div>
                <div class="metric-val">15 Nodes</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card critical">
                <div class="metric-lbl">Critical Risks</div>
                <div class="metric-val">64 Nodes</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Top 5 Replenishment Opportunities")
        st.info("""
        1. **Store 6, Item 27** | Decision Score: **72.40/100** | Action: Increase inventory by 6.3%.
        2. **Store 7, Item 17** | Decision Score: **68.91/100** | Action: Increase inventory by 5.7%.
        3. **Store 4, Item 4**  | Decision Score: **68.80/100** | Action: Increase inventory by 8.8%.
        4. **Store 2, Item 45** | Decision Score: **67.68/100** | Action: Increase inventory by 6.2%.
        5. **Store 3, Item 24** | Decision Score: **67.26/100** | Action: Increase inventory by 8.7%.
        """)
        
        st.markdown("### Top 5 Severe Inventory Risks")
        st.warning("""
        1. **Store 2, Item 15** | Risk Score: **98.39/100** | Level: Critical Risk (High volatility/residuals).
        2. **Store 2, Item 28** | Risk Score: **97.92/100** | Level: Critical Risk (High volatility/residuals).
        3. **Store 2, Item 18** | Risk Score: **95.04/100** | Level: Critical Risk (Slow demand trend).
        4. **Store 8, Item 15** | Risk Score: **93.47/100** | Level: Critical Risk (High volatility/residuals).
        5. **Store 2, Item 13** | Risk Score: **92.75/100** | Level: Critical Risk (High volatility/residuals).
        """)
        
        st.markdown("### Executive Decision Report File (`executive_decision_report.txt`)")
        with st.expander("📄 Click to View Full Executive Report", expanded=True):
            st.markdown(f'<div class="report-block">{report_text}</div>', unsafe_allow_html=True)

    # ----------------- PAGE 9: FRAMEWORK COMPARISON -----------------
    elif page_name == "9. Framework Comparison":
        st.subheader("Framework Evaluation & Research Contribution (Phase 1)")
        
        comp_txt = load_txt(os.path.join(di_dir, "framework_comparison_report.txt"))
        ablation_txt = load_txt(os.path.join(di_dir, "ablation_study_report.txt"))
        contrib_txt = load_txt(os.path.join(di_dir, "research_contribution.txt"))
        
        tab_struct, tab_ablat, tab_novelty, tab_kpis = st.tabs([
            "📊 Structural Comparison", 
            "🧪 Ablation Study", 
            "🎓 Research & Novelty",
            "🎛️ E-IDSS KPI Dashboard"
        ])
        
        with tab_struct:
            st.markdown("### Structural Capability Comparison Table")
            comparison_matrix = {
                "Evaluation Dimension": [
                    "Prioritization Capability", 
                    "Explainability Quality", 
                    "Risk Awareness", 
                    "Business Actionability", 
                    "Executive Insight Quality"
                ],
                "Rule-Based Recommendation Engine": [
                    "Binary thresholds (No Ranking)",
                    "Static text blocks (Fixed context)",
                    "Not integrated (Flat buffer assumptions)",
                    "Generic inventory alerts (Increase/Reduce)",
                    "No cross-tab or prioritization rankings"
                ],
                "Decision Intelligence (E-IDSS)": [
                    "0-100 Decision Score MCDA Ranking",
                    "SHAP local feature explanations per node",
                    "Volatility & model uncertainty scoring (RMSE)",
                    "Categorized strategic business actions",
                    "Multi-criteria matrices, reports & heatmaps"
                ]
            }
            st.dataframe(pd.DataFrame(comparison_matrix), use_container_width=True)
            
            with st.expander("📄 View Comparison Report Text"):
                st.markdown(f'<div class="report-block">{comp_txt}</div>', unsafe_allow_html=True)
                
        with tab_ablat:
            st.markdown("### Ablation Study Matrix")
            ablation_matrix = {
                "System Stage Configuration": [
                    "A. Forecasting Only (Standard ML)", 
                    "B. Forecasting + SHAP (XAI ML)", 
                    "C. Forecasting + SHAP + Recommendation Engine (Rule DSS)", 
                    "D. Complete E-IDSS (Decision Intelligence)"
                ],
                "Explainability": ["None", "Global (Feature importance)", "Global (Rules)", "Local SHAP per node"],
                "Business Intelligence": ["Low", "Medium", "Medium", "High"],
                "Decision Support": ["Low", "Low", "Medium", "High"],
                "Operational Value": ["Baseline", "Incremental", "Moderate", "Maximal"]
            }
            st.dataframe(pd.DataFrame(ablation_matrix), use_container_width=True)
            
            with st.expander("📄 View Ablation Study Text"):
                st.markdown(f'<div class="report-block">{ablation_txt}</div>', unsafe_allow_html=True)
                
        with tab_novelty:
            st.markdown("### Research & Methodological Novelty")
            st.markdown("#### Multi-Criteria Decision Score Formulation")
            st.latex(r"Decision\_Score = 0.30 \times Growth\_Norm + 0.25 \times Trend\_Strength + 0.20 \times SHAP\_Impact + 0.15 \times Seasonality\_Norm + 0.10 \times Stability\_Norm")
            st.markdown("""
            * **Growth Rate (30%)**: Highlights acceleration compared to historical averages.
            * **Trend Strength (25%)**: Captures 7-day vs 30-day moving average demand velocities.
            * **SHAP Impact (20%)**: Factors in game-theoretic feature importance weights.
            * **Seasonality Strength (15%)**: Accounts for summer demand spikes.
            * **Demand Stability (10%)**: Incentivizes low coefficient of variation to reward stable demand curves.
            """)
            
            st.markdown("#### Risk Assessment Score Formulation")
            st.latex(r"Risk\_Score = 0.25 \times Volatility\_Norm + 0.25 \times Hist\_Variance\_Norm + 0.25 \times Trend\_Instability\_Norm + 0.25 \times Uncertainty\_Norm")
            st.markdown("""
            * **Volatility (25%)**: Forecast standard deviation showing projected swing levels.
            * **Historical Variance (25%)**: Standard deviation of historical sales records.
            * **Trend Instability (25%)**: Variance of predictions' residuals.
            * **Forecasting Uncertainty (25%)**: Root Mean Square Error (RMSE) of test partition.
            """)
            
            with st.expander("📄 View Research Contribution Document"):
                st.markdown(f'<div class="report-block">{contrib_txt}</div>', unsafe_allow_html=True)
                
        with tab_kpis:
            col_eng, col_dss = st.columns(2)
            with col_eng:
                st.markdown("#### Rule-Based Recommendation Engine")
                st.markdown("""
                <div class="metric-card critical">
                    <div class="metric-lbl">Explainability Score</div>
                    <div class="metric-val">Medium (Global)</div>
                </div>
                <div class="metric-card critical">
                    <div class="metric-lbl">Risk Awareness</div>
                    <div class="metric-val">Low (None)</div>
                </div>
                <div class="metric-card critical">
                    <div class="metric-lbl">Prioritization</div>
                    <div class="metric-val">Low (None)</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_dss:
                st.markdown("#### Complete Decision Intelligence (E-IDSS)")
                st.markdown("""
                <div class="metric-card success">
                    <div class="metric-lbl">Explainability Score</div>
                    <div class="metric-val">High (Local SHAP)</div>
                </div>
                <div class="metric-card success">
                    <div class="metric-lbl">Risk Awareness</div>
                    <div class="metric-val">High (dynamic RMSE)</div>
                </div>
                <div class="metric-card success">
                    <div class="metric-lbl">Prioritization</div>
                    <div class="metric-val">High (0-100 MCDA)</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            st.markdown("### Final Research Conclusion")
            st.success(
                "The E-IDSS extends traditional forecasting systems by integrating forecasting, "
                "explainability, risk assessment, prioritization, and executive decision analytics "
                "into a unified decision support framework."
            )
