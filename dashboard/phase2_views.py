"""
dashboard/phase2_views.py
-------------------------
Contains all 9 dedicated pages for Phase 2 (Scopus Journal Extension).
Connects directly to the frozen experimental artifacts (EXP-011 through EXP-020),
models, diagnostics, conformal intervals, inventory simulation, and scientific audits.
"""

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np


def render_phase2_page(page_name: str, project_root: str, journal_root: str, helpers: dict):
    load_csv = helpers["load_csv"]
    load_txt = helpers["load_txt"]
    load_image = helpers["load_image"]

    res_dir = os.path.join(journal_root, "results")
    audit_dir = os.path.join(journal_root, "audit")
    final_audit_dir = os.path.join(res_dir, "final_audit")
    unc_dir = os.path.join(res_dir, "uncertainty")
    inv_dir = os.path.join(res_dir, "inventory", "exp_019")
    ms_dir = os.path.join(res_dir, "model_selection")
    fc_dir = os.path.join(res_dir, "forecasting")
    models_dir = os.path.join(res_dir, "forecasting_models")

    # ================= PAGE 1: JOURNAL OVERVIEW & ARCHITECTURE =================
    if page_name == "1. Journal Overview & System Architecture":
        st.subheader("Phase 2 Journal Extension: System Architecture & Research Overview")
        
        st.markdown("""
        **Manuscript Title:**  
        *Bridging Decision-Time-Safe Multi-Horizon Demand Forecasting and Uncertainty-Aware Retail Replenishment: An Empirical Conformal Prediction and Discrete-Event Simulation Study*
        
        This journal extension expands the preliminary conference study (IEEE E-IDSS) into an end-to-end,
        leakage-controlled, distribution-free decision-support framework evaluated across 500 store-item daily time series (2013–2017).
        """)

        # KPI Header Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Direct Forecasters</div>
                <div class="metric-val">7 Global XGBoost</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>Horizons h=1..7, lag τ ≥ 7 days</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card warning">
                <div class="metric-lbl">Residual Scale Shift</div>
                <div class="metric-val">3.55x – 4.30x</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>2017 std expansion (p < 10⁻⁴²)</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Portfolio Fill Rate Gain</div>
                <div class="metric-val">+7.29 to +10.64 pp</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>Paired Wilcoxon p < 10⁻⁷⁵</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Lost Sales Prevented</div>
                <div class="metric-val">1,139,914 Units</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>-20.03% shortage at L=1</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Integrated 7-Stage Pipeline Architecture")
        col_flow, col_desc = st.columns([1, 1])
        with col_flow:
            st.markdown(r"""
            1. **Longitudinal Dataset Partitioning:** 2013–2015 Train (547k rows), 2016 Validation (183k rows), 2017 Untouched OOS Test (182.5k rows).
            2. **Decision-Time-Safe Feature Protocol:** Strictly enforces minimum 7-day lag isolation ($\tau \ge 7$ days) to eliminate look-ahead leakage.
            3. **Decoupled Direct Multi-Horizon Estimators:** 7 independent global XGBoost models ($M_1 \dots M_7$) eliminating recursive error accumulation.
            4. **Rolling-Origin OOS Residual Calibration:** $N = 608,165$ out-of-sample calibration residuals generated without forward test access.
            5. **Distribution Shift & Scale Diagnostics:** Two-sample location-scale decomposition diagnosing test-year variance explosion.
            6. **Rolling Forecast-Conditional Conformal Calibration:** Non-parametric prediction intervals scaled to forecasted demand magnitude.
            7. **Discrete-Event Replenishment Simulation:** 365 daily discrete simulation events under lost-sales dynamics across $L \in \{1, 3, 5, 7\}$.
            """)
        with col_desc:
            arch_img = load_image(os.path.join(project_root, "results", "plots", "timeseries_split_visualization.png"))
            if arch_img:
                st.image(arch_img, caption="Dataset Partitioning & Chronological Boundaries", use_container_width=True)

        st.markdown("### Core Research Questions Addressed")
        st.info("""
        - **RQ1 (Forecasting Architecture):** Does a decoupled direct multi-horizon architecture maintain competitive accuracy while guaranteeing decision-time safety across lead times?  
          *Finding: Yes, achieves validation MAE 6.0388 and RMSE 7.8638 with 10/10 automated temporal leakage checks passed.*
        - **RQ2 (Uncertainty Calibration under Shift):** How do conformal calibration methods perform when forward demand violates exchangeability?  
          *Finding: Raw conformal coverage collapses from 90% to ~29% due to 3.55x–4.30x residual scale expansion; forecast-conditional calibration restores top-decile high-demand coverage by +8.80 pp and minimizes Winkler score.*
        - **RQ3 (Inventory Service Translation):** How effectively do conformal uncertainty margins translate into replenishment service improvements?  
          *Finding: Conformal safety stock improves portfolio fill rate by +7.29 to +10.64 pp and prevents up to 1.14M units of lost sales (p < 10⁻⁷⁵).*
        - **RQ4 (Economic Cost-Service Trade-off):** What is the total operating cost impact across lead times and shortage penalty regimes?  
          *Finding: Conformal lowers cost at L=1 (-10.82%); for multi-day lead times (L ≥ 3), holding stock overhead is cost-preferred whenever unit shortage penalty p ≥ $2.00.*
        """)

    # ================= PAGE 2: DIRECT MULTI-HORIZON FORECASTING =================
    elif page_name == "2. Direct Multi-Horizon Forecasting (EXP-011)":
        st.subheader("EXP-011: Decoupled Direct Multi-Horizon Forecasting Architecture")
        
        st.markdown(r"""
        To avoid recursive feedback error propagation and eliminate intermediate look-ahead leakage within supplier delivery lead times,
        we deploy **seven decoupled global direct XGBoost regressors** ($M_1, M_2, \dots, M_7$). Each model independently forecasts
        target demand for horizon $h \in \{1, \dots, 7\}$ using exclusively causal features realized at or before $t-7$.
        """)

        # Load validation summary and per-horizon table
        summary_path = os.path.join(fc_dir, "direct_multihorizon_summary.csv")
        val_path = os.path.join(fc_dir, "direct_multihorizon_validation.csv")
        
        sum_df = load_csv(summary_path)
        val_df = load_csv(val_path)

        col1, col2, col3 = st.columns(3)
        if sum_df is not None:
            with col1:
                st.markdown(f"""
                <div class="metric-card success">
                    <div class="metric-lbl">Aggregate Validation MAE</div>
                    <div class="metric-val">{float(sum_df['aggregate_validation_mae'].iloc[0]):.4f} Units</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card success">
                    <div class="metric-lbl">Aggregate Validation RMSE</div>
                    <div class="metric-val">{float(sum_df['aggregate_validation_rmse'].iloc[0]):.4f} Units</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card info">
                    <div class="metric-lbl">Decoupled Models</div>
                    <div class="metric-val">{int(sum_df['number_of_trained_models'].iloc[0])} Regressors</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### Table 3: Multi-Horizon Validation Results (2016 Holdout Split)")
        if val_df is not None:
            st.dataframe(val_df, use_container_width=True, hide_index=True)
            csv = val_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download EXP-011 Validation Metrics CSV", data=csv, file_name="direct_multihorizon_validation.csv", mime="text/csv")

        st.markdown("### Interactive Model Artifact Inspector")
        st.markdown("Inspect the serialized XGBoost models saved in `journal_extension/results/forecasting_models/`:")
        
        selected_h = st.selectbox("Select Forecast Horizon (M_h)", [1, 2, 3, 4, 5, 6, 7], format_func=lambda x: f"Model M_{x} (Horizon h={x} days)")
        model_file = os.path.join(models_dir, f"xgb_direct_h{selected_h}.joblib")
        
        if os.path.exists(model_file):
            file_size_kb = os.path.getsize(model_file) / 1024
            st.success(f"✓ Model `xgb_direct_h{selected_h}.joblib` loaded successfully ({file_size_kb:.1f} KB).")
            try:
                model_obj = joblib.load(model_file)
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("#### Model Hyperparameters")
                    params = {
                        "n_estimators": model_obj.n_estimators,
                        "max_depth": model_obj.max_depth,
                        "learning_rate": model_obj.learning_rate,
                        "subsample": model_obj.subsample,
                        "colsample_bytree": model_obj.colsample_bytree,
                        "random_state": model_obj.random_state
                    }
                    st.json(params)
                with col_m2:
                    st.markdown("#### Input Feature Specification (16 Features)")
                    st.info("""
                    - **Autoregressive Lags:** sales_lag_7, sales_lag_14, sales_lag_30 (all τ ≥ 7)
                    - **Rolling Averages:** rolling_mean_7, rolling_mean_30 (lagged 7 days)
                    - **Calendar Encodings:** year, month, day, weekday, weekofyear, quarter, is_weekend, month_start, month_end
                    - **Identifiers:** store, item
                    """)
            except Exception as e:
                st.error(f"Error inspecting model: {e}")
        else:
            st.warning(f"Model file `{model_file}` not found.")

    # ================= PAGE 3: DISTRIBUTION SHIFT DIAGNOSTICS =================
    elif page_name == "3. Distribution Shift & Scale Diagnostics (EXP-014)":
        st.subheader("EXP-014: Distribution Shift and Residual Scale Expansion Diagnostics")
        
        st.markdown("""
        When classical conformal prediction was deployed on the untouched 2017 test holdout, empirical coverage collapsed
        from the nominal 90% target to between 15.5% and 34.7%. EXP-014 diagnoses the root cause through a formal two-sample
        location-scale decomposition between historical calibration residuals (2013–2016) and 2017 test residuals.
        """)

        # Load scale expansion data
        scale_csv = os.path.join(unc_dir, "diagnostics_2017", "calibration_vs_test_residuals.csv")
        bias_csv = os.path.join(unc_dir, "diagnostics_2017", "forecast_bias_by_quarter.csv")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card critical">
                <div class="metric-lbl">L=1 Std Dev Expansion</div>
                <div class="metric-val">3.55x</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>12.35 → 43.89 units</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card critical">
                <div class="metric-lbl">L=7 Std Dev Expansion</div>
                <div class="metric-val">4.30x</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>70.19 → 301.60 units</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card warning">
                <div class="metric-lbl">2-Sample KS Statistic</div>
                <div class="metric-val">D = 0.170 – 0.197</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>p < 10⁻⁴² (Non-exchangeable)</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Seasonal Signed Error</div>
                <div class="metric-val">-18.84 to +20.95</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>Q1 under-est → Q3 over-est</p>
            </div>
            """, unsafe_allow_html=True)

        tab_scale, tab_drift, tab_viol = st.tabs([
            "📈 Residual Scale Expansion (Table 5 & Fig 4)",
            "❄️ Seasonal Bias Swings",
            "⚠️ Violation Decomposition & Quantiles"
        ])

        with tab_scale:
            st.markdown("#### Table 5: Location-Scale Decomposition Across Lead Times")
            df_scale = load_csv(scale_csv)
            if df_scale is not None:
                st.dataframe(df_scale, use_container_width=True)
            
            img_scale = load_image(os.path.join(unc_dir, "diagnostics_2017", "calibration_vs_test_residual_scale.png"))
            if img_scale:
                st.image(img_scale, caption="Figure 4: Empirical Distribution Shift & Residual Scale Expansion (EXP-014)", use_container_width=True)
            else:
                st.info("calibration_vs_test_residual_scale.png not found.")

        with tab_drift:
            st.markdown("#### Quarterly Seasonal Mean Signed Error Swings")
            df_bias = load_csv(bias_csv)
            if df_bias is not None:
                st.dataframe(df_bias, use_container_width=True)
                
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                img_drift = load_image(os.path.join(unc_dir, "diagnostics_2017", "residual_drift_by_year.png"))
                if img_drift:
                    st.image(img_drift, caption="Residual Drift Across Years (2013-2017)", use_container_width=True)
            with col_d2:
                img_bias_h = load_image(os.path.join(unc_dir, "diagnostics_2017", "forecast_bias_by_horizon.png"))
                if img_bias_h:
                    st.image(img_bias_h, caption="Forecast Bias Breakdown Across Lead Times", use_container_width=True)

        with tab_viol:
            st.markdown("#### Interval Violation Symmetry & Quantile Alignment")
            st.info("""
            **Diagnostic Finding:** Lower violations (35.3%) and upper violations (35.3%) were exactly balanced at the nominal 90% confidence target.
            This proves that the coverage collapse was driven by **unmodeled forward scale explosion**, rather than a simple one-sided mean forecasting bias!
            """)
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                img_viol = load_image(os.path.join(unc_dir, "diagnostics_2017", "lower_vs_upper_violations.png"))
                if img_viol:
                    st.image(img_viol, caption="Symmetric Upper vs Lower Violation Breakdown", use_container_width=True)
            with col_v2:
                img_qq = load_image(os.path.join(unc_dir, "diagnostics_2017", "quantile_vs_actual_error.png"))
                if img_qq:
                    st.image(img_qq, caption="Quantile vs Actual Calibration Residuals", use_container_width=True)

    # ================= PAGE 4: CONFORMAL UNCERTAINTY CALIBRATION =================
    elif page_name == "4. Conformal Uncertainty & Calibration (EXP-015–017)":
        st.subheader("EXP-015 to EXP-017: Conformal Prediction Calibration & Method Comparison")
        
        st.markdown(r"""
        To mitigate distribution shift, we benchmark four conformal prediction architectures on the 2017 test holdout:
        1. **Raw Rolling Conformal:** Standard non-parametric rolling window ($W = 100$).
        2. **Trailing Scale-Normalized:** Normalizes residuals by trailing historical standard deviation ($\hat{\sigma}_{t,L}$).
        3. **Hard-Regime Forecast-Conditional:** Partitions calibration residuals into demand terciles (Low/Med/High).
        4. **Locally Weighted Forecast-Conditional:** Weights calibration residuals by forecast-scale proximity ($s = |e| / (\hat{D} + \epsilon)$).
        """)

        # Load uncertainty results
        unc_results_path = os.path.join(final_audit_dir, "uncertainty_results.csv")
        df_unc = load_csv(unc_results_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Top-Decile Coverage Gain</div>
                <div class="metric-val">+8.80 pp</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>High-demand volume protection (p < 0.001)</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Winkler Score Improvement</div>
                <div class="metric-val">409.66 vs 432.15</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>Paired Wilcoxon p = 1.44 × 10⁻²⁸ (L=1)</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Evaluated Origins</div>
                <div class="metric-val">364 Daily Steps</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>Full 2017 Out-of-Sample Holdout</p>
            </div>
            """, unsafe_allow_html=True)

        if df_unc is not None:
            st.markdown("### Interactive Uncertainty Calibration Explorer")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                sel_lead = st.selectbox("Select Delivery Lead Time (L)", [1, 3, 5, 7], format_func=lambda x: f"L = {x} Days")
            with col_f2:
                sel_target = st.selectbox("Select Nominal Confidence Level", [0.80, 0.90, 0.95], format_func=lambda x: f"{int(x*100)}% Nominal Confidence")
                
            filtered_unc = df_unc[(df_unc["lead_time"] == sel_lead) & (np.isclose(df_unc["nominal_coverage"], sel_target))]
            st.dataframe(filtered_unc[["method", "nominal_coverage", "empirical_coverage", "mean_interval_width", "interval_score", "lower_violation_rate", "upper_violation_rate"]], use_container_width=True, hide_index=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            img_cov = load_image(os.path.join(unc_dir, "final_2017", "nominal_vs_empirical_coverage.png"))
            if img_cov:
                st.image(img_cov, caption="Figure 5: Empirical Coverage Breakdown on 2017 Test Holdout (EXP-015/016)", use_container_width=True)
            else:
                img_cov_alt = load_image(os.path.join(unc_dir, "conditional_016", "raw_scale_conditional_coverage.png"))
                if img_cov_alt:
                    st.image(img_cov_alt, caption="Empirical Coverage Across Methods", use_container_width=True)
        with col_p2:
            img_winkler = load_image(os.path.join(unc_dir, "conditional_016", "raw_scale_conditional_winkler.png"))
            if img_winkler:
                st.image(img_winkler, caption="Figure 6: Mean Winkler Interval Score Comparison (EXP-016/017)", use_container_width=True)

    # ================= PAGE 5: CALIBRATION SENSITIVITY & K SWEEP =================
    elif page_name == "5. Calibration Sensitivity & K Sweep (EXP-018)":
        st.subheader("EXP-018: Robustness Analysis & Calibration Neighborhood Size (K) Sweep")
        
        st.markdown(r"""
        To establish parameter stability for locally weighted forecast-conditional calibration, EXP-018 sweeps
        the neighborhood size parameter $K \in [10, 100]$ over the 2016 validation partition.
        """)

        k_sens_path = os.path.join(unc_dir, "local_conditional_018", "k_sensitivity_2016.csv")
        df_k = load_csv(k_sens_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Stability Plateau</div>
                <div class="metric-val">K ∈ [30, 100]</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>< 0.67% Winkler Score Variation</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Frozen Hyperparameter</div>
                <div class="metric-val">K* = 100</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>Full rolling window calibration N</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Tie-Breaking Criterion</div>
                <div class="metric-val">Pre-Registered Rule</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>Avoids arbitrary subsampling</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Table 6: Calibration Neighborhood Sensitivity Sweep (2016 Validation)")
        if df_k is not None:
            st.dataframe(df_k, use_container_width=True, hide_index=True)

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            img_kw = load_image(os.path.join(unc_dir, "local_conditional_018", "k_vs_winkler.png"))
            if img_kw:
                st.image(img_kw, caption="Figure 7: Validation Winkler Loss vs. Neighborhood Size K (EXP-018)", use_container_width=True)
        with col_k2:
            img_kc = load_image(os.path.join(unc_dir, "local_conditional_018", "k_vs_coverage_error.png"))
            if img_kc:
                st.image(img_kc, caption="Coverage Error vs. Neighborhood Size K", use_container_width=True)

    # ================= PAGE 6: INVENTORY SIMULATION =================
    elif page_name == "6. Discrete-Event Inventory Simulation (EXP-019)":
        st.subheader("EXP-019: Discrete-Event Inventory Replenishment Simulation")
        
        st.markdown(r"""
        EXP-019 translates the calibrated uncertainty bounds into dynamic safety stock buffers ($SS_{t,L} = Upper_{t,L} - \hat{D}_{t,L}$)
        within a continuous-review replenishment policy ($ROP = \hat{D} + SS$), evaluated under daily discrete events with **lost sales**
        across 500 store-item combinations over the full 2017 test holdout (365 days).
        """)

        # Load lead time service data and bootstrap CIs
        lead_csv = os.path.join(inv_dir, "lead_time_comparison.csv")
        boot_csv = os.path.join(inv_dir, "bootstrap_confidence_intervals.csv")
        df_lead = load_csv(lead_csv)
        df_boot = load_csv(boot_csv)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">L=1 Fill Rate Gain</div>
                <div class="metric-val">+10.64 pp</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>46.87% → 57.51% (Portfolio)</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">L=1 Bootstrap 95% CI</div>
                <div class="metric-val">[+10.19, +10.55]</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>1,000 paired series resamples</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Lost Sales Cut (L=1)</div>
                <div class="metric-val">-20.03%</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>1,139,914 unmet units prevented</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Stockout Rate Reduction</div>
                <div class="metric-val">-16.14 pp</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>67.24% → 51.10% (L=1)</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Table 7: Portfolio Replenishment Service Performance (EXP-019)")
        if df_lead is not None:
            st.dataframe(df_lead, use_container_width=True, hide_index=True)
            
        st.markdown("### 1,000-Resample Paired Bootstrap Confidence Intervals (95% CI)")
        if df_boot is not None:
            st.dataframe(df_boot, use_container_width=True, hide_index=True)

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            img_fr = load_image(os.path.join(inv_dir, "fill_rate_by_lead_time.png"))
            if img_fr:
                st.image(img_fr, caption="Figure 8: Fill Rate Performance Across Lead Times (EXP-019)", use_container_width=True)
        with col_i2:
            img_short = load_image(os.path.join(inv_dir, "shortage_units_by_lead_time.png"))
            if img_short:
                st.image(img_short, caption="Annual Lost Sales Shortage Units Mitigated (EXP-019)", use_container_width=True)

        st.markdown("### Interactive Per-Series Simulation Inspector")
        per_series_csv = os.path.join(inv_dir, "per_series_results.csv")
        df_ps = load_csv(per_series_csv)
        if df_ps is not None:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                store_sel = st.selectbox("Select Store ID", sorted(df_ps["store"].unique()))
            with col_s2:
                item_sel = st.selectbox("Select Item ID", sorted(df_ps[df_ps["store"] == store_sel]["item"].unique()))
                
            series_data = df_ps[(df_ps["store"] == store_sel) & (df_ps["item"] == item_sel)]
            st.dataframe(series_data, use_container_width=True, hide_index=True)

    # ================= PAGE 7: ECONOMIC COST & PENALTY FRONTIER =================
    elif page_name == "7. Economic Cost & Penalty Frontier (EXP-019)":
        st.subheader("EXP-019: Portfolio Operating Cost Breakdown & Shortage Penalty Sensitivity")
        
        st.markdown(r"""
        To assess real-world business viability, we simulate total annual inventory operating costs:
        $$\text{Total Operating Cost} = \text{Ordering Cost} + \text{Holding Cost} + \text{Shortage Cost}$$
        across four delivery lead times ($L \in \{1, 3, 5, 7\}$ days) and four shortage penalty regimes ($p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$).
        """)

        cost_csv = os.path.join(inv_dir, "cost_breakdown.csv")
        sens_csv = os.path.join(inv_dir, "shortage_penalty_sensitivity.csv")
        df_cost = load_csv(cost_csv)
        df_sens = load_csv(sens_csv)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">L=1 Baseline Savings (p=$1)</div>
                <div class="metric-val">-$790,834</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>-10.82% Total Cost Reduction</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card warning">
                <div class="metric-lbl">L=3 Cost Trade-off (p=$1)</div>
                <div class="metric-val">+$413,090</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>+$1.39M holding vs -$928k shortage</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Economic Crossover Point</div>
                <div class="metric-val">p ≥ $2.00 / Unit</div>
                <p style='color:#64748b; font-size:0.8rem; margin-top:0.2rem;'>Conformal cost-dominant across multi-day horizons</p>
            </div>
            """, unsafe_allow_html=True)

        tab_base, tab_sens, tab_tests = st.tabs([
            "💵 Baseline Cost Breakdown (p=$1.00)",
            "📊 Penalty Sensitivity Frontier (p=$0.60 to $4.00)",
            "📐 Inferential Statistical Hypothesis Tests"
        ])

        with tab_base:
            st.markdown("#### Table 8: Simulated Annual Operating Cost Breakdown (Baseline p = $1.00)")
            if df_cost is not None:
                st.dataframe(df_cost, use_container_width=True, hide_index=True)
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                img_cb = load_image(os.path.join(inv_dir, "cost_breakdown_bars.png"))
                if img_cb:
                    st.image(img_cb, caption="Cost Component Breakdown Across Lead Times", use_container_width=True)
            with col_b2:
                img_tc = load_image(os.path.join(inv_dir, "total_cost_by_lead_time.png"))
                if img_tc:
                    st.image(img_tc, caption="Total Simulated Cost Across Policies", use_container_width=True)

        with tab_sens:
            st.markdown("#### Table 9: Shortage Penalty Sensitivity Matrix (Net Cost Delta ΔTC)")
            st.info("""
            **Managerial Takeaway:** Whenever unit shortage costs exceed $2.00 (representing high-margin items or steep stockout penalties),
            conformal replenishment achieves **lower total operating costs across multi-day horizons**, generating net annual savings of
            **-$515,118** at L=3 (p=$2.00) and **-$2,371,534** at L=3 (p=$4.00).
            """)
            if df_sens is not None:
                st.dataframe(df_sens, use_container_width=True, hide_index=True)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                img_front = load_image(os.path.join(inv_dir, "shortage_penalty_sensitivity.png"))
                if img_front:
                    st.image(img_front, caption="Figure 9: Operating Cost Preference Matrix Across Lead Times & Penalties", use_container_width=True)
            with col_s2:
                img_csf = load_image(os.path.join(inv_dir, "cost_service_frontier.png"))
                if img_csf:
                    st.image(img_csf, caption="Cost-Service Efficient Frontier", use_container_width=True)

        with tab_tests:
            st.markdown("#### Table 10: Inferential Statistical Hypothesis Tests")
            wilcoxon_csv = os.path.join(inv_dir, "wilcoxon_tests.csv")
            df_wil = load_csv(wilcoxon_csv)
            if df_wil is not None:
                st.dataframe(df_wil, use_container_width=True, hide_index=True)
            else:
                st.info("wilcoxon_tests.csv not found.")

    # ================= PAGE 8: EXPLAINABILITY & MODEL ROUTING =================
    elif page_name == "8. Explainability & Model Routing":
        st.subheader("Model Explainability, Routing & Demand Profiling")
        
        st.markdown("""
        To understand how model selection decisions are made across the 500 store-item series, we evaluate the
        **Adaptive Model Router** and analyze the underlying time-series demand profiles using the Syntetos-Boylan-Croston (SBC) taxonomy.
        """)

        fi_csv = os.path.join(journal_root, "results", "explainability", "router_feature_importance.csv")
        profile_csv = os.path.join(ms_dir, "demand_profiles.csv")
        df_fi = load_csv(fi_csv)
        df_prof = load_csv(profile_csv)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Profiled Series</div>
                <div class="metric-val">500 Combinations</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Dominant Router Driver</div>
                <div class="metric-val">Mean Sales (47.6%)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card info">
                <div class="metric-lbl">Volatility Feature</div>
                <div class="metric-val">CV of Sales (42.2%)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card success">
                <div class="metric-lbl">Routing Accuracy</div>
                <div class="metric-val">70.4% (359 XGB / 141 RF)</div>
            </div>
            """, unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### Router Feature Importance")
            if df_fi is not None:
                st.dataframe(df_fi, use_container_width=True, hide_index=True)
            img_rfi = load_image(os.path.join(ms_dir, "router_feature_importance.png"))
            if img_rfi:
                st.image(img_rfi, caption="Feature Importance for Adaptive Model Router", use_container_width=True)
        with col_r2:
            st.markdown("#### Demand Characteristics & Volatility")
            img_cv = load_image(os.path.join(ms_dir, "cv_distribution.png"))
            if img_cv:
                st.image(img_cv, caption="Coefficient of Variation Distribution across 500 Series", use_container_width=True)

        st.markdown("### Interactive Demand Profile Browser (500 Series)")
        if df_prof is not None:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                store_filter = st.selectbox("Filter Store", ["All"] + sorted(df_prof["store"].unique().tolist()))
            with col_f2:
                cat_filter = st.selectbox("Filter SBC Category", ["All"] + sorted(df_prof["sbc_category"].unique().tolist()))
                
            filtered_prof = df_prof.copy()
            if store_filter != "All":
                filtered_prof = filtered_prof[filtered_prof["store"] == int(store_filter)]
            if cat_filter != "All":
                filtered_prof = filtered_prof[filtered_prof["sbc_category"] == cat_filter]
                
            st.dataframe(filtered_prof, use_container_width=True, hide_index=True)

    # ================= PAGE 9: SCIENTIFIC EVIDENCE & AUDIT =================
    elif page_name == "9. Scientific Evidence & Methodology Audit (EXP-020)":
        st.subheader("EXP-020: Scientific Evidence Table & Methodology Freeze Audit")
        
        st.markdown("""
        All experimental outputs across EXP-011 through EXP-020 have been formally audited and frozen.
        This page exposes the complete, immutable evidence chain, demonstrating zero data leakage, exact numerical consistency,
        and strict boundary alignment.
        """)

        tab_ev, tab_claims, tab_freeze, tab_audit = st.tabs([
            "📋 Final Evidence Table (EXP-011 to EXP-019)",
            "🛡️ Scientific Claim Boundaries",
            "❄️ Methodology Freeze & Verdict",
            "🔍 Visual & Numerical Audit"
        ])

        with tab_ev:
            st.markdown("#### Authoritative Experimental Evidence Table")
            ev_csv = os.path.join(final_audit_dir, "FINAL_EVIDENCE_TABLE.csv")
            df_ev = load_csv(ev_csv)
            if df_ev is not None:
                st.dataframe(df_ev, use_container_width=True, hide_index=True)
                csv = df_ev.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Final Evidence Table CSV", data=csv, file_name="FINAL_EVIDENCE_TABLE.csv", mime="text/csv")

        with tab_claims:
            st.markdown("#### Scientific Claims, Empirical Status, and Allowed Wording")
            claims_csv = os.path.join(final_audit_dir, "limitations_and_claims.csv")
            df_claims = load_csv(claims_csv)
            if df_claims is not None:
                st.dataframe(df_claims, use_container_width=True, hide_index=True)

        with tab_freeze:
            st.markdown("#### Formal Methodology Freeze Declaration")
            freeze_txt = load_txt(os.path.join(final_audit_dir, "METHODOLOGY_FREEZE.md"))
            verdict_txt = load_txt(os.path.join(final_audit_dir, "FINAL_SCIENTIFIC_VERDICT.md"))
            
            with st.expander("📄 View METHODOLOGY_FREEZE.md", expanded=True):
                st.markdown(f'<div class="report-block">{freeze_txt}</div>', unsafe_allow_html=True)
            with st.expander("📄 View FINAL_SCIENTIFIC_VERDICT.md", expanded=True):
                st.markdown(f'<div class="report-block">{verdict_txt}</div>', unsafe_allow_html=True)

        with tab_audit:
            st.markdown("#### Final Manuscript Visual & Numerical Audit")
            audit_report = load_txt(os.path.join(audit_dir, "conference_journal_extension_audit.md"))
            with st.expander("📄 View Conference-to-Journal Audit Report", expanded=True):
                st.markdown(f'<div class="report-block">{audit_report}</div>', unsafe_allow_html=True)
