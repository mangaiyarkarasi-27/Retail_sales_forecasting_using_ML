"""
dashboard/views/validation_view.py
---------------------------------
Research Validation & Technical Audit Repository for E-IDSS.
Provides complete reproducibility audit tables, Conference vs Journal extension matrix,
statistical hypothesis testing ledgers, and formal limitations declarations.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np


def render_validation_view(project_root: str, journal_root: str, helpers: dict):
    load_image = helpers["load_image"]

    st.markdown("""
    <div style="background: linear-gradient(135deg, #374151 0%, #1f2937 50%, #111827 100%); 
                padding: 1.5rem 1.8rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
                box-shadow: 0 10px 25px -5px rgba(17, 24, 39, 0.25);">
        <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 700;">
            🏛️ Research Validation & Scientific Audit
        </h2>
        <p style="color: #9ca3af; margin-top: 0.4rem; font-size: 0.95rem; margin-bottom: 0;">
            Peer-reviewed scientific audit trail, formal claim boundary verification, 
            and frozen reproducible experimental records across EXP-011 through EXP-020.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_mat, tab_audit, tab_stat, tab_limits = st.tabs([
        "1. Architectural Extension Matrix (Table 1)", 
        "2. Claims & Evidence Ledger", 
        "3. Statistical Hypothesis Tests", 
        "4. Scientific Limitations & Boundaries"
    ])

    # 1. Architectural Extension Matrix
    with tab_mat:
        st.markdown("### 📐 Table 1: Conference Baseline vs Journal Extension Matrix")
        st.markdown("""
        Systematic architectural comparison between the initial conference proof-of-concept and the comprehensive journal extension:
        """)

        matrix_data = [
            {
                "System Dimension": "Forecasting Paradigm",
                "Conference Baseline (Phase 1)": "Recursive single-step XGBoost (ŷ_t+1 fed back into feature vector)",
                "Journal Extension (Phase 2)": "Direct decoupled multi-horizon forecasting (7 independent models M₁...M₇)",
                "Scientific Benefit": "Eliminates autoregressive error cascading and runaway prediction divergence"
            },
            {
                "System Dimension": "Causal Feature Cutoff",
                "Conference Baseline (Phase 1)": "Look-ahead vulnerability (lag-1 used for multi-day horizons)",
                "Journal Extension (Phase 2)": "Strict causal pre-forecast feature boundary (τ ≥ 7 days for all lags and rolling windows)",
                "Scientific Benefit": "Guarantees decision-time validity with zero future information leakage"
            },
            {
                "System Dimension": "Uncertainty Quantification",
                "Conference Baseline (Phase 1)": "Static Gaussian heuristics and constant variance assumptions",
                "Journal Extension (Phase 2)": "Distribution-free forecast-conditional conformal prediction",
                "Scientific Benefit": "Finite-sample valid prediction intervals adaptive to dynamic demand volume"
            },
            {
                "System Dimension": "Distribution Shift Audit",
                "Conference Baseline (Phase 1)": "Assumed temporal stationarity and i.i.d. exchangeability",
                "Journal Extension (Phase 2)": "Quantitative empirical diagnostics of 3.55×–4.30× test-year residual variance expansion",
                "Scientific Benefit": "Uncovers the real-world breakdown of theoretical coverage guarantees under shift"
            },
            {
                "System Dimension": "Decision Integration",
                "Conference Baseline (Phase 1)": "Unvalidated prioritization ranking based purely on point estimates",
                "Journal Extension (Phase 2)": "Daily discrete-event continuous-review (s, S) replenishment simulation",
                "Scientific Benefit": "Empirically measures fill rate improvements and stockout prevention across 500 series"
            },
            {
                "System Dimension": "Economic Cost Frontier",
                "Conference Baseline (Phase 1)": "No cost evaluation or economic sensitivity modeling",
                "Journal Extension (Phase 2)": "Shortage penalty sensitivity sweep revealing economic crossover at p ≥ $2.00 / unit",
                "Scientific Benefit": "Provides actionable business guidance on when uncertainty buffers achieve net cost optimality"
            }
        ]
        st.dataframe(pd.DataFrame(matrix_data), hide_index=True, use_container_width=True)

    # 2. Claims & Evidence Ledger
    with tab_audit:
        st.markdown("### 📋 Formal Claims & Experimental Evidence Audit")
        st.markdown("""
        Every scientific assertion in the manuscript is tied directly to a frozen experimental artifact:
        """)
        claims_csv = os.path.join(journal_root, "results", "final_audit", "limitations_and_claims.csv")
        if os.path.exists(claims_csv):
            claims_df = pd.read_csv(claims_csv)
            st.dataframe(
                claims_df[["Claim_Topic", "Formal_Claim", "Experimental_Evidence", "Status"]],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Claims audit ledger loading...")

    # 3. Statistical Hypothesis Tests
    with tab_stat:
        st.markdown("### 🔬 Rigorous Non-Parametric Hypothesis Testing")
        st.markdown("""
        To establish that improvements in service level, shortage reduction, and calibration efficiency are 
        statistically significant rather than random sampling fluctuations, non-parametric paired Wilcoxon 
        signed-rank tests were performed across all 500 store-item series:
        """)

        wilc_csv = os.path.join(journal_root, "results", "inventory", "exp_019", "wilcoxon_tests.csv")
        if os.path.exists(wilc_csv):
            w_df = pd.read_csv(wilc_csv)
            st.dataframe(w_df, hide_index=True, use_container_width=True)
            st.markdown("""
            > [!TIP]
            > **Statistical Verdict:**  
            > In all lead-time evaluations ($L \\in \\{1, 3, 5, 7\\}$), test statistics yield $W = 0.0$ and $p < 10^{-75}$, 
            > decisively rejecting the null hypothesis of equal fill rates between deterministic and conformal policies.
            """)
        else:
            st.info("Statistical tests table loading...")

    # 4. Scientific Limitations & Boundaries
    with tab_limits:
        st.markdown("### ⚠️ Declared Scope Boundaries & Limitations")
        st.markdown("""
        In accordance with transparent scientific reporting, the following operational boundaries are explicitly disclosed:

        1. **Temporal Non-Exchangeability:**  
           While conformal prediction offers mathematical finite-sample coverage guarantees under exchangeability, 
           forward time-series distribution shift in 2017 caused residual variance to expand by **3.55× to 4.30×**, 
           reducing empirical test coverage to ~58.7% without dynamic online recalibration.

        2. **Economic Shortage Penalty Crossover:**  
           Conformal replenishment is **not universally cost-minimizing**. When stockout penalties are negligible 
           ($p \\le \\$1.50$), the extra inventory holding costs exceed stockout savings. Conformal policies become 
           decisively cost-saving only when stockouts carry substantial business consequences ($p \\ge \\$2.00$).

        3. **Inventory Model Assumptions:**  
           The discrete-event simulation evaluates non-perishable merchandise with deterministic lead times and 
           lost sales dynamics. Future extensions should incorporate supplier lead-time stochasticity and product perishability.
        """)
