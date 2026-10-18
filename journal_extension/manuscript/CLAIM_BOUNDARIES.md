# CLAIM BOUNDARIES & WRITING GOVERNANCE

This document defines the strict **epistemological boundaries and writing governance rules** for drafting the journal manuscript. Every claim made in the paper must strictly adhere to this three-tier classification.

---

## 1. SAFE TO CLAIM (Directly Supported by Empirical Evidence)

These claims are mathematically, statistically, and experimentally verified across EXP-011 through EXP-020 and may be stated without hedging:

| Claim Area | Precise Permitted Phrasing | Direct Experimental Evidence |
| :--- | :--- | :--- |
| **Leakage-Free Multi-Horizon Forecasting** | *"Direct multi-horizon XGBoost models eliminate recursive error propagation and satisfy strict decision-time causality across all evaluated forecast horizons ($h=1,\dots,7$) with lag $\ge 7$ days."* | **EXP-011**: 10/10 temporal leakage tests passed; stable validation MAE ($6.0388$). |
| **Fill Rate & Service Improvements** | *"Integrating forecast-conditional conformal uncertainty into replenishment reorder points achieves statistically significant portfolio-wide fill rate improvements (+7.28 to +10.64 pp) across all evaluated lead times."* | **EXP-019**: Paired Wilcoxon tests on 500 series yield $p < 10^{-75}$; 95% bootstrap CIs strictly exclude 0. |
| **Lost Sales Shortage Reduction** | *"Conformal uncertainty buffers reduce unmet customer demand by 20.0% to 29.6%, preventing up to 1.14 million units of lost sales annually across the 500 store-item portfolio."* | **EXP-019**: Shortage dropped from 5.69M to 4.55M units at $L=1$, and 2.61M to 1.84M units at $L=7$. |
| **Unconditional Win-Win at $L=1$** | *"For single-day lead-time replenishment ($L=1$), uncertainty-aware replenishment improves fill rate (+10.6 pp) while simultaneously lowering total operating costs by 10.8% ($-\$790,834$, $p = 6.65 \times 10^{-55}$)."* | **EXP-019**: Cost reduction driven by order consolidation ($-\$53.5\text{k}$) and stockout elimination ($-\$1.14\text{M}$). |
| **Monotonic Risk Control** | *"The nominal conformal coverage parameter $\beta \in \{80\%, 90\%, 95\%\}$ acts as a strictly monotonic control parameter for adjusting customer service levels and safety stock buffers."* | **EXP-019**: Fill rate, ending inventory, and stockout rate exhibit strict monotonicity across all 500 series. |
| **Calibration Window Robustness** | *"Forecast-conditional calibration performance exhibits a broad stability plateau across neighborhood sizes $K \in [30, 100]$ on validation data."* | **EXP-018**: Validation Winkler loss varied by $< 0.7\%$ across $K \in [30, 100]$, validating the selection of $W=100$. |
| **Interval Efficiency Gains** | *"Rolling forecast-conditional scale normalization significantly reduces mean Winkler loss ($p < 0.02$) and improves high-demand coverage by +8.8 pp compared to raw conformal prediction."* | **EXP-016/017/018**: High-regime coverage boosted from 28.0% to 36.8%; lower violations cut by 8.4 pp. |

---

## 2. CLAIM WITH QUALIFICATION (Contextually / Assumption-Bound Claims)

These claims are true within our experimental simulation setup but must always be accompanied by explicit qualification:

| Claim Area | Required Qualified Phrasing | Mandatory Boundary Condition |
| :--- | :--- | :--- |
| **Total Cost Trade-Offs at $L \ge 3$** | *"For multi-day lead times ($L \in \{3, 5, 7\}$), conformal replenishment creates an actionable cost-service efficient frontier, achieving total cost dominance when shortage penalties exceed $\$2.00/\text{unit}$."* | Must state that at $p=\$1.00$, holding costs outweigh shortage savings due to continuous multi-day safety stock accumulation. |
| **Simulated Cost Dynamics** | *"Under the simulated cost structure ($K_{\text{order}}=\$10, h=\$0.20/\text{unit/day}, p=\$1.00$), ordering frequency decreases due to uncertainty-driven order consolidation."* | Must explicitly state that inventory costs are derived from discrete-event simulation assumptions rather than proprietary enterprise accounting ledgers. |
| **Lost Sales Dynamics** | *"Under a lost-sales retail environment where unmet demand is permanently dropped, safety stock buffers provide critical downside protection."* | Must clarify that findings assume lost sales rather than full backlogging or partial customer backorders. |
| **Demand Regime Performance** | *"Forecast-conditional scaling improves interval adaptation across Medium and High volume tiers, though low-demand series remain prone to relative over-coverage."* | Must acknowledge non-uniform interval performance across demand regimes. |

---

## 3. DO NOT CLAIM (Strictly Prohibited & Unsupported Claims)

The following claims are scientifically invalid, unsupported by the evidence, or methodologically overreaching. **THEY MUST NEVER APPEAR IN THE MANUSCRIPT**:

| Strictly Prohibited Claim | Why It Is Prohibited | Required Corrected Framing |
| :--- | :--- | :--- |
| ❌ *"Conformal prediction guarantees 80%, 90%, or 95% out-of-sample coverage in retail demand."* | **Exchangeability violation**: 2017 test data underwent a 3.5x–4.5x residual variance expansion, causing empirical coverage to drop to 15.5%–39.5%. | State that empirical coverage degraded due to nonstationary temporal distribution shift, and frame $\beta$ as a nominal risk-tuning parameter. |
| ❌ *"Uncertainty-aware replenishment is universally cheaper / universally cost-dominant across all settings."* | **Empirically false**: At $L=3, 5, 7$ with $p=\$1.00$, higher holding costs make total simulated costs higher than deterministic replenishment. | Frame as a **cost-service trade-off frontier**, highlighting cost dominance only under high shortage penalty regimes ($p \ge \$2.00$). |
| ❌ *"Conformal safety stock eliminates all retail stockouts."* | **Empirically false**: Stockouts are substantially reduced (from 21%–67% down to 15%–54%), but not eliminated. | Report exact stockout rate reductions and lost sales units saved. |
| ❌ *"The proposed method is a K-Nearest Neighbors (KNN) local conformal prediction algorithm."* | **Methodological mismatch**: The optimal frozen parameter is $K^*=100$, which spans the entire sliding window $W=100$. | Describe the method as **"rolling forecast-conditional conformal calibration"** and treat $K$-sweeps as robustness audits. |
| ❌ *"The framework has been validated in live real-world retail store deployments."* | **Scope overreach**: Validated on 500 store-item benchmark series in rigorous discrete-event simulation, not live field deployment. | Explicitly describe as a simulation-based decision-support evaluation. |
| ❌ *"The framework generalizes to all supply chains (multi-echelon, stochastic lead times, capacity limits).'* | **Unchecked scope**: Evaluated on single-echelon, fixed lead times ($L \le 7$), and unconstrained supplier capacity. | List multi-echelon networks, stochastic lead times, and capacity constraints as explicit future research directions. |
