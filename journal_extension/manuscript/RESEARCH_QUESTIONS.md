# RESEARCH QUESTIONS & FORMAL RESEARCH EXPECTATIONS

All research questions are formulated based on the completed, frozen experimental evidence (EXP-011 through EXP-019). No additional experiments are required or permitted.

---

## Research Question 1 (Forecasting Architecture & Decision-Time Causality)

### **RQ1**: 
> *How does a strictly causal, direct multi-horizon gradient boosting architecture perform across varying forecast horizons ($h=1,\dots,7$) in a high-dimensional retail demand environment, and how does eliminating autoregressive feature leakage affect predictive error?*

### Formal Research Expectation ($E_1$):
- **Expectation**: Direct multi-horizon global XGBoost regressors ($M_1,\dots,M_7$) using features with minimum lag $\ge 7$ days will eliminate look-ahead leakage and produce stable point forecasts across all horizons, with minor increases in validation error compared to single-day lag baselines representing the cost of true decision-time operational safety.
- **Experimental Mapping**: **EXP-011** (Direct Multi-Horizon Implementation & Validation).
- **Empirical Verdict**: **CONFIRMED**. Validation MAE across horizons $h=1..7$ remained remarkably stable between 6.0285 and 6.0518 (aggregate MAE: **6.0388**, RMSE: **7.8638**), passing 10/10 automated temporal leakage checks.

---

## Research Question 2 (Conformal Calibration Under Temporal Distribution Shift)

### **RQ2**: 
> *How does rolling forecast-conditional conformal uncertainty calibration behave under forward retail demand distribution shift, and why does empirical coverage deviate from nominal confidence targets in nonstationary time series?*

### Formal Research Expectation ($E_2$):
- **Expectation**: Temporal distribution shifts (specifically variance expansion and seasonal phase misalignment) will cause classical rolling conformal prediction intervals to suffer empirical coverage degradation below nominal targets (80%, 90%, 95%), while scale-normalizing nonconformity scores by point forecasts will significantly improve interval efficiency and Winkler score relative to raw conformal intervals.
- **Experimental Mapping**: **EXP-013, EXP-014, EXP-015, EXP-016, EXP-017, EXP-018**.
- **Empirical Verdict**: **CONFIRMED**. Diagnostic decomposition (EXP-014) identified a 3.55x–4.58x residual variance expansion in 2017. Rolling forecast-conditional conformal calibration (EXP-018) significantly reduced mean Winkler loss (2198.81 vs 2244.38 for Raw, $p < 0.02$) and boosted high-demand coverage by +8.8 pp, while empirical coverage ranged from 15.5% to 39.5% due to non-exchangeability under distribution shift.

---

## Research Question 3 (Operational Value of Uncertainty in Inventory Replenishment)

### **RQ3**: 
> *How does incorporating non-parametric conformal uncertainty margins into replenishment reorder points affect customer fill rate, stockout frequency, lost sales, and average inventory investment across varying lead times ($L \in \{1, 3, 5, 7\}$) in a lost-sales environment?*

### Formal Research Expectation ($E_3$):
- **Expectation**: Integrating conformal uncertainty margins as dynamic safety stock ($ROP_{t,L} = \hat{D}_{t,L} + q_{t,L} \cdot \hat{D}_{t,L}$) will produce statistically significant improvements in aggregate fill rate and substantial reductions in lost sales across all lead times compared to deterministic point-forecast replenishment, accompanied by higher average inventory holding.
- **Experimental Mapping**: **EXP-019** (Inventory Decision-Support Simulation).
- **Empirical Verdict**: **CONFIRMED**. Conformal replenishment achieved statistically significant fill rate increases (+10.64 pp at $L=1$, +8.70 pp at $L=3$, +7.96 pp at $L=5$, +7.28 pp at $L=7$; paired Wilcoxon $p < 10^{-75}$) and reduced lost sales by 20.0% to 29.6% (preventing up to 1.14 million units of stockouts), while average inventory scaled proportionately with lead time.

---

## Research Question 4 (Economic Trade-offs & Shortage Cost Sensitivity)

### **RQ4**: 
> *How does the total operational cost of uncertainty-aware replenishment compare to deterministic replenishment across different lead-time constraints and asymmetric shortage penalty regimes ($p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$)?*

### Formal Research Expectation ($E_4$):
- **Expectation**: The economic dominance of uncertainty-aware replenishment will depend strongly on lead time and shortage penalties: for short lead times ($L=1$), it will reduce total operating costs unconditionally; for longer lead times ($L \ge 3$), it will establish a cost-service efficient frontier where conformal replenishment becomes cost-dominant whenever shortage penalties exceed holding cost accumulation.
- **Experimental Mapping**: **EXP-019** (Cost Sensitivity Grid & Frontier Analysis).
- **Empirical Verdict**: **CONFIRMED**. At $L=1$ ($p=\$1.00$), Conformal reduced total cost by 10.8% ($-\$790,834$, $p = 6.65 \times 10^{-55}$). At $L \ge 3$, higher holding stock at $p=\$1.00$ resulted in higher total simulated costs, but as shortage penalties increased ($p \ge \$2.00$), the conformal policy achieved total cost dominance across $L=1, 3, 5$ (e.g., at $L=3, p=\$4.00$, saving $\$2.37\text{M}$ in net costs).
