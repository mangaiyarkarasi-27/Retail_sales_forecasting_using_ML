# FINAL SCIENTIFIC VERDICT — JOURNAL EXTENSION

**Document Reference**: EXP-020 Final Audit & Evidence Freeze  
**Date**: 2026-09-19  
**Review Status**: COMPLETE — EXPERIMENTAL PHASE TERMINATED

---

## 1. Overall Status

### **SUPPORTED**

The experimental program (EXP-011 through EXP-019) has successfully established a leakage-free, statistically verified, and operationally meaningful decision-support framework that combines direct multi-horizon forecasting, rolling forecast-conditional conformal calibration, and inventory replenishment simulation.

---

## 2. What the Evidence Demonstrates (Directly Supported Findings)

1. **Leakage-Free Multi-Horizon Forecasting**: Direct global XGBoost models with lag $\ge 7$ features completely eliminate autoregressive look-ahead leakage while achieving high point accuracy (validation MAE 6.0388).
2. **Statistically Significant Service Level Gains**: Integrating forecast-conditional conformal uncertainty quantification into replenishment reorder points achieves massive, statistically significant fill rate improvements across all 500 store-item series and all evaluated lead times (+7.28 to +10.64 percentage points, paired Wilcoxon $p < 10^{-75}$).
3. **Substantial Lost Sales Reduction**: Conformal safety stock buffers prevent 20.0% to 29.6% of lost sales across horizons, saving up to 1.14 million units of unmet customer demand annually.
4. **Unconditional Economic Superiority at $L=1$**: For single-day replenishment, uncertainty awareness is a pure win-win, improving fill rate by +10.6 pp while lowering total operational costs by 10.8% (saving $\$790,834$, $p = 6.65 \times 10^{-55}$) due to order consolidation and shortage elimination.
5. **Cost-Service Efficient Frontier at $L \ge 3$**: For multi-day lead times ($L \in \{3, 5, 7\}$), conformal uncertainty creates a well-defined efficient frontier trading holding stock for service gains, achieving lower total costs whenever shortage penalties exceed $\$2.00$.
6. **Strict Monotonic Risk Control**: Tuning nominal coverage $\beta \in \{80\%, 90\%, 95\%\}$ monotonically shifts service level, stockout frequency, and holding inventory across all series.
7. **Robust Calibration Window**: Calibration performance exhibits a broad stability plateau across neighborhood sizes $K \in [30, 100]$, validating the selection of the full rolling window $W=100$.

---

## 3. What the Evidence Does Not Demonstrate (Unsupported or Overstrong Claims)

1. **No Formal Finite-Sample Coverage Guarantee on 2017**: Conformal prediction's exchangeability assumption is violated by out-of-sample temporal distribution shift (a 3.5x-4.5x residual variance expansion in 2017). Empirical coverage ranged from 15.5% to 39.5%, far below nominal 80%-95% levels. The journal manuscript must NOT claim guaranteed coverage calibration on 2017 data.
2. **No Universal Cost Minimization**: Conformal replenishment does NOT universally minimize total costs across all parameters. At $L \ge 3$ under moderate shortage penalties ($p=\$1.00$), holding costs exceed shortage savings. It must be presented as an efficient frontier / decision-support tool rather than an unconditionally cost-dominant policy.
3. **No Generalization Beyond Evaluated Grid**: Findings are validated on the 500 store-item benchmark series under lost sales and fixed lead times, and cannot be claimed for multi-echelon networks, stochastic lead times, or capacity-constrained suppliers without further empirical research.

---

## 4. Main Contribution

> **Primary Contribution**: A rigorous, leakage-free decision-support framework that bridges direct multi-horizon machine learning forecasting, rolling forecast-conditional conformal uncertainty quantification, and discrete-event inventory optimization, demonstrating empirically that non-parametric conformal safety stock delivers statistically significant fill rate gains and establishes an actionable cost-service efficient frontier under realistic demand shifts.

---

## 5. Main Limitation

> **Primary Limitation**: Forward temporal distribution shift and variance expansion cause conformal empirical coverage to fall substantially below nominal target levels, demonstrating that while forecast-conditional scaling improves interval efficiency relative to raw conformal baselines, rolling historical residuals alone cannot fully absorb abrupt forward macroeconomic volatility.

---

## 6. Journal Readiness

### **EXPERIMENTAL PHASE COMPLETE — READY FOR JOURNAL MANUSCRIPT PREPARATION**

- No critical methodological flaws, data leakages, or statistical inconsistencies were found.
- All 15 required experimental result files, 14 publication figures, and 7 audit CSV tables are verified, consistent, and permanently frozen.
- The study is fully reproducible from the provided codebase and pre-processed data.
- **NO FURTHER EXPERIMENTS ARE REQUIRED OR PERMITTED.**
