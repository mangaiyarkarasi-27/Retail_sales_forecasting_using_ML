# SECTION 4: RESULTS AND DISCUSSION

## 4.1 Experimental Evaluation Setting

To evaluate the proposed decision-support framework and address the core research questions (RQ1–RQ4), we present the empirical evidence collected across the complete, frozen experimental sequence (EXP-011 through EXP-020). The empirical evaluation was conducted on the benchmark dataset comprising $N=500$ daily store-item time series (10 retail stores $\times$ 50 merchandise items) observed over five full calendar years (2013-01-01 through 2017-12-31, totaling 913,000 daily observations).

The empirical evaluation strictly adheres to the chronological partitioning protocol established in Section 3:
1. **Model Training Partition (2013–2015, 1,095 days)**: Used exclusively for fitting the decoupled direct multi-horizon gradient boosted tree ensembles ($M_1, \dots, M_7$).
2. **Validation and Calibration Development Partition (2016, 366 days)**: Used for hyperparameter tuning, feature protocol verification, rolling-origin out-of-sample (OOS) residual calibration pooling, and calibration window/neighborhood robustness analysis.
3. **Untouched Out-of-Sample Test Partition (2017, 365 days)**: Preserved as a strictly forward, out-of-sample operational test bed. Zero 2017 observations were accessed during feature selection, model training, uncertainty parameter selection, or initial inventory sizing.

The operational evaluation investigates replenishment performance across four supplier lead times ($L \in \{1, 3, 5, 7\}$ days), three nominal conformal confidence levels ($\beta \in \{80\%, 90\%, 95\%\}$ with baseline $\beta = 90\%$), and four shortage penalty regimes ($p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$ per unfulfilled unit with baseline $p = \$1.00$), under fixed commercial ordering overhead ($K_{\text{order}} = \$10.00/\text{order}$) and daily holding carrying cost ($h = \$0.20/\text{unit/day}$).

---

## 4.2 Leakage-Free Multi-Horizon Forecasting Results

We first examine the predictive performance of the **Decoupled Direct Multi-Horizon Forecasting Architecture** (EXP-011). Under the causal feature engineering protocol, all dynamic autoregressive and rolling window indicators enforce a minimum lag constraint of $\tau \ge 7$ days. This guarantees that all seven horizon-specific models ($M_1, \dots, M_7$) operate without access to ground-truth demand realizations within the forward execution window $[t+1, t+7]$.

Table 3 reports the out-of-sample predictive accuracy and computational latency for each horizon model evaluated across the 182,500 daily forecasts of the 2016 validation partition.

```
TABLE 3: Multi-Horizon Validation Performance Across Horizons h=1 to 7 (2016 Partition)
========================================================================================================================
Forecast Horizon (h)   Validation MAE   Validation RMSE   Training Time (s)   Inference Time (s)   Model Architecture
------------------------------------------------------------------------------------------------------------------------
h = 1 day ahead             6.0416           7.8677             2.6753              0.0494         Global Direct XGBoost
h = 2 days ahead            6.0432           7.8667             1.0070              0.0546         Global Direct XGBoost
h = 3 days ahead            6.0366           7.8604             0.8832              0.0499         Global Direct XGBoost
h = 4 days ahead            6.0384           7.8622             0.8614              0.0492         Global Direct XGBoost
h = 5 days ahead            6.0318           7.8556             0.8543              0.0438         Global Direct XGBoost
h = 6 days ahead            6.0285           7.8544             0.8891              0.0464         Global Direct XGBoost
h = 7 days ahead            6.0518           7.8795             0.8741              0.0530         Global Direct XGBoost
------------------------------------------------------------------------------------------------------------------------
Portfolio Aggregate         6.0388           7.8638             8.0444              0.3463         7 Decoupled Ensembles
========================================================================================================================
```

### Empirical Observations on Forecasting Accuracy
1. **Error Stability Across Horizons**: Predictive accuracy remains relatively stable across the seven direct horizons in the validation period. The validation MAE ranges narrowly from $6.0285$ ($h=6$) to $6.0518$ ($h=7$), with an aggregate portfolio MAE of **$6.0388$** and RMSE of **$7.8638$**. Decoupling horizon estimation avoids the compounding error propagation inherent in iterative recursive forecasting.
2. **Computational Efficiency**: Training all seven global gradient boosted tree ensembles across 500 time series required a cumulative training latency of **8.04 seconds**, with total portfolio-wide forward inference executing in **0.35 seconds** (Figure 2).
3. **Protocol Clarification**: These validation error metrics reflect the decision-time-safe, minimum-lag $\ge 7$ feature configuration. They cannot be directly compared to recursive single-step error metrics from prior conference literature where 1-day lagged demand was iteratively fed forward. By decoupling horizon estimation and enforcing lag isolation, the forecasting layer provides a verified predictive foundation for downstream uncertainty calibration.

---

## 4.3 Raw Conformal Calibration Under Forward Distribution Shift

Having established point-forecasting stability, we evaluate the baseline **Rolling Raw Conformal Prediction** methodology (EXP-013) on the untouched 2017 out-of-sample test partition. Conformal prediction intervals were generated using an out-of-sample residual calibration pool ($N=608,165$ historical residuals) extracted via rolling-origin cross-validation over 2013–2016, with a sliding calibration window of $W=100$ days.

Table 4 presents the empirical coverage, mean interval width, and Winkler interval loss on the 2017 test partition across lead times $L \in \{1, 3, 5, 7\}$ and nominal confidence levels $\beta \in \{80\%, 90\%, 95\%\}$.

```
TABLE 4: Comprehensive Conformal Uncertainty Performance Comparison on 2017 Test Partition
=================================================================================================================================
Lead Time (L)  Nominal Cov  Raw Conformal Cov  Scale-Norm Cov  Hard-Regime Cov  Frozen K*=100 Cov  Raw Winkler  Scale Winkler  Frozen Winkler
---------------------------------------------------------------------------------------------------------------------------------
L = 1 day         80%            23.70%            20.60%          25.96%            25.57%          253.33        265.11         243.61
L = 1 day         90%            29.37%            25.54%          32.58%            31.98%          436.51        468.28         409.66
L = 1 day         95%            34.69%            30.04%          38.91%            37.88%          760.19        836.62         700.92
---------------------------------------------------------------------------------------------------------------------------------
L = 3 days        80%            17.98%            15.10%          19.46%            19.18%          804.26        839.83         792.60
L = 3 days        90%            21.67%            18.31%          23.35%            22.92%         1453.40       1544.76        1419.00
L = 3 days        95%            24.86%            21.16%          26.82%            26.17%         2669.94       2874.90        2591.12
---------------------------------------------------------------------------------------------------------------------------------
L = 5 days        80%            16.57%            13.77%          17.79%            17.57%         1360.15       1417.73        1351.29
L = 5 days        90%            19.47%            16.34%          20.82%            20.58%         2496.17       2639.85        2466.51
L = 5 days        95%            21.83%            18.47%          23.35%            22.92%         4674.83       4993.46        4594.95
---------------------------------------------------------------------------------------------------------------------------------
L = 7 days        80%            15.50%            12.96%          16.55%            16.46%         1930.11       2004.42        1924.66
L = 7 days        90%            17.90%            15.23%          19.03%            18.91%         3579.80       3759.76        3552.93
L = 7 days        95%            19.92%            17.00%          21.15%            21.03%         6757.34       7163.93        6675.53
=================================================================================================================================
```

### Empirical Coverage Degradation Under Distribution Shift
1. **Severe Empirical Under-Coverage**: Although the rolling raw conformal calibration pipeline was strictly leakage-safe, empirical coverage on the 2017 test partition fell substantially below nominal confidence targets across all lead times (Figure 3). At the primary 90% nominal confidence level:
   - For $L=1$, empirical coverage reached **29.37%** (coverage error: $60.63\text{ pp}$).
   - For $L=3$, empirical coverage reached **21.67%** (coverage error: $68.33\text{ pp}$).
   - For $L=5$, empirical coverage reached **19.47%** (coverage error: $70.53\text{ pp}$).
   - For $L=7$, empirical coverage reached **17.90%** (coverage error: $72.10\text{ pp}$).
2. **Horizon-Dependent Degradation**: Empirical coverage degradation increased with lead time $L$, while average interval width expanded from $34.37$ units at $L=1$ to $156.50$ units at $L=7$. Consequently, the Winkler interval loss escalated from $436.51$ to $3579.80$. At nominal 80% and 95% levels, empirical coverage similarly degraded to 15.50%–34.69%.
3. **Scientific Finding**: The experiment showed severe empirical coverage degradation when the 2017 evaluation distribution differed from the historical calibration regime. The observed behavior was inconsistent with the stability required for the historical calibration distribution to remain representative, documenting that static or raw rolling calibration residuals can fail under forward distribution shifts.

---

## 4.4 Why Did Coverage Degrade? Distribution-Shift Diagnosis

To investigate the empirical mechanisms underlying coverage degradation, we executed the **Distribution Shift Diagnostic Framework** (EXP-014), examining location and residual-scale characteristics between the calibration and test periods (Figure 5).

The empirical diagnostics identified two primary mechanisms in the evaluated retail dataset:

### 1. Residual-Scale and Error-Dispersion Expansion
Comparing the EXP-012 OOS calibration residuals (2013–2016) against 2017 test set residuals revealed a substantial expansion in error scale:
- **Residual Standard Deviation Expansion**: Expanded by **3.55x** at $L=1$ ($12.35 \to 43.89$), **4.09x** at $L=3$ ($31.77 \to 129.81$), **4.22x** at $L=5$ ($51.11 \to 215.47$), and **4.30x** at $L=7$ ($70.19 \to 301.60$).
- **Mean Absolute Error (MAE) Expansion**: Expanded by **3.76x** at $L=1$ ($9.26 \to 34.81$), **4.35x** at $L=3$ ($23.80 \to 103.57$), **4.49x** at $L=5$ ($38.35 \to 172.28$), and **4.58x** at $L=7$ ($52.75 \to 241.41$).
- **90th Percentile Error (P90) Expansion**: Expanded by **3.56x** at $L=1$ ($20.64 \to 73.53$) to **4.11x** at $L=7$ ($123.09 \to 505.92$).

Because rolling calibration windows sampled historical residuals from earlier, lower-dispersion periods, the extracted conformal quantiles were structurally undersized relative to the residual dispersion observed in 2017.

### 2. Seasonal Demand Shifts and Location Mismatch
Tracking quarterly mean signed forecast bias ($\text{Bias}_Q$) demonstrated seasonal phase misalignment. In $Q_1$ 2017, the point model over-predicted demand (mean error: $-18.84$ units/day, bias ratio: $-40.92\%$), whereas in peak summer $Q_2$ and $Q_3$, the model under-predicted demand (mean error: $+10.29$ units in $Q_2$, $+20.95$ units in $Q_3$).

### 3. Bilateral Violation Breakdown
At the nominal 90% level for $L=1$, lower tail violations ($\alpha_{\text{lower}} = 35.31\%$) and upper tail violations ($\alpha_{\text{upper}} = 35.32\%$) occurred with nearly identical frequency (total violations: $70.63\%$). This near-equal split demonstrates that coverage failure was driven by broad residual-error dispersion expansion and seasonal oscillations, rather than a simple one-sided monotonic annual demand trend.

> *Epistemological Boundary*: This diagnosis is an empirical finding for the evaluated retail dataset. The observed 2017 behavior is inconsistent with the stability required for the historical calibration distribution to remain representative.

---

## 4.5 Scale-Normalized Conformal Calibration

To test whether normalizing nonconformity scores by local volatility could mitigate the scale mismatch, we evaluated **Scale-Normalized Conformal Prediction** (EXP-015), which scales residuals by a trailing moving-average estimate of standard deviation ($\hat{\sigma}_{i,t,L}$).

As shown in Table 4, scale normalization reduced mean interval width by 14% to 17% (e.g., at $L=1, 90\%$, interval width shrank from $34.37$ to $29.49$ units). However, this interval narrowing led to **worse empirical 2017 coverage**:
- At $L=1$, empirical coverage decreased from **29.37%** (Raw) to **25.54%** (Scale-Normalized).
- At $L=3$, empirical coverage decreased from **21.67%** to **18.31%**.
- At $L=5$, empirical coverage decreased from **19.47%** to **16.34%**.
- At $L=7$, empirical coverage decreased from **17.90%** to **15.23%**.

Because interval narrowing occurred without restoring empirical coverage, the Winkler interval score degraded across all horizons (e.g., at $L=1, 90\%$, Winkler loss increased from $436.51$ to $468.28$, paired Wilcoxon $p < 10^{-40}$).

> *Core Finding*: Scale normalization alone did not restore empirical coverage. Trailing historical scale normalization lagged behind forward volatility expansion, producing tighter intervals that increased out-of-sample boundary breaches.

---

## 4.6 Forecast-Conditional Conformal Results

We next evaluated **Forecast-Conditional Conformal Calibration** (EXP-016), which normalizes nonconformity scores directly by forecasted demand magnitude ($s = |e| / (\hat{D} + \epsilon)$), conditioning uncertainty margins on expected sales volume.

As reported in Table 4 and illustrated in Figure 4:
1. **Empirical Coverage Changes**: Forecast-conditional calibration achieved higher empirical coverage than Raw and Scale-Normalized methods across all lead times:
   - $L=1$: Coverage reached **32.58%** (Hard-Regime) and **31.98%** (Frozen $K^*=100$), compared to **29.37%** (Raw) and **25.54%** (Scale).
   - $L=3$: Coverage reached **23.35%** (Hard) and **22.92%** ($K^*=100$), compared to **21.67%** (Raw).
   - $L=5$: Coverage reached **20.82%** (Hard) and **20.58%** ($K^*=100$), compared to **19.47%** (Raw).
   - $L=7$: Coverage reached **19.03%** (Hard) and **18.91%** ($K^*=100$), compared to **17.90%** (Raw).
2. **Winkler Score Improvements**: Forecast-conditional scaling achieved lower Winkler scores across all four lead times (e.g., at $L=1, 90\%$, Winkler loss improved from $436.51$ to **$409.66$** under $K^*=100$, paired Wilcoxon $p < 10^{-27}$).
3. **Important Boundary Qualification**: While forecast conditioning provided an empirical improvement in interval scoring and relative coverage, absolute coverage remained far below nominal 90%. Therefore, this represents an empirical improvement in interval scoring/calibration behavior, not a restoration of nominal coverage guarantees under distribution shift.

### Demand-Regime Sliced Analysis
Evaluating performance across demand volume tiers demonstrated that forecast conditioning does **not** improve all regimes equally. For single-day forecasts ($L=1, 90\%$):
- **High-Demand Series**: Empirical coverage increased from **28.03%** (Raw) to **37.30%** (Forecast-Conditional).
- **Medium-Demand Series**: Empirical coverage increased moderately from **36.30%** to **38.74%**.
- **Low-Demand Series**: Empirical coverage degraded from **26.57%** to **22.04%**, as percentage scaling tightened interval widths on low-velocity items.

Retaining this finding is critical: forecast-conditional calibration transfers uncertainty width toward high-velocity items where absolute unit risk is largest, but it can result in relative under-coverage on low-demand series.

---

## 4.7 K-Sensitivity and Final Uncertainty Configuration

In EXP-017 and EXP-018, we audited neighborhood sizes $K \in \{10, 20, 30, 40, 50, 60, 75, 100\}$ to determine whether local conditioning required fine-grained neighborhood filtering or broad rolling window pooling.

Table 5 presents the 2016 validation parameter sweep used to freeze the calibration configuration prior to 2017 test evaluation.

```
TABLE 5: Calibration Neighborhood Size (K) Sensitivity Profile on 2016 Validation Partition
========================================================================================================================
Method / Architecture     Neighborhood (K)   Mean Cov Error (2016)   Mean Winkler Loss (2016)   Relative Diff from Best
------------------------------------------------------------------------------------------------------------------------
Locally Weighted Conformal     K = 75               0.6393                   2087.59                Best Local K (0.00%)
Locally Weighted Conformal     K = 100              0.6430                   2099.35                +0.56% (Within 1%)
Forecast-Conditional Hard      Baseline             0.6408                   2100.06                +0.60% (Within 1%)
Locally Weighted Conformal     K = 50               0.6410                   2100.30                +0.61% (Within 1%)
Locally Weighted Conformal     K = 30               0.6382                   2101.51                +0.67% (Within 1%)
Locally Weighted Conformal     K = 60               0.6443                   2112.40                +1.19%
Locally Weighted Conformal     K = 40               0.6446                   2120.77                +1.59%
Locally Weighted Conformal     K = 20               0.6425                   2135.66                +2.30%
Locally Weighted Conformal     K = 10               0.6723                   2249.78                +7.77%
------------------------------------------------------------------------------------------------------------------------
Raw Conformal Benchmark        Baseline             0.6440                   2079.84                Benchmark
Scale-Normalized Benchmark     Baseline             0.6353                   2020.42                Benchmark
========================================================================================================================
```

### Identification of the Robustness Plateau
1. **Broad Stability Plateau ($K \in [30, 100]$)**: The validation sweep established a broad robustness plateau (Figure 6). Mean Winkler loss varied by less than **0.67%** across $K \in \{30, 50, 75, 100\}$, indicating that calibration performance is stable across broad neighborhood choices.
2. **Degradation at Small $K$**: For smaller neighborhoods ($K \le 20$), Winkler loss increased ($2135.66$ at $K=20$, $2249.78$ at $K=10$) due to finite-sample quantile variance when sampling from too few points.
3. **Selected Frozen Parameter ($K^*=100$)**: Following the predefined selection rule on 2016 validation data, the largest candidate within 1.0% of the minimum was selected: **$K^* = 100$**. This retains all $W=100$ observations within the rolling window, maximizing sample stability.
4. **2017 Test Comparison**: On the 2017 test partition, comparing $K=50$ (portfolio Winkler: **2200.27**) against $K^*=100$ (portfolio Winkler: **2198.81**) showed a difference of less than **0.1%** ($1.46$ units), confirming robustness without localized fragility.

---

## 4.8 Deterministic vs. Uncertainty-Aware Replenishment

We now present the central decision-support findings (EXP-019), comparing deterministic point-forecast replenishment ($ROP = \hat{D}$) against uncertainty-aware replenishment ($ROP = \hat{D} + SS^{(0.90)}$) across the 500 store-item series over the full 2017 test year.

Table 6 and Table 7 summarize inventory service and financial cost metrics at the primary shortage penalty $p = \$1.00/\text{unit}$ across all four lead times ($L \in \{1, 3, 5, 7\}$).

```
TABLE 6: Portfolio Inventory Service and Shortage Metrics (2017 Test Partition, Nominal 90%, p=$1.00)
=================================================================================================================================
Lead Time (L)  Policy Configuration     Demand-Weighted Fill Rate  Mean Series Fill Rate  Stockout Rate  OTIF Service  Shortage Units  Shortage Delta
---------------------------------------------------------------------------------------------------------------------------------
L = 1 day      Policy A (Deterministic)           46.87%                  55.28%             67.32%        32.68%       5,691,649           --
L = 1 day      Policy B (Conformal 90%)           57.51%                  65.64%             54.41%        45.59%       4,551,735     -1,139,914 (-20.03%)
---------------------------------------------------------------------------------------------------------------------------------
L = 3 days     Policy A (Deterministic)           65.26%                  73.26%             36.66%        63.34%       3,704,454           --
L = 3 days     Policy B (Conformal 90%)           73.96%                  80.45%             27.22%        72.78%       2,776,247       -928,208 (-25.06%)
---------------------------------------------------------------------------------------------------------------------------------
L = 5 days     Policy A (Deterministic)           71.35%                  78.75%             26.54%        73.46%       3,042,716           --
L = 5 days     Policy B (Conformal 90%)           79.31%                  84.92%             18.86%        81.14%       2,197,856       -844,860 (-27.77%)
---------------------------------------------------------------------------------------------------------------------------------
L = 7 days     Policy A (Deterministic)           75.36%                  82.16%             21.54%        78.46%       2,608,148           --
L = 7 days     Policy B (Conformal 90%)           82.64%                  87.59%             14.97%        85.03%       1,837,018       -771,130 (-29.57%)
=================================================================================================================================
```

```
TABLE 7: Portfolio Inventory Financial Cost Breakdown and Efficiency (2017 Test Partition, Nominal 90%, p=$1.00)
=================================================================================================================================
Lead Time (L)  Policy Configuration     Average On-Hand  Orders Placed  Ordering Cost   Holding Cost   Shortage Cost    Total Cost     Cost/Demand
---------------------------------------------------------------------------------------------------------------------------------
L = 1 day      Policy A (Deterministic)      9.44 units     127,016      $1,270,160      $343,722       $5,691,649      $7,305,531       $0.6820
L = 1 day      Policy B (Conformal 90%)     20.49 units     121,664      $1,216,640      $746,322       $4,551,735      $6,514,697       $0.6082
               *Delta (Policy B - A)*       +11.05 units      -5,352       -$53,520     +$402,600      -$1,139,914       -$790,834      -$0.0738 (-10.83%)
---------------------------------------------------------------------------------------------------------------------------------
L = 3 days     Policy A (Deterministic)     64.97 units      83,545        $835,450    $2,353,426       $3,704,454      $6,893,330       $0.6464
L = 3 days     Policy B (Conformal 90%)    103.23 units      79,125        $791,250    $3,738,924       $2,776,247      $7,306,420       $0.6852
               *Delta (Policy B - A)*       +38.25 units      -4,420       -$44,200   +$1,385,498        -$928,208       +$413,090      +$0.0388 (+5.99%)
---------------------------------------------------------------------------------------------------------------------------------
L = 5 days     Policy A (Deterministic)    134.03 units      65,516        $655,160    $4,827,921       $3,042,716      $8,525,797       $0.8027
L = 5 days     Policy B (Conformal 90%)    198.22 units      62,251        $622,510    $7,140,208       $2,197,856      $9,960,575       $0.9378
               *Delta (Policy B - A)*       +64.19 units      -3,265       -$32,650   +$2,312,287        -$844,860     +$1,434,777      +$0.1351 (+16.83%)
---------------------------------------------------------------------------------------------------------------------------------
L = 7 days     Policy A (Deterministic)    207.04 units      56,549        $565,490    $7,416,471       $2,608,148     $10,590,109       $1.0007
L = 7 days     Policy B (Conformal 90%)    295.10 units      53,900        $539,000   $10,570,690       $1,837,018     $12,946,708       $1.2234
               *Delta (Policy B - A)*       +88.05 units      -2,649       -$26,490   +$3,154,219        -$771,130     +$2,356,599      +$0.2227 (+22.25%)
=================================================================================================================================
```

---

## 4.9 Service-Level Effects Across Lead Times

In the evaluated discrete-event simulation, the uncertainty-aware policy produced higher fill rates across all evaluated lead times (Figure 7):

1. **Demand-Weighted Fill Rate Differences**:
   - **$L=1$**: Demand-weighted fill rate increased from **46.87%** (Deterministic) to **57.51%** (Conformal 90%), a gain of **+10.64 percentage points** ($p = 1.26 \times 10^{-83}$). The mean series-level paired fill rate increased from **55.28%** to **65.64%** (+10.36 pp).
   - **$L=3$**: Demand-weighted fill rate increased from **65.26%** to **73.96%**, a gain of **+8.70 percentage points** ($p = 5.69 \times 10^{-83}$). Mean series-level fill rate increased from **73.26%** to **80.45%** (+7.19 pp).
   - **$L=5$**: Demand-weighted fill rate increased from **71.35%** to **79.31%**, a gain of **+7.95 percentage points** ($p = 2.33 \times 10^{-80}$). Mean series-level fill rate increased from **78.75%** to **84.92%** (+6.17 pp).
   - **$L=7$**: Demand-weighted fill rate increased from **75.36%** to **82.64%**, a gain of **+7.29 percentage points** ($p = 1.94 \times 10^{-76}$). Mean series-level fill rate increased from **82.16%** to **87.59%** (+5.43 pp).
2. **Lost Sales Reductions**: Across the 500 store-item series, the conformal safety buffer reduced unfulfilled customer demand by:
   - **1,139,914 units** at $L=1$ (a 20.03% shortage reduction).
   - **928,208 units** at $L=3$ (a 25.06% shortage reduction).
   - **844,860 units** at $L=5$ (a 27.77% shortage reduction).
   - **771,130 units** at $L=7$ (a 29.57% shortage reduction).
3. **Stockout Frequency Changes**: Day-based stockout probability dropped from $67.32\%$ to $54.41\%$ at $L=1$, from $36.66\%$ to $27.22\%$ at $L=3$, from $26.54\%$ to $18.86\%$ at $L=5$, and from $21.54\%$ to $14.97\%$ at $L=7$.
4. **Order Consolidation**: Conformal safety stock consolidated replenishment orders, reducing the total purchase orders placed by **5,352 orders** at $L=1$, **4,420 orders** at $L=3$, **3,265 orders** at $L=5$, and **2,649 orders** at $L=7$, resulting in administrative ordering cost savings of $\$26,490$ to $\$53,520$.

---

## 4.10 Service-Cost Trade-Off

The financial results demonstrate that **higher service levels do not automatically imply lower total operational costs** across all lead times. The economic outcome depends directly on the balance between inventory holding costs and shortage penalties.

### 1. The Single-Day Evaluated Scenario ($L=1$)
For single-day replenishment ($L=1$), the Conformal policy achieved **lower total simulated cost in the evaluated scenario** (saving **\$790,834** annually across the portfolio, a **10.83%** reduction, $p = 6.65 \times 10^{-55}$):
- Ordering Cost Savings: $-\$53,520$
- Shortage Cost Savings: $-\$1,139,914$
- Holding Cost Increase: $+\$402,600$
- **Net Simulated Cost Difference**: **$-\$790,834$** (Cost per demanded unit decreased from $\$0.6820$ to $\$0.6082$; cost per fulfilled unit decreased from $\$1.4552$ to $\$1.0575$).

At $L=1$, stock is replenished daily, requiring smaller holding stock increases ($20.49$ vs. $9.44$ units). Consequently, shortage and ordering cost reductions outweighed the holding cost increase.

### 2. Multi-Day Lead Times ($L \ge 3$) and Holding Cost Accumulation
For multi-day horizons ($L \in \{3, 5, 7\}$) at the baseline penalty $p = \$1.00$, maintaining safety stock over multi-day delivery intervals required carrying higher average inventory ($103.23$ vs $64.97$ units at $L=3$; $295.10$ vs $207.04$ units at $L=7$). Because holding costs accumulate across the lead time, holding cost increases exceeded shortage cost reductions at $p = \$1.00$, resulting in higher total simulated operating costs (+5.99% at $L=3$, +16.83% at $L=5$, +22.25% at $L=7$).

### 3. Asymmetric Shortage Penalty Sensitivity ($p \in \{\$0.60, \$1.00, \$2.00, \$4.00\}$)
Table 8 summarizes simulated operating costs across penalty regimes (Figure 8).

```
TABLE 8: Shortage Penalty Sensitivity Across Lead Times L in {1, 3, 5, 7}
========================================================================================================================
Lead Time (L)   Penalty (p)   Deterministic Cost   Conformal (90%) Cost   Dollar Delta (B - A)   Percentage Delta   Economic Preference
------------------------------------------------------------------------------------------------------------------------
L = 1 day          $0.60          $5,028,872            $4,694,003             -$334,868              -6.66%        Conformal Lower Cost
L = 1 day          $1.00          $7,305,531            $6,514,697             -$790,834             -10.83%        Conformal Lower Cost
L = 1 day          $2.00         $12,997,181           $11,066,433           -$1,930,748             -14.86%        Conformal Lower Cost
L = 1 day          $4.00         $24,380,480           $20,169,904           -$4,210,576             -17.27%        Conformal Lower Cost
------------------------------------------------------------------------------------------------------------------------
L = 3 days         $0.60          $5,411,549            $6,195,921             +$784,373             +14.49%        Deterministic Lower Cost
L = 3 days         $1.00          $6,893,330            $7,306,420             +$413,090              +5.99%        Deterministic Lower Cost
L = 3 days         $2.00         $10,597,785           $10,082,667             -$515,118              -4.86%        Conformal Lower Cost
L = 3 days         $4.00         $18,006,694           $15,635,160           -$2,371,534             -13.17%        Conformal Lower Cost
------------------------------------------------------------------------------------------------------------------------
L = 5 days         $0.60          $7,308,711            $9,081,432           +$1,772,721             +24.26%        Deterministic Lower Cost
L = 5 days         $1.00          $8,525,797            $9,960,575           +$1,434,777             +16.83%        Deterministic Lower Cost
L = 5 days         $2.00         $11,568,514           $12,158,431             +$589,918              +5.10%        Deterministic Lower Cost
L = 5 days         $4.00         $17,653,946           $16,554,144           -$1,099,802              -6.23%        Conformal Lower Cost
------------------------------------------------------------------------------------------------------------------------
L = 7 days         $0.60          $9,546,850           $12,211,901           +$2,665,051             +27.92%        Deterministic Lower Cost
L = 7 days         $1.00         $10,590,109           $12,946,708           +$2,356,599             +22.25%        Deterministic Lower Cost
L = 7 days         $2.00         $13,198,257           $14,783,726           +$1,585,469             +12.01%        Deterministic Lower Cost
L = 7 days         $4.00         $18,414,554           $18,457,763              +$43,209              +0.23%        Parity (+0.2%)
========================================================================================================================
```

Under higher shortage penalty settings:
- At $L=3$, the Conformal policy achieves lower total cost for $p \ge \$2.00$ (saving **\$515,118** at $p=\$2.00$ and **\$2,371,534** at $p=\$4.00$).
- At $L=5$, the Conformal policy is cost-preferred at $p=\$4.00$ (saving **\$1,099,802**).
- At $L=7$, the two policies reach parity at $p=\$4.00$ (+0.2% difference).

---

## 4.11 Efficient-Frontier Interpretation

The experimental results characterize an **Empirical Service–Cost Trade-Off Frontier** (Figure 9).

Evaluating nominal coverage levels $\beta \in \{80\%, 90\%, 95\%\}$ demonstrated consistent operational responsiveness:
- At $L=1$: Demand-weighted fill rate scaled from **46.87%** (Deterministic) $\to$ **55.67%** ($\beta=0.80$) $\to$ **57.51%** ($\beta=0.90$) $\to$ **59.09%** ($\beta=0.95$). Total simulated operating costs decreased from \$7.31M $\to$ \$6.63M $\to$ \$6.51M $\to$ \$6.42M.
- At $L=3$: Fill rate scaled from **65.26%** $\to$ **72.78%** $\to$ **73.96%** $\to$ **74.89%**, while ending inventory increased from $64.97 \to 96.20 \to 103.23 \to 109.28$ units.
- At $L=5$: Fill rate scaled from **71.35%** $\to$ **78.39%** $\to$ **79.31%** $\to$ **79.98%**.
- At $L=7$: Fill rate scaled from **75.36%** $\to$ **81.92%** $\to$ **82.64%** $\to$ **83.23%**.

Tuning the nominal coverage parameter $\beta \in [0.80, 0.95]$ allows inventory planners to adjust the empirical trade-off between customer service commitments and inventory holding investment.

---

## 4.12 Portfolio and Series-Level Statistical Evidence

To evaluate whether observed service and cost differences are statistically significant across the 500 store-item series, we analyzed per-series paired metric deltas using **Two-Sided Paired Wilcoxon Signed-Rank Tests** and **1,000-Iteration Paired Bootstrap Confidence Intervals**.

Table 10 reports inferential statistical results for the primary configuration ($\beta=0.90, p=\$1.00$).

```
TABLE 10: Paired Inferential Statistics and 95% Bootstrap Confidence Intervals (500 Series, Nominal 90%, p=$1.00)
=================================================================================================================================
Lead Time (L)  Metric                     Mean Delta (\bar{\Delta})   95% Bootstrap CI Lower   95% Bootstrap CI Upper   Wilcoxon p-value
---------------------------------------------------------------------------------------------------------------------------------
L = 1 day      Fill Rate (Percentage)           +10.36 pp                  +10.19 pp                +10.55 pp           1.26 x 10^-83
L = 1 day      Shortage Units                 -2,279.8 units             -2,395.9 units           -2,174.7 units        1.26 x 10^-83
L = 1 day      Average Inventory               +11.05 units               +10.19 units             +11.88 units         1.26 x 10^-83
L = 1 day      Total Operational Cost           -$1,581.67                -$1,735.40               -$1,434.24           6.65 x 10^-55
---------------------------------------------------------------------------------------------------------------------------------
L = 3 days     Fill Rate (Percentage)            +7.19 pp                   +6.88 pp                 +7.51 pp           5.69 x 10^-83
L = 3 days     Shortage Units                 -1,856.4 units             -1,993.6 units           -1,726.0 units        5.69 x 10^-83
L = 3 days     Average Inventory               +38.25 units               +36.27 units             +40.22 units         1.26 x 10^-83
L = 3 days     Total Operational Cost             +$826.18                  +$595.64                +$1,059.38          1.44 x 10^-07
---------------------------------------------------------------------------------------------------------------------------------
L = 5 days     Fill Rate (Percentage)            +6.17 pp                   +5.78 pp                 +6.55 pp           2.33 x 10^-80
L = 5 days     Shortage Units                 -1,689.7 units             -1,837.8 units           -1,540.4 units        2.33 x 10^-80
L = 5 days     Average Inventory               +64.19 units               +61.29 units             +67.58 units         1.26 x 10^-83
L = 5 days     Total Operational Cost           +$2,869.55                +$2,585.73               +$3,173.88           2.39 x 10^-51
---------------------------------------------------------------------------------------------------------------------------------
L = 7 days     Fill Rate (Percentage)            +5.43 pp                   +5.08 pp                 +5.77 pp           1.94 x 10^-76
L = 7 days     Shortage Units                 -1,542.3 units             -1,679.0 units           -1,402.4 units        1.94 x 10^-76
L = 7 days     Average Inventory               +88.05 units               +83.93 units             +91.98 units         1.26 x 10^-83
L = 7 days     Total Operational Cost           +$4,713.20                +$4,342.41               +$5,052.34           4.15 x 10^-78
=================================================================================================================================
```

### Statistical Reconciliation and Analysis
1. **Reconciliation of Fill-Rate Metrics**:
   - The **portfolio demand-weighted fill-rate improvement** at $L=1$ is **+10.64 percentage points** ($46.87\% \to 57.51\%$), weighting each unit of demand equally across all series.
   - The **mean per-series paired fill-rate improvement** across the 500 individual series is **+10.36 percentage points** ($55.28\% \to 65.64\%$), with median difference $+10.38\text{ pp}$. The 1,000-iteration paired bootstrap 95% confidence interval for this series-level mean delta is **$[+10.19\text{ pp}, +10.55\text{ pp}]$**.
2. **Statistical Significance of Service Differences**: Fill rate improvements and shortage unit reductions yield Wilcoxon $p < 10^{-75}$ across all 500 series. The 95% bootstrap confidence intervals for series-level fill rate improvements strictly exclude zero across all horizons.
3. **Cost Significance**: Total cost differences are statistically significant across all configurations, confirming that cost reductions at $L=1$ (mean delta: $-\$1,581.67/\text{series}$, 95% CI: $[-\$1,735.40, -\$1,434.24]$) and holding cost increases at $L \ge 3$ reflect systematic differences in simulated inventory dynamics.

---

## 4.13 From Forecast Accuracy to Decision Quality

Synthesizing the empirical evidence from EXP-011 through EXP-020 highlights the relationship between point-forecast accuracy and inventory replenishment performance.

The empirical evidence chain comprises six connected stages:
1. **Stage 1 (Causal Forecasting Foundation)**: Decoupled direct XGBoost models achieved stable predictive accuracy ($\text{MAE} \approx 6.04$ across horizons), satisfying decision-time causality without recursive error accumulation.
2. **Stage 2 (Conformal Coverage Under Nonstationarity)**: Raw rolling conformal calibration experienced severe empirical under-coverage (15.5%–34.7% coverage at nominal 90%), demonstrating that temporal distribution shifts invalidate standard exchangeability assumptions in retail operations.
3. **Stage 3 (Distribution Shift Diagnosis)**: Empirical diagnostics identified residual standard deviation expansion (3.55x to 4.30x) and seasonal demand shifts as the primary drivers of interval under-coverage.
4. **Stage 4 (Forecast-Conditional Calibration)**: Normalizing nonconformity scores by forecasted demand magnitudes captured heteroskedasticity, lowering Winkler interval loss relative to raw and scale-normalized alternatives.
5. **Stage 5 (Uncertainty-to-Replenishment Translation)**: Mapping upper conformal prediction bounds into dynamic safety stock buffers ($SS_{t,L} = q_{t,L} \cdot \hat{D}_{t,L}$) provided an asymmetric mechanism for buffering demand fluctuations.
6. **Stage 6 (Operational Simulation)**: In discrete-event simulation, uncertainty-aware replenishment produced higher fill rates (+7.29 to +10.64 pp demand-weighted, reducing lost sales by 20.0% to 29.6%), yielding lower total costs at $L=1$ (-10.83%) and an empirical cost-service trade-off frontier for multi-day lead times ($L \ge 3$).

---

## 4.14 Conference-to-Journal Experimental Extension

Table 9 provides a structural comparison distinguishing the preliminary conference baseline study \cite{e_idss_ieee} from the completed journal extension.

```
TABLE 9: Methodological and Empirical Comparison Between Conference Baseline and Journal Extension
========================================================================================================================
Methodological Dimension       Conference Baseline (E-IDSS) \cite{e_idss_ieee}      Journal Extension (This Study)
------------------------------------------------------------------------------------------------------------------------
Primary Research Scope         Single-step point forecasting & XAI ranking          End-to-end UQ & inventory decision support
Forecasting Architecture       Single-step global XGBoost regression                7 Decoupled Global Direct XGBoost models (M1..M7)
Feature Temporal Protocol      Standard lag indicators (potential leak at L>1)      Strictly causal protocol (Minimum lag >= 7 days)
Calibration Data Source        In-sample training residuals                         Rolling-Origin Out-of-Sample Pool (N=608,165)
Uncertainty Quantification     Heuristic MCDA priority scoring (volatility index)   Rolling Forecast-Conditional Conformal Calibration
Distribution Shift Analysis    Assumed stationary operational environment           Location-scale & violation asymmetry diagnostics
Robustness & Window Audit      Fixed single heuristic scoring rule                  Continuum sweep across K in [10, 100] (K*=100)
Inventory Decision Policy      Categorical recommendations (Increase/Cut)           Continuous-review order-up-to policy (ROP = D + SS)
Simulation Engine              No physical inventory simulation executed            8-step daily discrete-event simulation (Lost Sales)
Lead-Time Evaluation           Implicit single-period assumption                    Evaluated across 4 lead times (L in {1, 3, 5, 7})
Cost Model & Financials        No financial accounting model                        Holding ($0.20), Ordering ($10), Shortage ($1.00)
Shortage Penalty Sensitivity   None (fixed qualitative priority)                    Sensitivity grid across p in {$0.60, $1, $2, $4}
Statistical Significance       Point error metrics (MAE, RMSE, R2)                  Paired Wilcoxon (p < 10^-75) + 1,000 bootstrap CIs
Portfolio Evaluation Scale     Static portfolio summary                             500 series simulated daily over 365 test days
========================================================================================================================
```

By connecting causal multi-horizon forecasting, scale-adaptive conformal calibration, and discrete-event inventory simulation, this study presents a decision-centric empirical framework for retail replenishment planning.
