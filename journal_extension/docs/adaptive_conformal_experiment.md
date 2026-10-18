# EXP-015: Scale-Adaptive Conformal Uncertainty

This document reports the methodology, validation, and final test results for **EXP-015: Scale-Adaptive Conformal Uncertainty Calibration**.

---

## 1. Research Question

> **Research Question**: *"Can scale-normalized conformal calibration improve uncertainty coverage under the observed temporal scale shift while preserving leakage-free decision-time operation?"*

EXP-015 tests whether normalizing conformal nonconformity scores by a trailing local error-scale estimate resolves the empirical under-coverage observed in EXP-013 and diagnosed in EXP-014, without utilizing future test information.

---

## 2. Motivation from EXP-014

In EXP-014, exhaustive diagnostics revealed:
1.  **Scale / Variance Explosion**: The standard deviation and MAE of forecast errors in the 2017 test period expanded by **$3.55\times$ to $4.58\times$** compared to historical calibration-era residuals.
2.  **Symmetric Violations**: Interval violations were symmetrically distributed (~35% lower and ~35% upper breaches at 90% nominal), indicating that interval width collapse (underestimated spread) was the primary failure mode rather than a simple monotonic demand bias.
3.  **Quantile Underestimation**: Static rolling conformal quantiles ($W=100$) derived from historical training folds systematically underestimated actual 2017 errors in **70% to 82%** of test observations.

---

## 3. Data Partitioning

Strict chronological partitioning for method development and evaluation:
*   **Development Period (2013–2015)**: Used for training-derived profile baseline calculations and historical residual generation.
*   **Method-Validation Period (2016)**: Used exclusively for evaluating candidate adaptive conformal methods, selecting the optimal configuration, and freezing the methodology.
*   **Final Test Period (2017)**: Kept completely untouched until the adaptive method and all parameters were frozen.

---

## 4. Frozen Raw Conformal Baseline

The baseline reproduces the frozen rolling-origin conformal calibration from EXP-013:
*   **Calibration Window**: $W = 100$
*   **Nominal Coverage Levels**: $80\%, 90\%, 95\%$
*   **Lead Times**: $L \in \{1, 3, 5, 7\}$
*   **Temporal Eligibility**: Only historical residuals $r$ satisfying $r \le t - L - 1$ are eligible at forecast origin $t$.
*   **Nonconformity Score**: Raw absolute lead-time error $S_{r,L} = |E_{r,L}|$.

---

## 5. Scale-Normalized Nonconformity Score

For candidate Method A (**Scale-Normalized Conformal**), the nonconformity score is defined as:

$$S_{r,L} = \frac{|E_{r,L}|}{\max(\sigma_{r,L}, \epsilon)}$$

where:
*   $|E_{r,L}|$ is the absolute cumulative lead-time forecast error: $|D_{r,L} - \hat{D}_{r,L}|$.
*   $\sigma_{r,L}$ is the local trailing error-scale estimate available prior to origin $r$.
*   $\epsilon = 10^{-4}$ is a numerical stability constant.

For candidate Method B (**Volume-Normalized Conformal**), the nonconformity score is defined as:

$$S_{r,L} = \frac{|E_{r,L}|}{\max(\bar{y}_{\text{volume}}, \epsilon)}$$

where $\bar{y}_{\text{volume}}$ is the series-specific mean demand derived strictly from the training period (2013–2015).

---

## 6. Scale Estimation

The scale estimator $\sigma_{t,L}$ is computed strictly from previously realized out-of-sample residuals:

$$\sigma_{t,L} = \frac{1}{K} \sum_{k=1}^{K} |E_{r_k, L}|$$

where $\{r_1, \dots, r_K\}$ are the $K = 50$ most recent eligible historical forecast origins satisfying $r_k \le t - L - 1$. No future actual sales or future prediction errors enter $\sigma_{t,L}$.

---

## 7. Temporal Eligibility

At decision time $t$, the adaptive conformal quantile is constructed as:
1.  Filter historical OOS residuals to eligible set: $\mathcal{R}_t = \{r : r \le t - L - 1\}$.
2.  Compute trailing scale $\sigma_{r,L}$ for each historical point and calculate normalized scores $S_{r,L}$.
3.  Compute finite-sample conformal quantile $q_{\text{norm}}(t,L)$ at level $1 - \alpha = \lceil (n+1)(1-\alpha) \rceil / n$ over calibration window $W=100$.
4.  Compute current scale estimate $\sigma_{t,L}$ from $\mathcal{R}_t[-50:]$.
5.  Construct adaptive interval:

$$q_{\text{adaptive}}(t,L) = q_{\text{norm}}(t,L) \times \sigma_{t,L}$$

$$\text{Lower}_{t,L} = \max\left(0, \hat{D}_{t,L} - q_{\text{adaptive}}(t,L)\right), \quad \text{Upper}_{t,L} = \hat{D}_{t,L} + q_{\text{adaptive}}(t,L)$$

---

## 8. Method Validation on 2016

Candidate methods were evaluated on the 2016 validation period across all 500 store-item combinations:

| Method | Lead Time ($L$) | Nominal Cov | Emp Cov (2016) | Cov Error | Mean Width | Winkler Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw Conformal** | 1 | 0.80 | 0.2572 | 0.5428 | 29.65 | 234.46 |
| **Raw Conformal** | 1 | 0.90 | 0.3218 | 0.5782 | 36.43 | 395.56 |
| **Raw Conformal** | 1 | 0.95 | 0.3766 | 0.5734 | 42.22 | 680.28 |
| **Scale Normalized** | 1 | 0.80 | **0.2606** | **0.5394** | 30.43 | **231.20** |
| **Scale Normalized** | 1 | 0.90 | **0.3252** | **0.5748** | 37.16 | **388.55** |
| **Scale Normalized** | 1 | 0.95 | **0.3822** | **0.5678** | 42.97 | **665.55** |
| **Scale Normalized** | 7 | 0.90 | **0.2081** | **0.6919** | 180.19 | **3218.50** |
| **Raw Conformal** | 7 | 0.90 | 0.1969 | 0.7031 | 168.70 | 3304.93 |
| **Volume Normalized**| 1 | 0.90 | 0.3218 | 0.5782 | 36.43 | 395.56 |

### Overall 2016 Summary Ranking

| Method | Mean Coverage Error (2016) | Mean Winkler Score (2016) | Mean Interval Width (2016) |
|:---|:---:|:---:|:---:|
| **Scale Normalized** | **0.6386** | **2084.85** | 109.55 |
| **Raw Conformal** | 0.6482 | 2154.89 | 103.42 |
| **Volume Normalized**| 0.6482 | 2154.89 | 103.42 |

---

## 9. Method Selection Rule

*   **Predefined Rule**: Select the candidate method that minimizes mean Winkler Interval Score while minimizing empirical coverage error on the 2016 validation period.
*   **Selected and Frozen Method**: **`Scale_Normalized`** (Method A).

---

## 10. Frozen Final Evaluation on 2017

The frozen `Scale_Normalized` method was evaluated on the untouched 2017 test set against the frozen `Raw_Conformal` baseline across all 500 series (182,100 evaluated origin-lead pairs per configuration).

---

## 11. Coverage Results (2017 Test Set)

| Lead Time ($L$) | Nominal Coverage | Raw Emp Coverage | Adaptive Emp Coverage | Coverage Change ($\Delta$) | Raw Coverage Error | Adaptive Coverage Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$L=1$** | 80% | **23.70%** | 20.60% | -3.10% | 56.30% | 59.40% (+3.10%) |
| **$L=1$** | 90% | **29.37%** | 25.54% | -3.83% | 60.63% | 64.46% (+3.83%) |
| **$L=1$** | 95% | **34.69%** | 30.04% | -4.65% | 60.31% | 64.96% (+4.65%) |
| **$L=3$** | 80% | **17.98%** | 15.10% | -2.88% | 62.02% | 64.90% (+2.88%) |
| **$L=3$** | 90% | **21.67%** | 18.31% | -3.36% | 68.33% | 71.69% (+3.36%) |
| **$L=3$** | 95% | **24.86%** | 21.16% | -3.70% | 70.14% | 73.84% (+3.70%) |
| **$L=5$** | 90% | **19.47%** | 16.34% | -3.13% | 70.53% | 73.66% (+3.13%) |
| **$L=7$** | 90% | **17.90%** | 15.23% | -2.67% | 72.10% | 74.77% (+2.67%) |
| **$L=7$** | 95% | **19.92%** | 17.00% | -2.92% | 75.08% | 78.00% (+2.92%) |

---

## 12. Sharpness Results (Mean Interval Width)

| Lead Time ($L$) | Nominal Coverage | Raw Interval Width | Adaptive Interval Width | Width Reduction ($\Delta$) |
|:---:|:---:|:---:|:---:|:---:|
| **$L=1$** | 80% | 27.90 | **23.90** | -4.00 (-14.3%) |
| **$L=1$** | 90% | 34.37 | **29.49** | -4.88 (-14.2%) |
| **$L=1$** | 95% | 40.35 | **34.60** | -5.75 (-14.2%) |
| **$L=3$** | 90% | 78.76 | **65.70** | -13.06 (-16.6%) |
| **$L=5$** | 90% | 119.89 | **99.74** | -20.15 (-16.8%) |
| **$L=7$** | 90% | 156.50 | **131.61** | -24.89 (-15.9%) |

---

## 13. Interval Violation Analysis

| Lead Time ($L$) | Nominal Coverage | Method | Empirical Coverage | Lower Violation Rate | Upper Violation Rate |
|:---:|:---:|:---|:---:|:---:|:---:|
| **$L=1$** | 90% | Raw Conformal | 29.37% | 35.31% | 35.32% |
| **$L=1$** | 90% | Scale Normalized | 25.54% | 36.81% | 37.64% |
| **$L=3$** | 90% | Raw Conformal | 21.67% | 38.50% | 39.84% |
| **$L=3$** | 90% | Scale Normalized | 18.31% | 39.90% | 41.79% |
| **$L=7$** | 90% | Raw Conformal | 17.90% | 39.82% | 42.29% |
| **$L=7$** | 90% | Scale Normalized | 15.23% | 40.95% | 43.83% |

*   **Observation**: The adaptive method reduced interval width, which caused both lower bound violations and upper bound violations to increase symmetrically by ~1.5 to 2.0 percentage points.

---

## 14. Series-Level Results Across 500 Series

| Lead Time ($L$) | Method | Nominal Cov | Mean Cov | Median Cov | P25 Cov | P75 Cov | Min Cov | Max Cov | % Series < 50% |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$L=1$** | Raw | 90% | 29.37% | 28.57% | 25.21% | 33.99% | 8.24% | 57.97% | **99.4%** |
| **$L=1$** | Adaptive | 90% | 25.54% | 24.73% | 21.70% | 29.12% | 6.04% | 52.75% | **99.8%** |
| **$L=7$** | Raw | 90% | 17.90% | 17.32% | 12.00% | 22.91% | 3.06% | 48.04% | **100.0%** |
| **$L=7$** | Adaptive | 90% | 15.23% | 14.80% | 9.78% | 19.55% | 1.96% | 44.41% | **100.0%** |

---

## 15. Demand-Profile Results (Volume and Seasonality Slices)

### Volume Breakdown ($L=1$, 90% Nominal)
*   **High Volume**: Raw Coverage = 31.88% | Adaptive Coverage = 27.67% (Width: 45.98 $\rightarrow$ 39.54)
*   **Medium Volume**: Raw Coverage = 28.81% | Adaptive Coverage = 25.07% (Width: 24.62 $\rightarrow$ 21.14)
*   **Low Volume**: Raw Coverage = 16.68% | Adaptive Coverage = 14.49% (Width: 17.43 $\rightarrow$ 14.89)

### Seasonality Breakdown ($L=1$, 90% Nominal)
*   **Strong Seasonality**: Raw Coverage = 30.56% | Adaptive Coverage = 26.58% (Width: 36.41 $\rightarrow$ 31.25)
*   **Moderate Seasonality**: Raw Coverage = 20.75% | Adaptive Coverage = 18.04% (Width: 19.67 $\rightarrow$ 16.90)

---

## 16. Statistical Comparison Across 500 Series

Paired non-parametric tests (**Wilcoxon signed-rank test**) across the 500 series on the 2017 test set:

| Lead Time ($L$) | Nominal Cov | Coverage Error Diff ($\text{Adapt} - \text{Raw}$) | $p$-value (Coverage Error) | Width Diff ($\text{Adapt} - \text{Raw}$) | $p$-value (Width) | Winkler Score Diff ($\text{Adapt} - \text{Raw}$) | $p$-value (Score) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$L=1$** | 80% | +0.0309 | $3.56 \times 10^{-63}$ | -4.01 | $1.38 \times 10^{-67}$ | +11.78 | $7.99 \times 10^{-68}$ |
| **$L=1$** | 90% | +0.0382 | $1.10 \times 10^{-63}$ | -4.88 | $4.56 \times 10^{-67}$ | +31.77 | $2.48 \times 10^{-67}$ |
| **$L=1$** | 95% | +0.0465 | $5.96 \times 10^{-65}$ | -5.75 | $8.64 \times 10^{-67}$ | +76.43 | $4.61 \times 10^{-67}$ |
| **$L=3$** | 90% | +0.0336 | $4.04 \times 10^{-55}$ | -13.07 | $7.36 \times 10^{-58}$ | +91.36 | $3.56 \times 10^{-58}$ |
| **$L=7$** | 90% | +0.0267 | $2.08 \times 10^{-42}$ | -24.88 | $1.58 \times 10^{-47}$ | +179.96 | $1.22 \times 10^{-47}$ |

*   **Statistical Significance**: The differences are statistically significant ($p < 10^{-40}$ across all configurations). Adaptive conformal significantly reduces interval width, but increases coverage error and Winkler penalties under test-set scale expansion.

---

## 17. Residual Distribution Stability Analysis Across Years (2013–2016)

| Year | Lead Time ($L$) | Raw MAE | Raw Std | Raw P90 | Norm Mean Score | Norm Std Score | Norm P90 Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2013** | $L=1$ | 9.86 | 7.31 | 20.07 | 1.17 | 0.73 | 2.14 |
| **2014** | $L=1$ | 9.17 | 8.93 | 21.19 | 1.01 | 0.71 | 1.97 |
| **2015** | $L=1$ | 8.46 | 7.84 | 18.31 | 1.02 | 0.72 | 2.00 |
| **2016** | $L=1$ | 9.97 | 9.67 | 22.71 | 1.01 | 0.70 | 1.95 |

*   **Stability Finding**: Normalization successfully achieved **improved residual-score stability** across training-era years (2013–2016), standardizing mean normalized scores to $\approx 1.01–1.02$ and 90th percentile scores to $\approx 1.95–2.00$.

---

## 18. Limitations

1.  **Lagged Scale Estimation**: Historical rolling scale estimation ($\sigma_{t,L}$) inherently lags abrupt forward distribution shifts.
2.  **No Extrapolation of Variance Expansion**: A historical scale estimator cannot anticipate that future unseen test errors will expand by $3.5\times$ unless provided with forward volatility indicators.
3.  **Non-Exchangeability**: Temporal autocorrelation and seasonal regime changes violate the exchangeability requirement of standard conformal calibration.

---

## 19. Scientific Conclusion

*   **Status**: **NOT SUPPORTED** for coverage improvement on the forward out-of-sample test period.
*   **Detailed Assessment**:
    1.  Scale-normalized conformal calibration **improved interval sharpness** (narrowing intervals by 14% to 17%) and improved residual-score stability across historical training folds.
    2.  However, because trailing historical scale estimates lag the severe 2017 variance expansion, narrower intervals led to **lower empirical coverage** (dropping from ~29.4% to ~25.5% at 90% nominal) and higher Winkler loss.
    3.  Therefore, simple trailing scale normalization is insufficient to overcome distribution shifts in non-stationary retail environments without forward volatility forecasting or conservative adaptive margins.
