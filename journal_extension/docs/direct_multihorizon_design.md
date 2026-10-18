# Direct Multi-Horizon Forecasting Design

This document details the formal training design, feature definitions, and information boundaries of the Direct Multi-Horizon point-forecasting layer for the journal extension.

---

## 1. Forecast Origin

The forecast origin $t$ is defined as the morning of day $t$ immediately before today's actual sales demand $y_t$ is observed. Consequently, the latest observed demand value is $y_{t-1}$ (yesterday's sales).

---

## 2. Decision-Time Information Set

At forecast origin $t$, the decision-time information set $\mathcal{I}_t$ comprises:
$$\mathcal{I}_t = \{ y_k \mid k \le t-1 \} \cup \{ \text{Calendar}_k \mid \forall k \} \cup \{ \hat{y}_k \mid k \le t-1 \}$$
where $y_k$ represent actual observed sales, $\text{Calendar}_k$ represents deterministic calendar metadata, and $\hat{y}_k$ represent past predictions. Today's sales $y_t$ and all subsequent future sales $y_{t+1}, \dots$ are strictly outside the information set.

---

## 3. Horizon-Specific Models

For each prediction horizon $h \in \{1, 2, 3, 4, 5, 6, 7\}$ days, we define a decoupled model $M_h$:
$$M_h : \mathcal{I}_t \to y_{t+h}$$
Every model $M_h$ generates predictions for its target day $t+h$ using the **same** forecast origin $t$ (morning of $t$).

---

## 4. Safe Feature Construction

To prevent future-demand leakage, all predictors used as inputs for $M_h$ must be calculated strictly from the information set $\mathcal{I}_t$ (i.e., using sales observations up to $y_{t-1}$). The safe feature definitions are:

| Feature Name | Definition at Origin $t$ | Latest Observation Date | Safe? |
|:---|:---|:---:|:---:|
| `store`, `item` | Static categorical codes | N/A | **YES** |
| `year`, `month`, `day` | Calendar components of target date $t+h$ | N/A (Deterministic) | **YES** |
| `weekday`, `weekofyear`, `quarter` | Calendar components of target date $t+h$ | N/A (Deterministic) | **YES** |
| `is_weekend`, `month_start`, `month_end` | Calendar indicators of target date $t+h$ | N/A (Deterministic) | **YES** |
| `sales_lag_7` | $y_{t-7}$ | $t-7$ | **YES** |
| `sales_lag_14` | $y_{t-14}$ | $t-14$ | **YES** |
| `sales_lag_30` | $y_{t-30}$ | $t-30$ | **YES** |
| `rolling_mean_7` | $\text{mean}(y_{t-7} \dots y_{t-1})$ | $t-1$ | **YES** |
| `rolling_mean_30` | $\text{mean}(y_{t-30} \dots y_{t-1})$ | $t-1$ | **YES** |

*Key Correction*: All rolling features are computed over windows that end strictly at day $t-1$, which corresponds to yesterday's sales. No today or future rolling statistics enter the feature vector.

---

## 5. Training Row Definition

For horizon $h$, a single training example consists of features $X_t$ and target $y_{t+h}$:
$$\text{Training Row} = \big( X_t, y_{t+h} \big)$$
where:
* $X_t$ is the feature vector computed strictly from observations $\le t-1$ and calendar metadata for target date $t+h$.
* $y_{t+h}$ is the target sales at day $t+h$.
* *Verification*: No target value in $X_t$ uses observations beyond $t-1$, guaranteeing that $y_{t+h}$ never contaminates the predictors.

---

## 6. Calendar Features

Calendar information is provided corresponding to the **target date $t+h$**.
* *Rationale*: Supply chain demand exhibits heavy seasonality (e.g., weekend spikes, month-end purchasing). The model must know whether the target date $t+h$ falls on a weekend or month-end. Because calendar dates are deterministic, this information is known in advance and does not cause target leakage.

---

## 7. Global vs. Series-Specific Models

We preserve the **Global Model** philosophy from the conference baseline:
* **Proposed Architecture**: $7$ global XGBoost models, one for each horizon $h \in \{1 \dots 7\}$, trained across all 500 store-item combinations.
* *Advantages*:
  * **Statistical Strength**: Reuses cross-series demand information (e.g. general seasonality) and prevents overfitting.
  * **Computational Simplicity**: We fit exactly 7 models rather than 3,500 models (7 horizons $\times$ 500 series).

---

## 8. Training/Validation Boundaries

To prevent target leakage into training rows:
* **Training origins**: 2013-01-01 to 2015-12-31. For horizon $h$, the training targets span up to 2015-12-31 $+ h$.
* **Validation origins**: 2016-01-01 to 2016-12-31. Used for out-of-sample model validation and parameter freezing.
* **Final test origins**: 2017-01-06 to 2017-12-31. Untouched during design and parameter selection.

---

## 9. Recursive vs. Direct Methodological Comparison

| Dimension | Recursive Design | Direct Design (Recommended) |
|:---|:---|:---|
| **Leakage Safety** | Safe | Safe |
| **Error Propagation** | High (predictions recursively used as features) | None (models are decoupled) |
| **Implementation** | Very High (requires recursive state loop) | Low (adjust pandas feature shift by horizon) |
| **Model Count** | 1 global model | 7 global models |
| **Computational Burden** | Low training, High prediction inference | Moderate training, Low prediction inference |
| **Feature Compatibility** | Requires modifying pandas index on-the-fly | Preserved (uses standard shifts) |
| **Reproducibility** | Moderate | High |

---

## 10. Forecast Vector

For any decision origin $t$, the forecasting layer outputs a complete point-forecast vector:
$$\text{forecast\_vector}_t = \big[ \hat{y}_{t+1}, \hat{y}_{t+2}, \dots, \hat{y}_{t+7} \big]$$
The downstream inventory system computes lead-time demand by summing components of this vector:
$$\hat{D}_{t, L} = \sum_{k=1}^L \hat{y}_{t+k}$$
The inventory simulator does not construct features or generate forecasts itself.

---

## 11. Conformal Integration

For prediction origin $t$ and lead time $L$:
* The cumulative demand forecast is $\hat{D}_{t, L}$.
* The true realized demand is $D_{t, L} = \sum_{k=1}^L y_{t+k}$.
* The residual is $E_{t, L} = |D_{t, L} - \hat{D}_{t, L}|$.
* **Temporal Eligibility**: At decision time $s$, the residual originating at $r$ is eligible for inclusion in the calibration window if and only if $r + L < s$.

---

## 12. Leakage Prevention

* **Target-Date Separator**: The model $M_h$ predicts $y_{t+h}$ using $X_t$ containing demand up to $y_{t-1}$.
* **Calendar Restrictions**: Calendar features are restricted to deterministic variables.

---

## 13. Computational Considerations

Since we train global models, the total model count is **exactly 7**. Training 7 global XGBoost models on ~700,000 rows takes less than 15 seconds in total, ensuring low computational overhead.

---

## 14. Limitations

* **Model Decoupling**: Direct models do not capture cross-horizon correlations (e.g. predicting $t+2$ does not depend on predictions at $t+1$).
* **Horizon Limits**: Predicting beyond $h=7$ requires training additional models.

---

## 15. Final Design Decision

We select **Direct Multi-Horizon XGBoost** with 7 global models. This architecture eliminates recursive error propagation, aligns perfectly with standard features, and guarantees zero future-demand leakage.
