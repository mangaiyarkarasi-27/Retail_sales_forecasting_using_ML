# Adaptive Model Routing Methodology

This document details the technical design, feature representation, out-of-fold label generation, and classification architecture of the meta-router in the retail forecasting pipeline.

---

## 1. Motivation

In heterogeneous retail networks, a single global forecasting model can be sub-optimal across different product demand structures. The meta-router acts as a data-driven model selector that maps the demand profile of a given series to the most appropriate forecasting expert, seeking to balance prediction variance and bias.

---

## 2. Router Inputs

The router uses only static demand-profile features computed strictly during the training partition:
* `mean_sales` (Daily volume scale)
* `cv_sales` (Historical demand volatility)
* `trend_strength_r2` (Proportion of variance explained by a linear trend)
* `lag365_autocorr` (Strength of annual seasonality)

*Note*: No future realized sales, test-period rolling statistics, or test-period errors are used, ensuring zero future-target leakage.

---

## 3. Leakage Prevention

To prevent look-ahead bias:
1. All demand-profile features are computed on the training split prior to evaluation.
2. The router's training labels are generated using out-of-fold predictions on a 5-fold TimeSeriesSplit.
3. The test set is untouched during the entire router design, selection, and training process, and evaluated only once at the final step.

---

## 4. Out-of-Fold Label Generation

To construct the training dataset for the router:
1. The training partition (first 80% chronologically) is split into 5 chronological validation folds using `TimeSeriesSplit(n_splits=5)`.
2. In each fold, the XGBoost and Random Forest experts are trained on the fold's sub-train partition.
3. Predictions are made on the fold's sub-validation partition.
4. For each store-item combination, the Mean Absolute Error (MAE) is calculated.
5. The winning expert is defined as:
   $$\text{Winner} = \begin{cases} 1 & \text{if } \text{MAE}_{\text{XGBoost}} \le \text{MAE}_{\text{Random Forest}} \\ 0 & \text{otherwise} \end{cases}$$
6. These records are compiled into [`oof_router_training_data.csv`](file:///c:/Users/rmang/Documents/retail_project/journal_extension/results/model_selection/oof_router_training_data.csv).

---

## 5. Router Architecture

The router uses a **Decision Tree Classifier** with:
* `max_depth = 3` (to guarantee high interpretability and prevent overfitting).
* `class_weight = 'balanced'` (to account for the class imbalance since XGBoost wins $84.2\%$ of the series).

---

## 6. Training & Validation Procedures

* **Training**: Fitted on the merged out-of-fold data and static training-period demand profiles.
* **Validation**: Evaluated using the sub-validation split of the training partition.
* **Test**: Evaluated once on the chronological test split.

---

## 7. Evaluation Metrics

* **Routing Metrics**: Routing accuracy, balanced accuracy, macro F1, and average regret.
* **Forecasting Metrics**: Aggregate MAE and RMSE obtained by executing the router's chosen expert predictions.

---

## 8. Interpretability

Decision trees are inherently interpretable. We compute and export feature importances (Gini importance) to trace which demand characteristics (e.g., volume vs. seasonality) govern the choice between the experts.

---

## 9. Limitations

* **Imbalance Sensitivity**: Since XGBoost is highly dominant, minor routing errors on XGBoost-optimal series can outweigh the accuracy improvements gained on Random Forest-optimal series.
* **Static Features**: The router does not adapt to dynamic within-year regime changes (e.g. sudden stockout-induced demand shifts).
