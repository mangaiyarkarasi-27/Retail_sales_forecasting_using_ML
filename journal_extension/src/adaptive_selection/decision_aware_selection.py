"""
decision_aware_selection.py
--------------------------
This module implements decision-aware evaluation structures
to evaluate forecasting model parameters using downstream inventory costs.
"""

import numpy as np

class DecisionAwareScorer:
    """
    Computes asymmetric downstream cost losses for model selection.
    """
    def __init__(self, holding_cost: float = 0.20, shortage_cost: float = 1.00):
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost

    def calculate_cost_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates asymmetric cost loss:
        - If y_pred > y_true: penalty is holding_cost * (y_pred - y_true)
        - If y_pred < y_true: penalty is shortage_cost * (y_true - y_pred)
        """
        residuals = y_pred - y_true
        holding_penalties = np.where(residuals > 0, residuals * self.holding_cost, 0.0)
        shortage_penalties = np.where(residuals < 0, np.abs(residuals) * self.shortage_cost, 0.0)
        return float(np.mean(holding_penalties + shortage_penalties))
