"""
inventory_policy.py
-------------------
Defines inventory replenishment policies for the decision-support layer.
1. DeterministicPointForecastPolicy: ROP = Dhat_{t,L}, Safety Stock = 0
2. ConformalUncertaintyPolicy: ROP = Dhat_{t,L} + SS_{t,L}, Safety Stock = Upper_{t,L} - Dhat_{t,L}
"""

from typing import Dict, Any


class InventoryPolicy:
    """Base class for replenishment policies."""
    
    def __init__(self, name: str):
        self.name = name

    def compute_rop(self, point_forecast: float, uncertainty_margin: float = 0.0) -> float:
        raise NotImplementedError


class DeterministicPointForecastPolicy(InventoryPolicy):
    """
    Deterministic Point-Forecast Policy (Policy A):
    ROP_t = Dhat_{t,L}
    Safety Stock SS_{t,L} = 0.0
    """
    
    def __init__(self):
        super().__init__(name="Deterministic_Point_Forecast")

    def compute_rop(self, point_forecast: float, uncertainty_margin: float = 0.0) -> float:
        return max(0.0, float(point_forecast))


class ConformalUncertaintyPolicy(InventoryPolicy):
    """
    Uncertainty-Aware Conformal Policy (Policy B):
    ROP_t = Dhat_{t,L} + SS_{t,L}
    Safety Stock SS_{t,L} = uncertainty_margin = Upper_{t,L} - Dhat_{t,L}
    """
    
    def __init__(self, nominal_coverage: float = 0.90):
        super().__init__(name=f"Conformal_Uncertainty_{int(nominal_coverage*100)}")
        self.nominal_coverage = nominal_coverage

    def compute_rop(self, point_forecast: float, uncertainty_margin: float = 0.0) -> float:
        ss = max(0.0, float(uncertainty_margin))
        return max(0.0, float(point_forecast) + ss)
