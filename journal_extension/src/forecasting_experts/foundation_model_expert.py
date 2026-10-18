"""
foundation_model_expert.py
--------------------------
This module implements the time-series foundation model expert.
Under current CPU-only, torch-less VM constraints, it records the 
unfeasibility status and raises a clear environment error upon instantiation.
"""

import logging

logger = logging.getLogger(__name__)


class FoundationModelExpert:
    """
    Placeholder class for a time-series foundation model (e.g. Chronos).
    Currently unfeasible in this environment due to missing PyTorch and 
    excessive CPU latency constraints.
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-tiny"):
        self.model_name = model_name
        self.is_feasible = False
        logger.error(
            f"FoundationModelExpert: Selected model '{self.model_name}' is not feasible "
            f"in the current environment due to missing PyTorch/Transformers "
            f"and CPU compute limits."
        )

    def forecast(self, history, horizon: int):
        """
        Generates predictions. Raises error due to unfeasibility.
        """
        raise RuntimeError(
            "FoundationModelExpert cannot generate forecasts in the current environment. "
            "Refer to docs/foundation_model_feasibility.md for details."
        )
