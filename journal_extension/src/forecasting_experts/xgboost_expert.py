"""
xgboost_expert.py
-----------------
Wrapper module around the verified conference XGBoost model weights.
Loads the serialized best model to generate forecasts in the journal extension.
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class XGBoostExpert:
    """
    XGBoost Forecasting Expert wrapping the verified best_model_v2.pkl weights.
    """

    def __init__(self, model_path: str = "results/models/best_model_v2.pkl"):
        self.model_path = os.path.normpath(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Serialized best model not found at {self.model_path}")
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def forecast(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the loaded XGBoost model.
        """
        return self.model.predict(X)
