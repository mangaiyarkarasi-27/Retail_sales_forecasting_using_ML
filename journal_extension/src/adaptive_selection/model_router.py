"""
model_router.py
--------------
This module implements the Meta-Router Classifier.
It trains a simple, interpretable model to predict the best forecasting expert 
(XGBoost vs Random Forest) based on static demand profiles.
"""

import os
import pickle
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Learns to route store-item series to the optimal forecasting expert.
    """

    def __init__(self, max_depth: int = 3, random_state: int = 42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.classifier = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight="balanced"
        )
        self.features_list = [
            "mean_sales",
            "cv_sales",
            "trend_strength_r2",
            "lag365_autocorr"
        ]
        self.model_map = {0: "Random_Forest", 1: "XGBoost"}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the routing classifier.
        """
        logger.info(f"Training ModelRouter (DecisionTreeClassifier, max_depth={self.max_depth})...")
        self.classifier.fit(X[self.features_list], y)

    def predict_best_expert(self, features_df: pd.DataFrame) -> List[str]:
        """
        Predicts which model is expected to win for each row in the features DataFrame.
        """
        preds = self.classifier.predict(features_df[self.features_list])
        return [self.model_map[p] for p in preds]

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importances for interpretability.
        """
        importances = self.classifier.feature_importances_
        return dict(zip(self.features_list, importances))

    def save_router(self, path: str = "results/models/model_router.pkl") -> None:
        """
        Serializes the trained router.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved model router weights to {path}")

    @classmethod
    def load_router(cls, path: str = "results/models/model_router.pkl"):
        """
        Deserializes the trained router.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Router file not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
