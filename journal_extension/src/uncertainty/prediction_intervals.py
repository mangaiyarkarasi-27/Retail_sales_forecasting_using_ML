"""
prediction_intervals.py
-----------------------
Implements structures for generating and handling prediction intervals.
"""

import numpy as np

class PredictionInterval:
    """
    Represents a time-series prediction interval containing point forecast,
    lower bound, upper bound, and interval width.
    """
    def __init__(self, point_forecast: float, margin: float, clip_zero: bool = True):
        self.point_forecast = point_forecast
        self.margin = margin
        self.lower = point_forecast - margin
        if clip_zero:
            self.lower = max(0.0, self.lower)
        self.upper = point_forecast + margin
        self.width = self.upper - self.lower

    def contains(self, actual: float) -> bool:
        """
        Checks if the actual value falls within the prediction interval.
        """
        return self.lower <= actual <= self.upper
