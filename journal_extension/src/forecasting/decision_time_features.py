"""
decision_time_features.py
--------------------------
Generates leakage-safe forecasting features at decision origin t
using only observations strictly preceding t (i.e. <= t-1).
"""

import pandas as pd
import numpy as np


def generate_origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the base features (lags and rolling statistics) for each store-item group
    based on the state of the series up to yesterday (origin t, using sales <= t-1).
    Note: The returned dataframe will contain base features aligned to the date of the origin t.
    """
    df_feat = df.copy()
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat = df_feat.sort_values(by=["store", "item", "date"]).reset_index(drop=True)

    # 1. Lags terminating at t-1 (relative to origin t, so sales_lag_K is shifted by K days)
    # Since the row date is the origin t, sales_lag_7 represents sales at t-7, etc.
    df_feat["sales_lag_7"] = df_feat.groupby(["store", "item"])["sales"].shift(7)
    df_feat["sales_lag_14"] = df_feat.groupby(["store", "item"])["sales"].shift(14)
    df_feat["sales_lag_30"] = df_feat.groupby(["store", "item"])["sales"].shift(30)

    # 2. Rolling means terminating strictly at t-1
    # For a row representing origin t, shift(1) means the window covers [t-7, t-1] or [t-30, t-1]
    df_feat["rolling_mean_7"] = df_feat.groupby(["store", "item"])["sales"].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )
    df_feat["rolling_mean_30"] = df_feat.groupby(["store", "item"])["sales"].transform(
        lambda x: x.shift(1).rolling(window=30).mean()
    )

    return df_feat


def construct_target_calendar_features(origin_dates: pd.Series, h: int) -> pd.DataFrame:
    """
    Constructs calendar features for the target date t+h, which are deterministic
    and known in advance at origin t.
    """
    target_dates = pd.to_datetime(origin_dates) + pd.Timedelta(days=h)
    
    cal_df = pd.DataFrame(index=origin_dates.index)
    cal_df["year"] = target_dates.dt.year
    cal_df["month"] = target_dates.dt.month
    cal_df["day"] = target_dates.dt.day
    cal_df["weekday"] = target_dates.dt.weekday
    cal_df["weekofyear"] = target_dates.dt.isocalendar().week.astype(int)
    cal_df["quarter"] = target_dates.dt.quarter
    cal_df["is_weekend"] = (cal_df["weekday"] >= 5).astype(int)
    cal_df["month_start"] = target_dates.dt.is_month_start.astype(int)
    cal_df["month_end"] = target_dates.dt.is_month_end.astype(int)
    
    return cal_df
