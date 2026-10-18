"""
dashboard/series_service.py
---------------------------
Data and inference service layer for E-IDSS.
Provides cached access to series history, direct multi-horizon forecasting,
SBC demand profiling, decision scoring, conformal uncertainty intervals,
and discrete-event inventory simulation outcomes for any Store-Item combination.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Add forecasting module to path
def _ensure_paths(journal_root: str):
    forecasting_src = os.path.join(journal_root, "src", "forecasting")
    if forecasting_src not in sys.path:
        sys.path.insert(0, forecasting_src)


@st.cache_data(show_spinner=False)
def load_full_dataset(project_root: str) -> pd.DataFrame:
    """Loads and caches the 913,000-row preprocessed store item demand dataset."""
    data_path = os.path.join(project_root, "data", "processed_train.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_series_history(project_root: str, store: int, item: int) -> pd.DataFrame:
    """Extracts the complete historical daily sales for a specific Store and Item."""
    df = load_full_dataset(project_root)
    if not df.empty:
        sub = df[(df["store"] == store) & (df["item"] == item)].sort_values("date").reset_index(drop=True)
        return sub
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def get_forecast_orchestrator(journal_root: str):
    """Loads and caches the 7 Direct Multi-Horizon XGBoost models."""
    _ensure_paths(journal_root)
    try:
        from forecast_vector import ForecastVectorOrchestrator
        models_dir = os.path.join(journal_root, "results", "forecasting_models")
        orchestrator = ForecastVectorOrchestrator(models_dir=models_dir)
        return orchestrator
    except Exception as e:
        print(f"Error loading ForecastVectorOrchestrator: {e}")
        return None


def generate_multihorizon_forecast(journal_root: str, project_root: str, store: int, item: int) -> pd.DataFrame:
    """
    Generates genuine live multi-horizon forecasts [y_t+1 ... y_t+7] using
    the 7 decoupled XGBoost models M_1 ... M_7 for the selected Store & Item.
    """
    orchestrator = get_forecast_orchestrator(journal_root)
    sub = get_series_history(project_root, store, item)
    
    if orchestrator is None or sub.empty or len(sub) < 60:
        return pd.DataFrame()

    sub_tail = sub.tail(60).copy()
    last_date = sub_tail["date"].iloc[-1]
    
    results = []
    cumulative_demand = 0.0
    
    for h in range(1, 8):
        target_date = last_date + pd.Timedelta(days=h)
        try:
            preds = orchestrator.predict_horizon_batch(sub_tail, h)
            pred_val = max(0.0, float(preds[-1]))
        except Exception:
            pred_val = float(sub_tail["sales"].mean())
            
        cumulative_demand += pred_val
        results.append({
            "horizon": h,
            "target_date": target_date,
            "target_day_name": target_date.strftime("%A"),
            "predicted_sales": round(pred_val, 2),
            "cumulative_demand": round(cumulative_demand, 2)
        })
        
    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def get_series_profile(journal_root: str, store: int, item: int) -> dict:
    """Retrieves Syntetos-Boylan-Croston (SBC) demand categorization and dynamics."""
    csv_path = os.path.join(journal_root, "results", "model_selection", "demand_profiles.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df[(df["store"] == store) & (df["item"] == item)]
        if not row.empty:
            return row.iloc[0].to_dict()
    return {}


@st.cache_data(show_spinner=False)
def get_series_decision_record(project_root: str, store: int, item: int) -> dict:
    """Retrieves MCDA Decision Score, Risk Score, and SHAP explanations."""
    csv_path = os.path.join(project_root, "results", "decision_intelligence", "priority_ranking.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df[(df["Store"] == store) & (df["Item"] == item)]
        if not row.empty:
            return row.iloc[0].to_dict()
    return {}


@st.cache_data(show_spinner=False)
def get_series_uncertainty_metrics(journal_root: str, store: int, item: int, lead_time: int = 1, nominal: float = 0.90) -> dict:
    """Retrieves series-level conformal coverage, interval width, and interval score."""
    csv_path = os.path.join(journal_root, "results", "uncertainty", "final_2017", "per_series_uncertainty_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df[(df["store"] == store) & (df["item"] == item) & (df["lead_time"] == lead_time) & (np.isclose(df["nominal_coverage"], nominal))]
        if not row.empty:
            return row.iloc[0].to_dict()
    return {}


@st.cache_data(show_spinner=False)
def get_series_simulation_results(journal_root: str, store: int, item: int, lead_time: int = 1, shortage_penalty: float = 1.0) -> dict:
    """
    Retrieves the exact discrete-event replenishment simulation outcome for the series,
    comparing Deterministic Policy A vs Conformal Uncertainty Policy B.
    """
    csv_path = os.path.join(journal_root, "results", "inventory", "exp_019", "per_series_results.csv")
    if not os.path.exists(csv_path):
        return {}
        
    df = pd.read_csv(csv_path)
    sub = df[(df["store"] == store) & (df["item"] == item) & (df["lead_time"] == lead_time) & (np.isclose(df["shortage_penalty"], shortage_penalty))]
    
    det_row = sub[sub["policy"] == "Deterministic"]
    conf_row = sub[sub["policy"] == "Conformal_Uncertainty"]
    
    if det_row.empty or conf_row.empty:
        return {}
        
    det = det_row.iloc[0]
    conf = conf_row.iloc[0]
    
    delta_fill_rate = conf["fill_rate"] - det["fill_rate"]
    units_prevented = det["shortage_units"] - conf["shortage_units"]
    cost_delta = conf["total_cost"] - det["total_cost"]
    
    return {
        "lead_time": lead_time,
        "shortage_penalty": shortage_penalty,
        "det_fill_rate": det["fill_rate"],
        "conf_fill_rate": conf["fill_rate"],
        "delta_fill_rate_pp": delta_fill_rate * 100.0,
        "det_stockout_rate": det["stockout_rate"],
        "conf_stockout_rate": conf["stockout_rate"],
        "det_shortage_units": det["shortage_units"],
        "conf_shortage_units": conf["shortage_units"],
        "units_prevented": units_prevented,
        "det_average_inventory": det["average_inventory"],
        "conf_average_inventory": conf["average_inventory"],
        "det_total_cost": det["total_cost"],
        "conf_total_cost": conf["total_cost"],
        "cost_delta": cost_delta,
        "det_holding_cost": det["holding_cost"],
        "conf_holding_cost": conf["holding_cost"],
        "det_shortage_cost": det["shortage_cost"],
        "conf_shortage_cost": conf["shortage_cost"],
        "det_ordering_cost": det["ordering_cost"],
        "conf_ordering_cost": conf["ordering_cost"],
        "total_demand": det["total_demand"]
    }
