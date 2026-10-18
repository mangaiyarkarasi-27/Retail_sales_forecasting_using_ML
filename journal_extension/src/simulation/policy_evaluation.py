"""
policy_evaluation.py
--------------------
Orchestrates portfolio-level inventory policy evaluation across 500 store-item series.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
try:
    from journal_extension.src.simulation.inventory_simulator import InventorySimulator
except ImportError:
    from inventory_simulator import InventorySimulator


def evaluate_portfolio_policy(
    series_dict: Dict[tuple, Dict[str, Any]],
    lead_time: int,
    policy_name: str,
    nominal_coverage: float,
    shortage_penalty: float = 1.00,
    order_cost: float = 10.0,
    holding_cost_rate: float = 0.20,
    burn_in_days: int = 0
) -> Dict[str, Any]:
    """
    Evaluates a specific replenishment policy across all 500 series in series_dict.
    
    Parameters:
    -----------
    series_dict: mapping (store, item) -> {
        "dates": np.ndarray,
        "sales": np.ndarray,
        "point_forecasts": np.ndarray, # Dhat_{t,L}
        "uncertainty_margins": {cov: np.ndarray}, # SS_{t,L}(cov)
        "pre_2017_mean": float # mu_i
    }
    lead_time: L in {1, 3, 5, 7}
    policy_name: "Deterministic" or "Conformal_Uncertainty"
    nominal_coverage: e.g. 0.90
    shortage_penalty: p in {0.60, 1.00, 2.00, 4.00}
    """
    simulator = InventorySimulator(
        order_cost=order_cost,
        holding_cost_rate=holding_cost_rate,
        shortage_penalty=shortage_penalty
    )

    if policy_name == "Deterministic":
        policy_func = lambda pt_f, u_m: max(0.0, float(pt_f))
    elif policy_name == "Conformal_Uncertainty":
        policy_func = lambda pt_f, u_m: max(0.0, float(pt_f) + max(0.0, float(u_m)))
    else:
        raise ValueError(f"Unknown policy_name: {policy_name}")

    per_series_records = []
    sample_daily_df = None

    for (store, item), s_data in series_dict.items():
        dates = s_data["dates"]
        sales = s_data["sales"]
        pt_f = s_data["point_forecasts"]
        u_margins = s_data["uncertainty_margins"].get(nominal_coverage, np.zeros(len(sales)))
        init_oh = lead_time * s_data["pre_2017_mean"]

        sim_res = simulator.simulate_series(
            dates=dates,
            actual_sales=sales,
            point_forecasts=pt_f,
            uncertainty_margins=u_margins,
            lead_time=lead_time,
            initial_on_hand=init_oh,
            policy_func=policy_func,
            burn_in_days=burn_in_days,
            end_buffer_days=lead_time
        )

        res_summary = sim_res["summary"]
        res_summary["store"] = store
        res_summary["item"] = item
        res_summary["lead_time"] = lead_time
        res_summary["policy"] = policy_name
        res_summary["nominal_coverage"] = nominal_coverage if policy_name == "Conformal_Uncertainty" else 0.0
        res_summary["shortage_penalty"] = shortage_penalty
        
        per_series_records.append(res_summary)

        if sample_daily_df is None and store == 1 and item == 1:
            sample_daily_df = sim_res["eval_df"].copy()
            sample_daily_df["store"] = store
            sample_daily_df["item"] = item
            sample_daily_df["policy"] = policy_name
            sample_daily_df["lead_time"] = lead_time
            sample_daily_df["nominal_coverage"] = nominal_coverage

    per_series_df = pd.DataFrame(per_series_records)

    # Compute portfolio-level aggregates
    portfolio_summary = {
        "policy": policy_name,
        "lead_time": lead_time,
        "nominal_coverage": nominal_coverage if policy_name == "Conformal_Uncertainty" else 0.0,
        "shortage_penalty": shortage_penalty,
        "total_demand": float(per_series_df["total_demand"].sum()),
        "total_fulfilled": float(per_series_df["total_fulfilled"].sum()),
        "shortage_units": float(per_series_df["shortage_units"].sum()),
        "fill_rate": float(per_series_df["total_fulfilled"].sum() / per_series_df["total_demand"].sum()),
        "mean_fill_rate_across_series": float(per_series_df["fill_rate"].mean()),
        "stockout_rate": float(per_series_df["stockout_rate"].mean()),
        "service_level": float(per_series_df["service_level"].mean()),
        "mean_ending_inventory": float(per_series_df["mean_ending_inventory"].mean()),
        "average_inventory": float(per_series_df["average_inventory"].mean()),
        "total_orders": int(per_series_df["number_of_orders"].sum()),
        "total_ordered_qty": float(per_series_df["total_ordered_qty"].sum()),
        "average_order_qty": float(per_series_df["average_order_qty"].mean()),
        "ordering_cost": float(per_series_df["ordering_cost"].sum()),
        "holding_cost": float(per_series_df["holding_cost"].sum()),
        "shortage_cost": float(per_series_df["shortage_cost"].sum()),
        "total_cost": float(per_series_df["total_cost"].sum()),
        "cost_per_demanded_unit": float(per_series_df["total_cost"].sum() / per_series_df["total_demand"].sum()),
        "cost_per_fulfilled_unit": float(per_series_df["total_cost"].sum() / per_series_df["total_fulfilled"].sum())
    }

    return {
        "portfolio_summary": portfolio_summary,
        "per_series_df": per_series_df,
        "sample_daily_df": sample_daily_df
    }
