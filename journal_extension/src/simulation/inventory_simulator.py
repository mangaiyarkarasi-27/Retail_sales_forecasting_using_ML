"""
inventory_simulator.py
-----------------------
Implements the daily event-driven inventory simulation engine for retail replenishment.
Strictly follows the 8-step daily event order with Lost Sales and deterministic lead time L.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class InventorySimulator:
    """
    Daily event-driven inventory simulator.
    """

    def __init__(
        self,
        order_cost: float = 10.0,
        holding_cost_rate: float = 0.20,
        shortage_penalty: float = 1.00
    ):
        self.order_cost = order_cost
        self.holding_cost_rate = holding_cost_rate
        self.shortage_penalty = shortage_penalty

    def simulate_series(
        self,
        dates: np.ndarray,
        actual_sales: np.ndarray,
        point_forecasts: np.ndarray,
        uncertainty_margins: np.ndarray,
        lead_time: int,
        initial_on_hand: float,
        policy_func,
        burn_in_days: int = 0,
        end_buffer_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Simulates a single store-item time series over the evaluation period.
        
        Parameters:
        -----------
        dates: array of date objects/strings
        actual_sales: daily observed demand y_t
        point_forecasts: Dhat_{t,L} (cumulative point forecast over lead time L)
        uncertainty_margins: SS_{t,L} (conformal safety margin for lead time L)
        lead_time: deterministic delivery lead time L (days)
        initial_on_hand: initial stock OH_1 = L * mu_i
        policy_func: callable(point_f, margin) -> ROP
        burn_in_days: number of days to discard at start (default 0)
        end_buffer_days: number of days to discard at end (default L)
        """
        n_days = len(actual_sales)
        if end_buffer_days is None:
            end_buffer_days = lead_time

        on_hand = float(initial_on_hand)
        on_order = 0.0
        
        # Pipeline tracking: map arrival_day_index -> arriving_quantity
        pipeline: Dict[int, float] = {}

        daily_records = []

        for t in range(n_days):
            # -------------------------------------------------------------
            # Step 1: Receive arrivals whose lead time has elapsed today
            # -------------------------------------------------------------
            arriving_qty = pipeline.pop(t, 0.0)
            if arriving_qty > 0.0:
                on_hand += arriving_qty
                on_order = max(0.0, on_order - arriving_qty)

            # -------------------------------------------------------------
            # Step 2: Update inventory position
            # -------------------------------------------------------------
            inventory_position = on_hand + on_order

            # -------------------------------------------------------------
            # Step 3: Generate forecast / ROP
            # -------------------------------------------------------------
            pt_f = point_forecasts[t] if t < len(point_forecasts) else 0.0
            u_margin = uncertainty_margins[t] if t < len(uncertainty_margins) else 0.0
            rop = policy_func(pt_f, u_margin)

            # -------------------------------------------------------------
            # Step 4 & 5: Replenishment decision & record order
            # -------------------------------------------------------------
            if inventory_position < rop:
                order_qty = rop - inventory_position
                order_placed = 1
                cost_order = self.order_cost
                # Schedule arrival at day t + lead_time
                arrival_day = t + lead_time
                pipeline[arrival_day] = pipeline.get(arrival_day, 0.0) + order_qty
                on_order += order_qty
                inventory_position = on_hand + on_order
            else:
                order_qty = 0.0
                order_placed = 0
                cost_order = 0.0

            # -------------------------------------------------------------
            # Step 6 & 7: Demand occurs & Fulfill / Shortage (Lost Sales)
            # -------------------------------------------------------------
            demand = float(actual_sales[t])
            if demand <= on_hand:
                fulfilled = demand
                shortage = 0.0
                on_hand -= demand
            else:
                fulfilled = on_hand
                shortage = demand - on_hand
                on_hand = 0.0  # Lost sales, inventory drops to zero

            # -------------------------------------------------------------
            # Step 8: Close day & compute costs
            # -------------------------------------------------------------
            cost_hold = self.holding_cost_rate * on_hand
            cost_short = self.shortage_penalty * shortage
            total_cost = cost_order + cost_hold + cost_short

            daily_records.append({
                "day_index": t,
                "date": dates[t],
                "on_hand": on_hand,
                "on_order": on_order,
                "inventory_position": on_hand + on_order,
                "rop": rop,
                "order_qty": order_qty,
                "order_placed": order_placed,
                "demand": demand,
                "fulfilled_demand": fulfilled,
                "shortage_units": shortage,
                "is_stockout_day": 1 if shortage > 0.0 else 0,
                "order_cost": cost_order,
                "holding_cost": cost_hold,
                "shortage_cost": cost_short,
                "total_cost": total_cost
            })

        daily_df = pd.DataFrame(daily_records)

        # Evaluation window excluding burn-in and end-buffer
        eval_start = burn_in_days
        eval_end = n_days - end_buffer_days if end_buffer_days > 0 else n_days
        eval_df = daily_df.iloc[eval_start:eval_end].copy()

        # Compute summary metrics over valid evaluation window
        total_demand = float(eval_df["demand"].sum())
        total_fulfilled = float(eval_df["fulfilled_demand"].sum())
        total_shortage = float(eval_df["shortage_units"].sum())
        n_eval_days = len(eval_df)
        stockout_days = int(eval_df["is_stockout_day"].sum())
        no_shortage_days = n_eval_days - stockout_days

        fill_rate = total_fulfilled / total_demand if total_demand > 0 else 1.0
        stockout_rate = stockout_days / n_eval_days if n_eval_days > 0 else 0.0
        service_level = no_shortage_days / n_eval_days if n_eval_days > 0 else 1.0

        mean_ending_inventory = float(eval_df["on_hand"].mean())
        median_ending_inventory = float(eval_df["on_hand"].median())
        max_inventory = float(eval_df["on_hand"].max())
        total_orders = int(eval_df["order_placed"].sum())
        total_ordered_qty = float(eval_df["order_qty"].sum())
        avg_order_qty = total_ordered_qty / total_orders if total_orders > 0 else 0.0

        tot_order_cost = float(eval_df["order_cost"].sum())
        tot_hold_cost = float(eval_df["holding_cost"].sum())
        tot_short_cost = float(eval_df["shortage_cost"].sum())
        tot_cost = tot_order_cost + tot_hold_cost + tot_short_cost

        cost_per_demanded_unit = tot_cost / total_demand if total_demand > 0 else 0.0
        cost_per_fulfilled_unit = tot_cost / total_fulfilled if total_fulfilled > 0 else 0.0

        summary = {
            "eval_days": n_eval_days,
            "total_demand": total_demand,
            "total_fulfilled": total_fulfilled,
            "shortage_units": total_shortage,
            "fill_rate": fill_rate,
            "stockout_rate": stockout_rate,
            "service_level": service_level,
            "mean_ending_inventory": mean_ending_inventory,
            "median_ending_inventory": median_ending_inventory,
            "average_inventory": mean_ending_inventory,
            "max_inventory": max_inventory,
            "number_of_orders": total_orders,
            "total_ordered_qty": total_ordered_qty,
            "average_order_qty": avg_order_qty,
            "ordering_cost": tot_order_cost,
            "holding_cost": tot_hold_cost,
            "shortage_cost": tot_short_cost,
            "total_cost": tot_cost,
            "cost_per_demanded_unit": cost_per_demanded_unit,
            "cost_per_fulfilled_unit": cost_per_fulfilled_unit
        }

        return {
            "summary": summary,
            "daily_df": daily_df,
            "eval_df": eval_df
        }
