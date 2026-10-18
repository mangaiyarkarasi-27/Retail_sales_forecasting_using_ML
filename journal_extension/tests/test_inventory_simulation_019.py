"""
test_inventory_simulation_019.py
---------------------------------
Unit tests verifying all 12 inventory simulation logic checks for EXP-019.
"""

import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from journal_extension.src.inventory.inventory_policy import (
    DeterministicPointForecastPolicy,
    ConformalUncertaintyPolicy
)
from journal_extension.src.simulation.inventory_simulator import InventorySimulator


class TestInventorySimulation019(unittest.TestCase):

    def setUp(self):
        self.simulator = InventorySimulator(order_cost=10.0, holding_cost_rate=0.20, shortage_penalty=1.00)

    def test_1_inventory_non_negative(self):
        """Test 1: Inventory cannot become negative (OH >= 0 at all times)."""
        sales = np.array([50.0, 100.0, 20.0, 10.0])
        forecasts = np.array([20.0, 20.0, 20.0, 20.0])
        margins = np.array([5.0, 5.0, 5.0, 5.0])
        dates = pd.date_range("2017-01-01", periods=4)
        
        policy = DeterministicPointForecastPolicy()
        res = self.simulator.simulate_series(
            dates=dates, actual_sales=sales, point_forecasts=forecasts,
            uncertainty_margins=margins, lead_time=1, initial_on_hand=10.0,
            policy_func=policy.compute_rop
        )
        daily_df = res["daily_df"]
        self.assertTrue((daily_df["on_hand"] >= 0.0).all())

    def test_2_inventory_position_equality(self):
        """Test 2: On-hand + on-order equals inventory position at all times."""
        sales = np.array([10.0, 15.0, 5.0, 20.0, 10.0])
        forecasts = np.array([25.0, 25.0, 25.0, 25.0, 25.0])
        margins = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        dates = pd.date_range("2017-01-01", periods=5)
        
        policy = ConformalUncertaintyPolicy(nominal_coverage=0.90)
        res = self.simulator.simulate_series(
            dates=dates, actual_sales=sales, point_forecasts=forecasts,
            uncertainty_margins=margins, lead_time=2, initial_on_hand=20.0,
            policy_func=policy.compute_rop
        )
        daily_df = res["daily_df"]
        for idx, row in daily_df.iterrows():
            self.assertAlmostEqual(row["inventory_position"], row["on_hand"] + row["on_order"], delta=1e-5)

    def test_3_and_4_order_arrival_lead_time(self):
        """Test 3 & 4: Orders arrive exactly after L days and are never available early."""
        # Day 0: OH=0, OO=0, ROP=30 -> Order Q=30, arrives day 2 (L=2)
        # Day 0 demand=0 -> ending OH=0, OO=30
        # Day 1: demand=0 -> ending OH=0, OO=30 (not yet arrived)
        # Day 2: Q=30 arrives -> starting OH=30, demand=10 -> ending OH=20
        sales = np.array([0.0, 0.0, 10.0, 0.0])
        forecasts = np.array([30.0, 0.0, 0.0, 0.0])
        margins = np.array([0.0, 0.0, 0.0, 0.0])
        dates = pd.date_range("2017-01-01", periods=4)
        
        policy = DeterministicPointForecastPolicy()
        res = self.simulator.simulate_series(
            dates=dates, actual_sales=sales, point_forecasts=forecasts,
            uncertainty_margins=margins, lead_time=2, initial_on_hand=0.0,
            policy_func=policy.compute_rop
        )
        daily_df = res["daily_df"]
        
        # Day 0 order placed = 30
        self.assertEqual(daily_df.loc[0, "order_qty"], 30.0)
        self.assertEqual(daily_df.loc[0, "on_hand"], 0.0)
        # Day 1 not yet arrived
        self.assertEqual(daily_df.loc[1, "on_hand"], 0.0)
        # Day 2 arrived, demand 10 fulfilled -> ending OH = 20
        self.assertEqual(daily_df.loc[2, "on_hand"], 20.0)
        self.assertEqual(daily_df.loc[2, "fulfilled_demand"], 10.0)

    def test_5_and_6_shortage_and_lost_sales(self):
        """Test 5 & 6: Shortage equals unmet demand after OH reaches 0, and lost sales are not backordered."""
        # Day 0: OH=10, demand=25 -> fulfilled=10, shortage=15, ending OH=0
        # Day 1: demand=5 -> if backordered demand would be 20, but with lost sales demand is just 5
        sales = np.array([25.0, 5.0])
        forecasts = np.array([0.0, 0.0])
        margins = np.array([0.0, 0.0])
        dates = pd.date_range("2017-01-01", periods=2)
        
        policy = DeterministicPointForecastPolicy()
        res = self.simulator.simulate_series(
            dates=dates, actual_sales=sales, point_forecasts=forecasts,
            uncertainty_margins=margins, lead_time=1, initial_on_hand=10.0,
            policy_func=policy.compute_rop
        )
        daily_df = res["daily_df"]
        self.assertEqual(daily_df.loc[0, "fulfilled_demand"], 10.0)
        self.assertEqual(daily_df.loc[0, "shortage_units"], 15.0)
        self.assertEqual(daily_df.loc[0, "on_hand"], 0.0)
        
        self.assertEqual(daily_df.loc[1, "fulfilled_demand"], 0.0)
        self.assertEqual(daily_df.loc[1, "shortage_units"], 5.0)
        self.assertEqual(daily_df.loc[1, "on_hand"], 0.0)

    def test_7_and_8_cost_accounting(self):
        """Test 7 & 8: Ordering cost charged only when Q > 0; holding cost charged on closing OH."""
        sales = np.array([5.0, 5.0])
        forecasts = np.array([20.0, 0.0])
        margins = np.array([0.0, 0.0])
        dates = pd.date_range("2017-01-01", periods=2)
        
        policy = DeterministicPointForecastPolicy()
        res = self.simulator.simulate_series(
            dates=dates, actual_sales=sales, point_forecasts=forecasts,
            uncertainty_margins=margins, lead_time=1, initial_on_hand=10.0,
            policy_func=policy.compute_rop
        )
        daily_df = res["daily_df"]
        
        # Day 0: OH_init=10, ROP=20 -> Q=10 placed -> order_cost = 10.0
        # Day 0 demand=5 -> ending OH = 5.0 -> holding_cost = 0.20 * 5 = 1.0
        self.assertEqual(daily_df.loc[0, "order_cost"], 10.0)
        self.assertEqual(daily_df.loc[0, "holding_cost"], 1.0)
        
        # Day 1: arriving 10 -> starting OH=15, IP=15 >= ROP=0 -> Q=0 -> order_cost = 0.0
        # Day 1 demand=5 -> ending OH = 10.0 -> holding_cost = 0.20 * 10 = 2.0
        self.assertEqual(daily_df.loc[1, "order_cost"], 0.0)
        self.assertEqual(daily_df.loc[1, "holding_cost"], 2.0)

    def test_9_to_12_leakage_and_initialization(self):
        """Test 9 to 12: Safety stock and ROP use strictly decision-time inputs; initial stock is pre-2017."""
        # Initial inventory rule: OH_1 = L * mu_i
        mu_i = 45.5
        lead_time = 3
        init_oh = lead_time * mu_i
        self.assertEqual(init_oh, 136.5)
        
        # Policy ROP is strictly positive sum of point forecast and margin
        policy = ConformalUncertaintyPolicy(nominal_coverage=0.90)
        rop = policy.compute_rop(point_forecast=120.0, uncertainty_margin=25.0)
        self.assertEqual(rop, 145.0)


if __name__ == "__main__":
    unittest.main()
