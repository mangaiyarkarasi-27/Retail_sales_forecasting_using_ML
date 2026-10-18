"""
run_oos_calibration_direct.py
------------------------------
Executes the out-of-sample (OOS) forecast generation on the training split,
calculates daily and cumulative lead-time demand residuals, and serializes
the calibration residual datasets under results/uncertainty/oos_direct/.
"""

import os
import time
import logging
import pandas as pd
import numpy as np

# Add src directories to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uncertainty")))

from oos_calibration_direct import OOSResidualsGeneratorDirect

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing OOS Conformal Calibration Residual Generator (Direct)...")
    
    # 1. Initialize Generator and run OOS predictions on training partition (2013-2016)
    generator = OOSResidualsGeneratorDirect()
    start_time = time.time()
    oos_df = generator.generate_oos_forecasts()
    logger.info(f"OOS forecasts completed in {time.time() - start_time:.2f} seconds.")
    
    # 2. Calculate daily residuals per store-item group
    daily_records = []
    lead_time_records = []
    
    lead_times = [1, 3, 5, 7]
    
    logger.info("Computing daily and cumulative lead-time demand residuals...")
    
    # Ensure sorting
    oos_df = oos_df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
    
    for (s, i), group in oos_df.groupby(["store", "item"]):
        group = group.reset_index(drop=True)
        sales_arr = group["sales"].values
        dates_arr = group["date"].values
        
        n = len(group)
        
        # A. Daily horizon residuals h=1..7
        for h in range(1, 8):
            preds_arr = group[f"pred_sales_h{h}"].values
            
            for t in range(n - h):
                actual_val = sales_arr[t + h]
                pred_val = preds_arr[t]
                signed_err = actual_val - pred_val
                abs_err = abs(signed_err)
                
                daily_records.append({
                    "store": int(s),
                    "item": int(i),
                    "forecast_origin": dates_arr[t],
                    "target_date": dates_arr[t + h],
                    "horizon": h,
                    "actual_sales": round(float(actual_val), 4),
                    "predicted_sales": round(float(pred_val), 4),
                    "signed_error": round(float(signed_err), 4),
                    "absolute_error": round(float(abs_err), 4)
                })
                
        # B. Cumulative lead-time demand residuals L=1,3,5,7
        for L in lead_times:
            # Predicted lead-time demand is the sum of pred_sales_h1...pred_sales_hL at origin t
            pred_lt_sum = np.zeros(n)
            for h in range(1, L + 1):
                pred_lt_sum += group[f"pred_sales_h{h}"].values
                
            for t in range(n - L):
                # Actual demand from t+1 to t+L
                actual_lt_demand = np.sum(sales_arr[t + 1 : t + L + 1])
                pred_lt_demand = pred_lt_sum[t]
                signed_lead_error = actual_lt_demand - pred_lt_demand
                abs_lead_error = abs(signed_lead_error)
                
                lead_time_records.append({
                    "store": int(s),
                    "item": int(i),
                    "forecast_origin": dates_arr[t],
                    "lead_time": L,
                    "actual_lead_demand": round(float(actual_lt_demand), 4),
                    "predicted_lead_demand": round(float(pred_lt_demand), 4),
                    "signed_lead_error": round(float(signed_lead_error), 4),
                    "absolute_lead_error": round(float(abs_lead_error), 4)
                })
                
    # 3. Create outputs dataframe and save
    logger.info("Saving residuals CSV files...")
    results_dir = "journal_extension/results/uncertainty/oos_direct"
    os.makedirs(results_dir, exist_ok=True)
    
    daily_df = pd.DataFrame(daily_records)
    daily_df.to_csv(os.path.join(results_dir, "daily_oos_residuals.csv"), index=False)
    
    lt_df = pd.DataFrame(lead_time_records)
    lt_df.to_csv(os.path.join(results_dir, "lead_time_oos_residuals.csv"), index=False)
    
    # 4. Generate summary stats per lead time
    summary_records = []
    for L in lead_times:
        lt_subset = lt_df[lt_df["lead_time"] == L]
        errors = lt_subset["signed_lead_error"].values
        abs_errors = lt_subset["absolute_lead_error"].values
        
        summary_records.append({
            "lead_time": L,
            "mean_signed_error": round(float(np.mean(errors)), 4),
            "MAE": round(float(np.mean(abs_errors)), 4),
            "RMSE": round(float(np.sqrt(np.mean(errors ** 2))), 4),
            "median_absolute_error": round(float(np.median(abs_errors)), 4),
            "std_deviation": round(float(np.std(errors)), 4),
            "minimum_error": round(float(np.min(errors)), 4),
            "maximum_error": round(float(np.max(errors)), 4),
            "residual_count": len(lt_subset)
        })
        
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(results_dir, "oos_residual_summary.csv"), index=False)
    logger.info("Direct OOS residuals and summary metrics computed and saved successfully.")


if __name__ == "__main__":
    main()
