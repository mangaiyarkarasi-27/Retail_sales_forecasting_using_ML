"""
eda.py
------
This module implements Module 2 (Exploratory Data Analysis) for the retail sales forecasting pipeline.
It defines an ExploratoryDataAnalysis class that loads processed training data, generates a series of
analytical visualizations, and compiles a decision-ready text summary report.
"""

import os
import logging
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure visualization aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})


class ExploratoryDataAnalysis:
    """
    Executes Exploratory Data Analysis on the preprocessed retail dataset.
    Generates time-series trends, distribution charts, heatmaps, and summary reports.
    """
    
    def __init__(
        self, 
        data_path: str = "data/processed_train.csv", 
        save_dir: str = "results/plots/", 
        report_path: str = "results/metrics/eda_summary.txt"
    ):
        """
        Initializes the EDA engine.
        
        Parameters:
        -----------
        data_path : str
            Path to the preprocessed training dataset.
        save_dir : str
            Directory where generated plots will be saved.
        report_path : str
            File destination path for the txt summary report.
        """
        self.data_path = data_path
        self.save_dir = os.path.normpath(save_dir)
        self.report_path = os.path.normpath(report_path)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the preprocessed training dataset.
        
        Returns:
        --------
        pd.DataFrame
            Loaded dataset.
        """
        logger.info(f"Loading preprocessed train dataset from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed training file not found at: {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        logger.info(f"Loaded processed dataset successfully. Shape: {self.df.shape}")
        return self.df

    def plot_overall_sales_trend(self) -> None:
        """
        Generates and saves the overall daily sales trend over time.
        """
        logger.info("Generating plot: Overall Sales Trend Over Time...")
        daily_sales = self.df.groupby("date")["sales"].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_sales["date"], daily_sales["sales"], color="#2563eb", linewidth=1.5, label="Daily Total Sales")
        
        # Overlay a rolling 30-day moving average to smooth seasonality
        daily_sales["rolling_ma30"] = daily_sales["sales"].rolling(window=30).mean()
        plt.plot(daily_sales["date"], daily_sales["rolling_ma30"], color="#ea580c", linewidth=2.5, label="30-Day Moving Avg")
        
        plt.title("Overall Daily Sales Trend (All Stores & Items)", pad=15)
        plt.xlabel("Date")
        plt.ylabel("Total Sales Quantity (Units)")
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        out_path = os.path.join(self.save_dir, "overall_sales_trend.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_monthly_sales_trend(self) -> None:
        """
        Generates and saves the monthly average sales seasonality trend.
        """
        logger.info("Generating plot: Monthly Sales Trend...")
        # Average sales per store/item grouping per month
        monthly_avg = self.df.groupby("month")["sales"].mean().reset_index()
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        plt.figure(figsize=(10, 5))
        sns.barplot(data=monthly_avg, x="month", y="sales", hue="month", palette="coolwarm", legend=False)
        plt.xticks(ticks=range(12), labels=month_labels)
        plt.title("Average Sales by Month (Seasonality Profile)", pad=15)
        plt.xlabel("Month")
        plt.ylabel("Average Sales per Transaction (Units)")
        plt.tight_layout()
        
        out_path = os.path.join(self.save_dir, "monthly_sales_trend.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_yearly_sales_trend(self) -> None:
        """
        Generates and saves the yearly sales growth trend.
        """
        logger.info("Generating plot: Yearly Sales Trend...")
        yearly_avg = self.df.groupby("year")["sales"].mean().reset_index()
        
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=yearly_avg, x="year", y="sales", marker="o", color="#16a34a", linewidth=2.5)
        plt.title("Average Sales Growth by Year", pad=15)
        plt.xlabel("Year")
        plt.ylabel("Average Sales per Transaction (Units)")
        plt.xticks(yearly_avg["year"])
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        
        out_path = os.path.join(self.save_dir, "yearly_sales_trend.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_store_wise_distribution(self) -> None:
        """
        Generates and saves a boxplot displaying sales distribution across stores.
        """
        logger.info("Generating plot: Store-wise Sales Distribution...")
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x="store", y="sales", hue="store", palette="Set2", legend=False)
        plt.title("Sales Quantity Distribution by Store", pad=15)
        plt.xlabel("Store ID")
        plt.ylabel("Sales Quantity (Units)")
        plt.tight_layout()
        
        out_path = os.path.join(self.save_dir, "store_sales_distribution.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_item_wise_distribution(self) -> None:
        """
        Generates and saves a distribution summary across all items.
        """
        logger.info("Generating plot: Item-wise Sales Distribution...")
        
        # Because we have 50 items, we make the figure wide and reduce spacing
        plt.figure(figsize=(16, 6))
        sns.boxplot(data=self.df, x="item", y="sales", color="#4f46e5")
        plt.title("Sales Quantity Distribution by Item", pad=15)
        plt.xlabel("Item ID")
        plt.ylabel("Sales Quantity (Units)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        out_path = os.path.join(self.save_dir, "item_sales_distribution.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_top_10_best_selling_items(self) -> None:
        """
        Generates and saves a bar chart listing the top 10 items by cumulative sales volume.
        """
        logger.info("Generating plot: Top 10 Best Selling Items...")
        top_items = self.df.groupby("item")["sales"].sum().reset_index()
        top_items = top_items.sort_values(by="sales", ascending=False).head(10)
        top_items["item"] = top_items["item"].astype(str)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_items, x="sales", y="item", hue="item", palette="viridis", legend=False)
        plt.title("Top 10 Best Selling Items (Cumulative Volume)", pad=15)
        plt.xlabel("Total Sales Volume (Units)")
        plt.ylabel("Item ID")
        plt.tight_layout()
        
        out_path = os.path.join(self.save_dir, "top_10_items.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_correlation_heatmap(self) -> None:
        """
        Generates and saves a correlation heatmap for numerical attributes.
        """
        logger.info("Generating plot: Correlation Heatmap...")
        
        # Select numeric features
        corr_cols = ["sales", "store", "item", "year", "month", "day", "weekday", "weekofyear", "quarter"]
        corr_matrix = self.df[corr_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Heatmap", pad=15)
        plt.tight_layout()
        
        out_path = os.path.join(self.save_dir, "correlation_heatmap.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_sales_distribution_histogram(self) -> None:
        """
        Generates and saves a sales distribution frequency histogram.
        """
        logger.info("Generating plot: Sales Distribution Histogram...")
        
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["sales"], bins=100, kde=True, color="#0891b2")
        plt.title("Sales Quantity Frequency Distribution", pad=15)
        plt.xlabel("Sales Quantity (Units)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        
        out_path = os.path.join(self.save_dir, "sales_distribution_histogram.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def generate_summary_report(self) -> None:
        """
        Analyzes the dataset, computes aggregate descriptive figures, 
        and writes a text report to disk.
        """
        logger.info("Calculating metrics and generating summary report...")
        
        # 1. Total Sales
        total_sales = int(self.df["sales"].sum())
        
        # 2. Average Daily Sales
        daily_total = self.df.groupby("date")["sales"].sum()
        avg_daily_sales = float(daily_total.mean())
        
        # 3. Highest Selling Store
        store_sales = self.df.groupby("store")["sales"].sum()
        highest_store = int(store_sales.idxmax())
        highest_store_sales = int(store_sales.max())
        
        # 4. Highest Selling Item
        item_sales = self.df.groupby("item")["sales"].sum()
        highest_item = int(item_sales.idxmax())
        highest_item_sales = int(item_sales.max())
        
        # 5. Peak Sales Month (seasonality peak)
        month_sales = self.df.groupby("month")["sales"].sum()
        peak_month = int(month_sales.idxmax())
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
            7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
        }
        peak_month_name = month_names[peak_month]
        
        # 6. Peak Sales Year
        year_sales = self.df.groupby("year")["sales"].sum()
        peak_year = int(year_sales.idxmax())
        
        # Log report details
        logger.info("Summary Metrics calculated:")
        logger.info(f" - Total Sales: {total_sales:,} units")
        logger.info(f" - Average Daily Sales: {avg_daily_sales:,.2f} units")
        logger.info(f" - Best Performing Store: Store {highest_store} ({highest_store_sales:,} units)")
        logger.info(f" - Best Selling Item: Item {highest_item} ({highest_item_sales:,} units)")
        logger.info(f" - Peak Seasonality Month: {peak_month_name}")
        logger.info(f" - Peak Growth Year: {peak_year}")

        # Ensure metrics folder exists and write the file
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                f.write("============================================================\n")
                f.write("EXPLORATORY DATA ANALYSIS (EDA) METRICS SUMMARY REPORT\n")
                f.write("============================================================\n\n")
                f.write(f"Project Title: An Intelligent Decision Support System for Retail Sales Forecasting\n")
                f.write(f"Dataset      : Kaggle Store Item Demand Forecasting dataset\n\n")
                f.write(f"1. Total Cumulative Sales        : {total_sales:,} units\n")
                f.write(f"2. Average Daily Sales Volume    : {avg_daily_sales:,.2f} units/day\n")
                f.write(f"3. Highest Performing Store      : Store {highest_store} (Cumulative Sales: {highest_store_sales:,} units)\n")
                f.write(f"4. Highest Selling Product Item  : Item {highest_item} (Cumulative Sales: {highest_item_sales:,} units)\n")
                f.write(f"5. Peak Seasonality Month        : {peak_month_name} (Month ID: {peak_month})\n")
                f.write(f"6. Peak Sales Volume Year        : {peak_year}\n\n")
                f.write("============================================================\n")
            logger.info(f"Summary report written successfully to {self.report_path}")
        except Exception as e:
            logger.error(f"Failed to write summary report: {e}")
            raise e

    def run_pipeline(self) -> None:
        """
        Runs the full EDA pipeline: directories initialization, plotting, and report writing.
        """
        logger.info("Starting EDA Pipeline...")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.load_data()
        
        self.plot_overall_sales_trend()
        self.plot_monthly_sales_trend()
        self.plot_yearly_sales_trend()
        self.plot_store_wise_distribution()
        self.plot_item_wise_distribution()
        self.plot_top_10_best_selling_items()
        self.plot_correlation_heatmap()
        self.plot_sales_distribution_histogram()
        
        self.generate_summary_report()
        logger.info("EDA Pipeline completed successfully.")


if __name__ == "__main__":
    eda = ExploratoryDataAnalysis()
    try:
        eda.run_pipeline()
    except Exception as error:
        logger.error(f"EDA pipeline execution failed: {error}")
