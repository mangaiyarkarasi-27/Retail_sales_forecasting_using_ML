"""
main.py
-------
Main execution script for the M.Tech project:
"An Intelligent Decision Support System for Retail Sales Forecasting Using Machine Learning and Explainable Analytics"

This file coordinates the execution of:
1. Module 1: Data Preprocessing (loading, cleaning, feature engineering)
2. Module 2: Exploratory Data Analysis (analytical plotting and reporting)
3. Module 3: Retail Sales Forecasting (model training, evaluation, comparison, selection)
4. Module 4: Explainable AI using SHAP (global/local model interpretability plotting and reporting)
5. Module 5: Intelligent Decision Recommendation Engine (inventory rules, trends, SHAP reasoning, confidence scores)
"""

import sys
import logging
from src.preprocessing import DataPreprocessor
from src.eda import ExploratoryDataAnalysis
from src.model_comparison import ModelComparer
from src.train_model import SalesForecaster
from src.shap_analysis import ShapExplainerPipeline
from src.recommendation_engine import DecisionRecommendationEngine
from src.decision_intelligence import DecisionIntelligenceFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Orchestrates:
    Preprocessing -> EDA -> Model Comparison -> Forecasting -> SHAP -> Recommendation Engine -> Decision Intelligence
    """
    logger.info("==============================================================")
    logger.info("STARTING RETAIL SALES FORECASTING PIPELINE: MODULES 1 TO 6")
    logger.info("==============================================================")
    
    try:
        # Step 1: Preprocessing
        logger.info("Executing Step 1: Preprocessing Pipeline...")
        preprocessor = DataPreprocessor(
            train_path="data/train.csv",
            test_path="data/test.csv"
        )
        preprocessor.run_pipeline()
        
        # Step 2: Exploratory Data Analysis (EDA)
        logger.info("Executing Step 2: Exploratory Data Analysis (EDA) Pipeline...")
        eda = ExploratoryDataAnalysis(
            data_path="data/processed_train.csv",
            save_dir="results/plots/",
            report_path="results/metrics/eda_summary.txt"
        )
        eda.run_pipeline()
        
        # Step 3A: Rigorous 5-Model Comparison Benchmark
        logger.info("Executing Step 3A: Model Comparison Benchmark Pipeline...")
        comparer = ModelComparer(
            data_path="data/processed_train.csv",
            metric_dir="results/metrics/",
            plot_dir="results/plots/"
        )
        comparer.run_benchmark()
        
        # Step 3B: Retail Sales Forecasting
        logger.info("Executing Step 3B: Sales Forecasting Pipeline...")
        forecaster = SalesForecaster(
            data_path="data/processed_train.csv",
            model_dir="results/models/",
            plot_dir="results/plots/",
            metric_dir="results/metrics/"
        )
        forecaster.run_forecasting_pipeline()
        
        # Step 4: Explainable AI (XAI) using SHAP
        logger.info("Executing Step 4: SHAP Explainable AI (XAI) Pipeline...")
        shap_explainer = ShapExplainerPipeline(
            model_path="results/models/best_model_v2.pkl",
            data_path="data/processed_train.csv",
            plot_dir="results/plots/shap/",
            report_path="results/metrics/shap_report.txt"
        )
        shap_explainer.run_shap_analysis()
        
        # Step 5: Intelligent Decision Recommendation Engine
        logger.info("Executing Step 5: Decision Recommendation Engine Pipeline...")
        rec_engine = DecisionRecommendationEngine(
            data_path="data/processed_train.csv",
            model_path="results/models/best_model_v2.pkl",
            out_dir="results/recommendations/"
        )
        rec_engine.run_recommendation_pipeline()
        
        # Step 6: Decision Intelligence Framework (E-IDSS)
        logger.info("Executing Step 6: Decision Intelligence Framework Pipeline...")
        di_framework = DecisionIntelligenceFramework(
            data_path="data/processed_train.csv",
            model_path="results/models/best_model_v2.pkl",
            out_dir="results/decision_intelligence/"
        )
        di_framework.run_framework_pipeline()
        
        logger.info("==============================================================")
        logger.info("FULL SYSTEM PIPELINE RUN (MODULES 1 TO 6) COMPLETED SUCCESSFULLY!")
        logger.info("All processed data, plots, summaries, forecasts, explanations, decisions, and E-IDSS reports are generated.")
        logger.info("==============================================================")
        
    except FileNotFoundError as fnf_error:
        logger.error(f"File Access Error during execution: {fnf_error}")
        sys.exit(1)
    except ValueError as val_error:
        logger.error(f"Validation Error during execution checks: {val_error}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error encountered during pipeline run: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
