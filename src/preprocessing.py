"""
preprocessing.py
----------------
This module implements Module 1 (Data Loading and Preprocessing) for the retail sales forecasting pipeline.
It defines a reusable DataPreprocessor class to handle schema validation, data cleaning, date conversions, 
feature engineering, dataset summarizing, and file exports for train and test datasets.
"""

import os
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A reusable data preprocessing pipeline for the Kaggle Store Item Demand dataset.
    Handles loading, schema validation, quality checks, feature engineering, and saving of processed data.
    """
    
    def __init__(self, train_path: str = "data/train.csv", test_path: str = "data/test.csv"):
        """
        Initializes the preprocessor with file paths.
        
        Parameters:
        -----------
        train_path : str
            Path to the training dataset CSV file.
        test_path : str
            Path to the test dataset CSV file.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        
        # Expected schemas
        self.expected_train_cols = ["date", "store", "item", "sales"]
        self.expected_test_cols = ["id", "date", "store", "item"]
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the train and test datasets from CSV files.
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Loaded train and test DataFrames.
        """
        logger.info("Step 1: Loading raw datasets...")
        
        # Validate train file existence
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training dataset file not found at: {self.train_path}")
        # Validate test file existence
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test dataset file not found at: {self.test_path}")
            
        try:
            self.train_df = pd.read_csv(self.train_path, sep="\t")
            logger.info(f"Loaded training data successfully. Shape: {self.train_df.shape}")
        except Exception as e:
            logger.error(f"Error reading training dataset: {e}")
            raise e
            
        try:
            self.test_df = pd.read_csv(self.test_path, sep="\t")
            logger.info(f"Loaded test data successfully. Shape: {self.test_df.shape}")
        except Exception as e:
            logger.error(f"Error reading test dataset: {e}")
            raise e
            
        return self.train_df, self.test_df

    def validate_columns(self, df: pd.DataFrame, expected_cols: list, dataset_name: str) -> None:
        """
        Validates that the required columns are present in the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to validate.
        expected_cols : list
            List of column names required in the DataFrame.
        dataset_name : str
            Name of the dataset (e.g., 'Train', 'Test') for logging.
        """
        logger.info(f"Step 2: Validating columns for {dataset_name} dataset...")
        missing_cols = [col for col in expected_cols if col not in df.columns]
        
        if missing_cols:
            error_msg = f"Schema validation failed for {dataset_name} dataset. Missing columns: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Column validation passed for {dataset_name} dataset.")

    def clean_data(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Cleans the dataset by:
        - Checking for and reporting missing values
        - Converting the date column to datetime
        - Reporting and dropping duplicate rows
        - Verifying column data types
        
        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to clean.
        dataset_name : str
            Name of the dataset for logging.
            
        Returns:
        --------
        pd.DataFrame
            The cleaned DataFrame.
        """
        logger.info(f"Step 3: Cleaning {dataset_name} dataset...")
        df_clean = df.copy()
        
        # 1. Convert Date Column
        logger.info(f"Converting 'date' column in {dataset_name} to datetime...")
        try:
            df_clean["date"] = pd.to_datetime(df_clean["date"], dayfirst=True)
        except Exception as e:
            logger.error(f"Error converting date column to datetime in {dataset_name}: {e}")
            raise e

        # 2. Check for Missing Values
        missing_counts = df_clean.isnull().sum()
        total_missing = missing_counts.sum()
        if total_missing > 0:
            logger.warning(f"Found {total_missing} missing values in {dataset_name} dataset:")
            for col, count in missing_counts[missing_counts > 0].items():
                logger.warning(f" - Column '{col}': {count} missing values")
            # Impute missing values (forward-fill then backward-fill for time-series order)
            logger.info("Imputing missing values using ffill and bfill...")
            df_clean = df_clean.ffill().bfill()
        else:
            logger.info(f"No missing values found in {dataset_name} dataset.")

        # 3. Check for Duplicate Rows
        duplicate_count = df_clean.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows in {dataset_name} dataset. Dropping duplicates...")
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            logger.info(f"Duplicates dropped. New shape: {df_clean.shape}")
        else:
            logger.info(f"No duplicate rows found in {dataset_name} dataset.")

        # 4. Check Data Types
        logger.info(f"Column data types in {dataset_name} dataset:")
        for col, dtype in df_clean.dtypes.items():
            logger.info(f" - Column '{col}': {dtype}")
            
        return df_clean

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates engineered date-based features:
        - year
        - month
        - day
        - weekday (0 = Monday, 6 = Sunday)
        - weekofyear (1 to 53)
        - quarter (1 to 4)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing a datetime 'date' column.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features.
        """
        logger.info("Step 4: Generating engineered features from date column...")
        df_engineered = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df_engineered["date"]):
            raise ValueError("The 'date' column must be converted to datetime before feature engineering.")
            
        # Extract features
        df_engineered["year"] = df_engineered["date"].dt.year
        df_engineered["month"] = df_engineered["date"].dt.month
        df_engineered["day"] = df_engineered["date"].dt.day
        df_engineered["weekday"] = df_engineered["date"].dt.dayofweek
        df_engineered["weekofyear"] = df_engineered["date"].dt.isocalendar().week.astype(int)
        df_engineered["quarter"] = df_engineered["date"].dt.quarter
        
        logger.info(f"Features generated. Added 6 new date-based columns.")
        return df_engineered

    def get_summary(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Computes and logs a summary of the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to summarize.
        dataset_name : str
            Name of the dataset for display.
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing summary statistics.
        """
        logger.info(f"Step 5: Generating {dataset_name} dataset summary...")
        
        row_count = len(df)
        col_count = len(df.columns)
        
        min_date = df["date"].min().strftime("%Y-%m-%d")
        max_date = df["date"].max().strftime("%Y-%m-%d")
        
        num_stores = df["store"].nunique()
        num_items = df["item"].nunique()
        
        summary = {
            "row_count": row_count,
            "col_count": col_count,
            "min_date": min_date,
            "max_date": max_date,
            "num_stores": num_stores,
            "num_items": num_items
        }
        
        logger.info(f"=== {dataset_name} DATASET SUMMARY ===")
        logger.info(f" - Row count: {row_count}")
        logger.info(f" - Column count: {col_count}")
        logger.info(f" - Date range: {min_date} to {max_date}")
        logger.info(f" - Unique stores: {num_stores}")
        logger.info(f" - Unique items: {num_items}")
        logger.info("====================================")
        
        return summary

    def save_processed_data(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        train_out_path: str = "data/processed_train.csv", 
        test_out_path: str = "data/processed_test.csv"
    ) -> None:
        """
        Saves the processed train and test datasets as CSV files.
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Processed train dataset.
        test_df : pd.DataFrame
            Processed test dataset.
        train_out_path : str
            File destination path for processed train data.
        test_out_path : str
            File destination path for processed test data.
        """
        logger.info("Step 6: Saving processed datasets...")
        
        try:
            os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
            train_df.to_csv(train_out_path, index=False)
            logger.info(f"Saved processed train data to {train_out_path}. Shape: {train_df.shape}")
        except Exception as e:
            logger.error(f"Error saving processed training dataset: {e}")
            raise e
            
        try:
            os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
            test_df.to_csv(test_out_path, index=False)
            logger.info(f"Saved processed test data to {test_out_path}. Shape: {test_df.shape}")
        except Exception as e:
            logger.error(f"Error saving processed test dataset: {e}")
            raise e

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the entire preprocessing pipeline sequentially.
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Processed train and test DataFrames.
        """
        logger.info("Starting Data Preprocessing Pipeline...")
        
        # Load raw data
        train_raw, test_raw = self.load_data()
        
        # Validate schemas
        self.validate_columns(train_raw, self.expected_train_cols, "Train")
        self.validate_columns(test_raw, self.expected_test_cols, "Test")
        
        # Clean data
        train_clean = self.clean_data(train_raw, "Train")
        test_clean = self.clean_data(test_raw, "Test")
        
        # Feature engineering
        train_processed = self.engineer_features(train_clean)
        test_processed = self.engineer_features(test_clean)
        
        # Summary analysis
        self.get_summary(train_processed, "Train")
        self.get_summary(test_processed, "Test")
        
        # Save processed data
        self.save_processed_data(train_processed, test_processed)
        
        logger.info("Data Preprocessing Pipeline completed successfully.")
        return train_processed, test_processed


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    try:
        preprocessor.run_pipeline()
    except Exception as error:
        logger.error(f"Preprocessing pipeline failed: {error}")
