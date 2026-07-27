# An Intelligent Decision Support System for Retail Sales Forecasting Using Machine Learning and Explainable Analytics

This repository contains the complete source code and directory structure for a final year M.Tech project. The system integrates machine learning (forecasting models), explainable artificial intelligence (SHAP interpretation), and inventory optimization rules to form an end-to-end Decision Support System (DSS) for retail store operations, configured to run on the **Kaggle Store Item Demand Forecasting Challenge** dataset.

---

## 🎯 Project Objective

Modern retail environments require precise, automated forecasting coupled with interpretable machine learning models to prevent stockouts and minimize inventory holding costs. 
This project:
1. **Predicts Retail Sales**: Evaluates historical trends, stores, and items to forecast sales quantities.
2. **Explains Predictions (XAI)**: Demystifies predictions using **SHAP (SHapley Additive exPlanations)**, highlighting global feature importance and individual forecast local drivers.
3. **Generates Decisions**: Converts forecasts directly into operational recommendations (e.g., *Urgent Restock*, *Run Clearance Promotion*, *Maintain Stock Level*).
4. **Kaggle Submission Generation**: Predicts demand on Kaggle's `test.csv` and outputs predictions in the required format.
5. **Interactive Dashboard**: Provides an executive-ready Streamlit dashboard for business operators to review analytical forecasts, explanations, and decisions.

---

## 📁 Folder Structure

```text
retail_project/
│
├── data/                       # Contains dataset files
│   ├── train.csv               # Kaggle training dataset (date, store, item, sales)
│   ├── test.csv                # Kaggle test dataset (id, date, store, item)
│   └── sample_submission.csv   # Kaggle sample submission layout (id, sales)
│
├── notebooks/                  # Directory for Jupyter notebooks
│
├── src/                        # Core Python pipeline modules
│   ├── preprocessing.py        # Loading, cleaning, and feature engineering (lags, rolling stats)
│   ├── eda.py                  # Exploratory Data Analysis & visual asset generation
│   ├── train_model.py          # Model training, serialization, and deserialization
│   ├── evaluate_model.py       # Metrics evaluation (MAE, RMSE, R2, MAPE) & performance plotting
│   ├── shap_analysis.py        # XAI interpretability calculations and SHAP plots
│   └── recommendation_engine.py# Decision support recommendations for stocking
│
├── dashboard/                  # Interactive user interface
│   └── app.py                  # Streamlit dashboard layout & plotly visualizers
│
├── results/                    # Output artifacts directory
│   ├── models/                 # Saved trained model files (.pkl)
│   ├── plots/                  # Visual plots (seasonality, errors, SHAP outputs)
│   └── metrics/                # Exported CSV recommendations and JSON evaluation metrics
│
├── reports/                    # Folder for final report PDFs, presentations, and draft documents
│
├── requirements.txt            # System library dependencies
├── README.md                   # Project documentation & run guide
└── main.py                     # Root execution script coordinating the full ML pipeline
```

---

## ⚙️ Installation Steps

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository
```bash
git clone <repository_url>
cd retail_project
```

### 2. Set Up a Virtual Environment (Recommended)
It is recommended to run the project in a dedicated Python environment to avoid package conflicts.
```bash
# Using Python venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using pip:
```bash
pip install -r requirements.txt
```

---

## 🚀 Data Setup and Execution Steps

### 1. Download Kaggle Dataset
1. Download the dataset from the Kaggle Competition: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only/data).
2. Unzip the downloaded files.
3. Place `train.csv`, `test.csv`, and `sample_submission.csv` inside the `data/` folder.

### 2. Execute the Pipeline
Run the main script from the project root to run the end-to-end forecasting pipeline (data loading, cleaning, training, evaluation, explainability, and recommendation export):
```bash
python main.py
```
This script will:
- Load the Kaggle training dataset (`data/train.csv`).
- Save analytical and seasonality plots (`results/plots/`).
- Train a Random Forest regressor (`results/models/sales_forecaster.pkl`).
- Save performance evaluation metrics (`results/metrics/evaluation_metrics.json`).
- Calculate SHAP values and output interpretability diagrams (`results/plots/shap_summary.png` and `results/plots/shap_local_waterfall.png`).
- Export decision recommendations (`results/metrics/inventory_recommendations.csv`).
- Output Kaggle submissions if `data/test.csv` is present (`results/metrics/submission.csv`).

### 3. Start the Interactive Dashboard
Launch the Streamlit web application to interact with forecasts and explainable insights:
```bash
streamlit run dashboard/app.py
```
*Note: If you launch the dashboard before running `main.py`, the app has a built-in initializer button that will automatically bootstrap and execute the pipeline for you.*

---

## 👥 Authors
- **M.Tech Project Candidate**
