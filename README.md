# 🔍 Fraud Detection ML System

A comprehensive machine learning system for fraud detection using ensemble methods and MLflow tracking. This project implements multiple algorithms with hyperparameter optimization and provides a deployment-ready API.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Deployment](#api-deployment)
- [Data Pipeline](#data-pipeline)
- [Monitoring](#monitoring)
- [Contributing](#contributing)

## 🎯 Overview

This fraud detection system uses advanced machine learning techniques to identify fraudulent transactions. The system implements:

- **Multiple ML algorithms**: XGBoost, LightGBM, and CatBoost
- **Ensemble methods**: Stacking and Voting classifiers
- **Custom evaluation metric** optimized for business impact
- **MLflow integration** for experiment tracking
- **FastAPI deployment** for real-time predictions

### Business Metric

The system optimizes a custom business metric that considers:
- Transaction amounts (`Monto`)
- False positive costs (blocked legitimate transactions)
- False negative costs (approved fraudulent transactions)
- Profit from legitimate transactions

```python
def metric(y_true, y_pred, amounts, p=0.25, fraud_loss=1.0):
    profit = 0.0
    profit += np.sum(p * amounts[true_negatives])      # Legitimate approved
    profit += np.sum(-p * amounts[false_positives])    # Legitimate blocked  
    profit += np.sum(fraud_loss * amounts[true_positives])   # Fraud blocked
    profit += np.sum(-fraud_loss * amounts[false_negatives]) # Fraud approved
    return profit
```

## 📁 Project Structure

```
test_meli/
├── 1_EDA.ipynb                    # Exploratory Data Analysis
├── 2_feature_engineering.ipynb    # Feature Engineering Pipeline
├── 3_training/                    # Training Pipeline
│   ├── base_model_optimizer.py    # Base optimizer class
│   ├── base_models.py            # ML model implementations
│   ├── data_handler.py           # Data processing utilities
│   ├── ensemble_classifiers.py   # Custom ensemble methods
│   ├── evaluation.py             # Custom evaluation metrics
│   ├── factory.py                # Model factory pattern
│   ├── pipeline.py               # Main training pipeline
│   ├── superlearner.py           # Ensemble optimization
│   └── train_test_plots.py       # Visualization utilities
├── 4_deployment/                 # API Deployment
│   └── api/
│       ├── main.py               # FastAPI application
│       ├── predict.py            # Prediction logic
│       ├── feature_engineering.py # Feature preprocessing
│       ├── schemas.py            # API schemas
│       └── artifacts/            # Model artifacts
└── data/
    ├── raw/                      # Raw data files
    └── processed/                # Processed data files
```

## ✨ Features

### Machine Learning
- **Multi-model approach**: XGBoost, LightGBM, CatBoost
- **Hyperparameter optimization** using Optuna with multi-objective optimization
- **Ensemble methods**: Stacking and Voting classifiers with greedy selection
- **Cross-validation** with stratified k-fold for robust evaluation
- **Imbalanced data handling** with custom oversampling strategies

### Experiment Tracking
- **MLflow integration** for comprehensive experiment tracking
- **Nested runs** for organized hyperparameter tuning
- **Artifact logging** for model persistence and plots
- **Parameter and metric tracking** across all experiments

### Data Processing
- **Automated feature engineering** pipeline
- **Missing value handling** with intelligent column removal
- **Categorical encoding** with OneHot encoding for country features
- **Data validation** and quality checks

### Deployment
- **FastAPI REST API** for real-time predictions
- **Pydantic schemas** for request/response validation
- **Model artifact management** with pickle serialization
- **Containerization ready** structure

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd test_meli

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Usage

### 1. Exploratory Data Analysis

```bash
jupyter notebook 1_EDA.ipynb
```

Key insights from EDA:
- **Target imbalance**: ~96% legitimate vs ~4% fraudulent transactions
- **Missing values**: Columns C and K removed due to high null percentage
- **Categorical features**: Country feature (J) with rare category grouping
- **Numerical features**: Multiple outliers that may indicate fraud patterns

### 2. Feature Engineering

```bash
jupyter notebook 2_feature_engineering.ipynb
```

Feature engineering steps:
- Remove high-missing columns (C, K)
- Group rare countries into 'OTHERS' category
- OneHot encode categorical features
- Fix numerical formatting (commas to decimals)
- Export processed data as Parquet

### 3. Model Training

```bash
cd 3_training
python pipeline.py
```

Training pipeline includes:
- **Data splitting** with stratification
- **Oversampling** minority class by 25%
- **Hyperparameter optimization** for each model
- **Ensemble creation** with greedy selection
- **Model evaluation** and comparison

### 4. API Deployment

```bash
cd 4_deployment/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /predict` - Single transaction prediction
- `GET /` - Health check endpoint

## 🔧 Model Training

### Training Configuration

The training pipeline supports multiple configurations:

```python
# Available models
models = ["xgboost", "lightgbm", "catboost"]

# Ensemble methods
ensemble_types = ["stacking", "voting"]

# Hyperparameter spaces are defined in base_models.py
```

### Hyperparameter Optimization

Each model uses Optuna for multi-objective optimization:

```python
# Example XGBoost hyperparameters
{
    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
    "max_depth": trial.suggest_int("max_depth", 4, 10),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
    "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2, 10),
}
```

### Ensemble Selection

The SuperLearner uses greedy ensemble selection:
1. Evaluate predefined model combinations
2. Select combination that improves over best base model
3. Early stopping when improvement is found

## 🚀 API Deployment

### Request Format

```json
{
    "A": 1,
    "B": 2,
    "D": 0,
    "E": 1,
    "F": 150.75,
    "G": 0.5,
    "H": 1,
    "I": 0,
    "J": "BR",
    "L": 2,
    "M": 1,
    "N": 0,
    "O": 1,
    "P": 0,
    "Q": 1200.50,
    "R": 800.25,
    "S": 1,
    "Monto": 1500.75
}
```

### Response Format

```json
{
    "prediction": 0
}
```

### Example Usage

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "A": 1, "B": 2, "D": 0, "E": 1,
    "F": 150.75, "G": 0.5, "H": 1, "I": 0,
    "J": "BR", "L": 2, "M": 1, "N": 0,
    "O": 1, "P": 0, "Q": 1200.50,
    "R": 800.25, "S": 1, "Monto": 1500.75
}

response = requests.post(url, json=data)
result = response.json()
print(f"Prediction: {result['prediction']}")
```

## 📈 Monitoring

### MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Access at http://localhost:5000
```

### Key Metrics Tracked
- **Business metric**: Custom profit-based evaluation
- **F1 Score**: Balanced precision and recall
- **Precision/Recall**: Individual class performance
- **Training time**: Model efficiency metrics

### Database Storage

Training results are stored in SQLite database:
```sql
-- Results table structure
CREATE TABLE training (
    id_experiment_parent TEXT,
    id_experiment_child TEXT,
    model TEXT,
    metric REAL,
    rank INTEGER
);
```