# ML Night Competition: 2nd Place Solution

![Project cover](assets/header.png)


![Python](https://img.shields.io/badge/Python-3.9-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.4-green)
![Decision Tree](https://img.shields.io/badge/Decision%20Tree-1.1.2-blue)
![RandomForest](https://img.shields.io/badge/RandomForest-1.1.2-yellow)
![GradientBoosting](https://img.shields.io/badge/GradientBoosting-1.1.2-orange)
![Ridge](https://img.shields.io/badge/Ridge-1.1.2-lightgrey)
![Lasso](https://img.shields.io/badge/Lasso-1.1.2-lightblue)
![ElasticNet](https://img.shields.io/badge/ElasticNet-1.1.2-purple)


## 📌 Scope

Understanding the potential cost of a media campaign is a crucial insight for marketing teams across all industries. It helps them make strategic decisions on how and where to allocate budgets, providing a competitive edge in the market.

## 📝 Objectives

The goal of this challenge is to predict the budgets invested in various media campaigns.

## 🚀 Project Overview

The Ad Campaign Budget Prediction project is a machine learning pipeline designed to forecast advertising campaign budgets based on historical campaign data and relevant performance indicators. The repository contains Python code, datasets, and Jupyter Notebooks demonstrating the complete workflow, including data preprocessing, feature engineering, model training and evaluation, and hyperparameter tuning.

## 🏆 Competition Results

- **Rank**: 2nd Place
- **Model**: XGBoost Regressor
- **Key Features**: Temporal feature engineering, outlier removal, hyperparameter optimization

## 🛠️ Tech Stack
- 🐍 Python 3.8+
- 📦 XGBoost, Decision Tree, RandomForestRegressor, GradientBoostingRegressor, Ridge, Lasso, ElasticNet for model training.
- 📊 Pandas, NumPy for data manipulation
- 📈 Matplotlib, Seaborn for EDA and visualization
- ⚙️ Scikit-learn for preprocessing, metrics, and pipelines
- 💻 Jupyter Notebook for development and reporting

## 🏁 Quick Start

### 📥 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mlnight-competition-solution.git
cd mlnight-competition-solution

# Create environment
conda env create -f environment.yml
conda activate mlnight-project

# Install package
pip install -e .

```

### ▶️ Usage

```bash
# Run complete pipeline
python main.py

# Or use Makefile
make train

# Run tests
make test

# Check code quality
make lint

# View ML experiments
mlflow ui
```

## 🐚 Windows (PowerShell) quick commands

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
# Run tests
make test
# Run pipeline (requires data in data/raw)
make train
```

## ⚠️ Notes
- Ensure `data/raw/train.csv`, `data/raw/test.csv`, and `data/raw/sample_submission.csv` exist before running `make train`.
- To avoid MLflow when running locally, call training with `use_mlflow=False` in `src/models/train.py` or set the parameter when using the training function.

## 🧑‍💻 Authors

**Mortadha Ferchichi**  
**Ayoub Ben Mahmoud**  
📧 ferchichii.mortadha@gmail.com  
📧 ayoubb917@gmail.com  
🌐 https://github.com/morta123456  
🌐 https://github.com/Ayoub-Ben-Mahmoud  
