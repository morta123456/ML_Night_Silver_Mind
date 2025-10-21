# ML Night Competition: 2nd Place Solution

![Python](https://img.shields.io/badge/Python-3.9-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning pipeline for predicting ad campaign budgets, achieving 2nd place in the ML Night competition.

## 🏆 Competition Results

- **Rank**: 2nd Place
- **Model**: XGBoost Regressor
- **Key Features**: Temporal feature engineering, outlier removal, hyperparameter optimization

## 🚀 Quick Start

### Installation

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

### Usage

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

Windows (PowerShell) quick commands

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
# Run tests
make test
# Run pipeline (requires data in data/raw)
make train
```

Notes
- Ensure `data/raw/train.csv`, `data/raw/test.csv`, and `data/raw/sample_submission.csv` exist before running `make train`.
- To avoid MLflow when running locally, call training with `use_mlflow=False` in `src/models/train.py` or set the parameter when using the training function.
