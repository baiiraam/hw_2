# Homework 2: Linear Models, Regularization, and Classification

**Author:** Bayram Bayramov
\
**Course:** AI-ENG-201 – Machine Learning – Fall 2026
\
**Date:** 2026-06-15

## Overview

This implementation covers all required components of Homework 2, including:

- **Linear Regression**: Closed-form OLS and batch gradient descent implementations
- **Regularization**: Ridge regression (closed-form) and Lasso regression (coordinate descent)
- **Classification**: Binary and multiclass logistic regression, Gaussian Naive Bayes
- **Text Classification**: Bag-of-Words and TF-IDF from scratch
- **Evaluation**: Learning curves, regularization paths, ROC curves, calibration curves

## Project Structure

```
hw2/
├── report.pdf                 # Analysis report (<=12 pages)
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── pledge.txt                 # Signed honor pledge
├── src/                       # Source code modules
│   ├── linear_regression.py   # OLS via normal equations
│   ├── linear_regression_gd.py # OLS via gradient descent
│   ├── ridge_regression.py    # Ridge regression
│   ├── lasso_regression.py    # Lasso with coordinate descent
│   ├── logistic_regression.py # Binary/multiclass logistic regression
│   ├── naive_bayes.py         # Gaussian Naive Bayes
│   ├── text_features.py       # BagOfWords and TF-IDF
│   ├── multinomial_naive_bayes.py # Multinomial Naive Bayes (bonus)
│   └── weighted_linear_regression.py # Weighted Linear Regression (bonus)
├── notebooks/
│   └── hw2_analysis.ipynb     # Complete analysis notebook
├── tests/                     # Unit tests
│   ├── test_linear_models.py  # Tests for linear/logistic models
│   ├── test_lasso.py          # Lasso regression tests
│   ├── test_naive_bayes.py    # Naive Bayes tests
│   ├── test_text_features.py  # Text feature tests
│   ├── test_multinomial_naive_bayes.py # Multinomial NB tests
│   └── test_weighted_linear_regression.py # Weighted LR tests
└── figures/                   # Generated plots (PDF format)
```

## Requirements

- Python >= 3.11
- Dependencies listed in `requirements.txt`

## Setup and Installation

### Using uv (Recommended)

```bash
uv init . && uv add {libraries_here}
```
or conventionally,
```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis
You do not need to install jupyter specifically. Ipykernel would also do the job.

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/hw2_analysis.ipynb

# Or run as Python script
jupyter nbconvert --to script notebooks/hw2_analysis.ipynb
python notebooks/hw2_analysis.py
```

## Running Tests
First command needs pytest, second command needs pytest-cov to check the coverage.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term

# Run specific test file
pytest tests/test_linear_models.py -v
```

## Test Coverage

| Module | Coverage |
|--------|----------|
| linear_regression.py | 87% |
| linear_regression_gd.py | 91% |
| ridge_regression.py | 89% |
| lasso_regression.py | 93% |
| logistic_regression.py | 95% |
| naive_bayes.py | 83% |
| text_features.py | 96% |
| multinomial_naive_bayes.py | 88% |
| weighted_linear_regression.py | 60% |
| **Overall** | **88%** |

## Implementation Details

### Linear Regression
- **OLS**: Normal equations with `np.linalg.solve` for numerical stability
- **Gradient Descent**: Batch gradient descent with MSE loss, vectorized operations

### Ridge Regression
- Closed-form solution: $(X^TX + lambda * I)^{-1}X^Ty$
- Intercept not regularized

### Lasso Regression
- Cyclic coordinate descent with soft-thresholding
- Feature standardization recommended

### Logistic Regression
- Gradient descent on cross-entropy loss
- L2 regularization support
- One-vs-rest for multiclass classification

### Naive Bayes
- Gaussian class-conditional densities
- Log-space computations for numerical stability
- Laplace smoothing for variances

### Text Features
- **BagOfWords**: Top 5000 words by frequency
- **TF-IDF**: $f_{t,d} \times \log(N / (1 + |\{d: f_{t,d} > 0\}|))$

### Bonus: Weighted Linear Regression
- Closed-form solution: $w = (X^T V X)^{-1} X^T V y$ where $V = diag(v)$
- Supports heteroscedasticity via two-step estimation

### Bonus: Multinomial Naive Bayes
- Multinomial distribution for count data
- Laplace smoothing to prevent zero probabilities
- Optimized for text classification with raw counts

## Results Summary

| Experiment | Key Finding |
|------------|-------------|
| Polynomial Fitting | Best degree = 8 |
| Gradient Descent | Converges in ~67 iterations (eta=0.1) |
| Lasso vs Ridge | Lasso provides sparsity |
| Hyperparameter Search | Random search more efficient |
| Wine Classification | Naive Bayes: 100%, LR: 98% |
| Text Classification | LR AUC: 0.997, NB AUC: 0.976 |

## Reproducibility

All random operations are seeded with `np.random.seed(42)` to ensure reproducible results.