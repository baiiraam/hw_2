"""
Unit tests for linear models comparing against sklearn.
"""

import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from src.linear_regression import LinearRegression
from src.ridge_regression import RidgeRegression
from src.logistic_regression import LogisticRegression


def test_ols_vs_sklearn():
    """Test that our OLS matches sklearn's LinearRegression."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_w = np.array([2, -1, 3, 0.5, 1])
    y = X @ true_w + 0.1 * np.random.randn(100)

    our_model = LinearRegression()
    our_model.fit(X, y)

    sk_model = SklearnLinearRegression()
    sk_model.fit(X, y)

    # Add checks to ensure values exist
    assert our_model.coef_ is not None
    assert our_model.intercept_ is not None

    assert np.allclose(our_model.coef_, sk_model.coef_, atol=1e-6)
    assert np.allclose(our_model.intercept_, sk_model.intercept_, atol=1e-6)

    our_pred = our_model.predict(X)
    sk_pred = sk_model.predict(X)
    assert np.allclose(our_pred, sk_pred, atol=1e-6)


def test_ridge_vs_sklearn():
    """Test that our Ridge matches sklearn's Ridge."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_w = np.array([2, -1, 3, 0.5, 1])
    y = X @ true_w + 0.1 * np.random.randn(100)
    lambda_val = 0.5

    our_model = RidgeRegression(lambda_=lambda_val)
    our_model.fit(X, y)

    sk_model = SklearnRidge(alpha=lambda_val)
    sk_model.fit(X, y)

    assert our_model.coef_ is not None
    assert our_model.intercept_ is not None

    assert np.allclose(our_model.coef_, sk_model.coef_, atol=1e-2)
    assert np.allclose(our_model.intercept_, sk_model.intercept_, atol=1e-2)


def test_logistic_vs_sklearn():
    """Test that our logistic regression matches sklearn's."""
    np.random.seed(42)
    X = np.random.randn(200, 4)
    true_w = np.array([1, -2, 1.5, -0.5])
    logits = X @ true_w
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.random(200) < probs).astype(int)

    our_model = LogisticRegression(lr=0.1, lambda_=0.0, max_iter=1000)
    our_model.fit(X, y)

    sk_model = SklearnLogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    sk_model.fit(X, y)

    our_pred = our_model.predict(X)
    sk_pred = sk_model.predict(X)
    accuracy_match = np.mean(our_pred == sk_pred)
    assert accuracy_match > 0.95


def test_logistic_decision_boundary():
    """Test logistic regression on linearly separable data."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

    model = LogisticRegression(lr=0.1, lambda_=0.01, max_iter=1000)
    model.fit(X, y)

    pred = model.predict(X)
    accuracy = np.mean(pred == y)
    assert accuracy > 0.95


def test_gradient_descent():
    """Test LinearRegressionGD basic functionality."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1, 2, 3]) + 0.1 * np.random.randn(50)

    from src.linear_regression_gd import LinearRegressionGD

    model = LinearRegressionGD(lr=0.01, max_iter=100)
    model.fit(X, y)
    pred = model.predict(X)

    assert model.w is not None
    assert len(pred) == len(y)


def test_lasso_basic():
    """Test LassoRegression basic functionality."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1, 2, 3]) + 0.1 * np.random.randn(50)

    from src.lasso_regression import LassoRegression

    model = LassoRegression(lambda_=0.1)
    model.fit(X, y)
    pred = model.predict(X)

    assert model.w is not None
    assert len(pred) == len(y)


def test_logistic_regression_multiclass():
    """Test multiclass logistic regression."""
    np.random.seed(42)
    from sklearn.datasets import load_iris
    from src.logistic_regression import LogisticRegression

    data = load_iris()
    X, y = data.data, data.target

    model = LogisticRegression(lr=0.1, max_iter=500, tol=1e-6)
    model.fit(X, y)
    pred = model.predict(X)

    assert model.is_multiclass is True
    assert len(model.models) == 3  # 3 classes
    assert len(pred) == len(y)
    assert model.predict_proba(X).shape == (len(y), 3)


def test_ridge_with_different_lambda():
    """Test Ridge with various lambda values."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = X @ np.array([1, -1, 2, -2]) + 0.1 * np.random.randn(50)

    from src.ridge_regression import RidgeRegression

    # Test different lambda values
    for lam in [0, 0.1, 1, 10]:
        model = RidgeRegression(lambda_=lam)
        model.fit(X, y)
        pred = model.predict(X)

        assert model.coef_ is not None
        assert len(pred) == len(y)
