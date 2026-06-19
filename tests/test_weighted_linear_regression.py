import numpy as np
from src.weighted_linear_regression import WeightedLinearRegression
from src.linear_regression import LinearRegression


def test_weighted_least_squares():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_w = np.array([2.0, -1.5, 3.0])
    y = X @ true_w + 0.1 * np.random.randn(100)

    weights = np.ones(100)
    weights[:50] = 10.0

    wls = WeightedLinearRegression()
    wls.fit(X, y, weights)

    assert wls.coef_ is not None
    assert len(wls.coef_) == 3


def test_weighted_vs_unweighted_equal_weights():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1, -2, 1.5]) + 0.1 * np.random.randn(50)

    weights = np.ones(50)

    wls = WeightedLinearRegression()
    wls.fit(X, y, weights)

    ols = LinearRegression()
    ols.fit(X, y)

    assert np.allclose(wls.coef_, ols.coef_, atol=1e-6)
