import numpy as np
from src.lasso_regression import LassoRegression


def test_lasso_basic():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([2, -1, 1.5]) + 0.1 * np.random.randn(50)

    model = LassoRegression(lambda_=0.1)
    model.fit(X, y)
    predictions = model.predict(X)

    assert model.w is not None
    assert len(predictions) == len(y)
