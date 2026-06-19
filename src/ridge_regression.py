"""
Ridge regression (L2 regularization) using closed-form solution.
"""

import numpy as np
from typing import Optional


class RidgeRegression:
    """
    Ridge regression model with L2 regularization.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        self.lambda_ = lambda_
        self.w: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """
        Fit ridge regression model using closed-form solution.
        (No standardization - matches sklearn behavior)
        """
        n_samples, n_features = X.shape

        # Add intercept column
        X_augmented = np.column_stack([np.ones(n_samples), X])

        # Ridge closed-form: (X^T X + λI) w = X^T y
        # Note: We don't regularize the intercept (first column)
        XtX = X_augmented.T @ X_augmented

        # Create regularization matrix (λ on diagonal, but 0 for intercept)
        regularization = np.eye(XtX.shape[0]) * self.lambda_
        regularization[0, 0] = 0  # No regularization for intercept
        XtX_reg = XtX + regularization

        try:
            self.w = np.linalg.solve(XtX_reg, X_augmented.T @ y)
        except np.linalg.LinAlgError:
            self.w = np.linalg.lstsq(XtX_reg, X_augmented.T @ y, rcond=None)[0]

        self.intercept_ = self.w[0]
        self.coef_ = self.w[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ridge regression model.
        """
        if self.w is None:
            raise ValueError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        X_augmented = np.column_stack([np.ones(n_samples), X])
        return X_augmented @ self.w
