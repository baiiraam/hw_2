"""
Ordinary Least Squares linear regression using normal equations.
"""

import numpy as np
from typing import Optional


class LinearRegression:
    """
    Linear regression model using closed-form OLS solution.

    Attributes
    ----------
    w : np.ndarray
        Weight vector including intercept as first element
    intercept_ : float
        Intercept term (bias)
    coef_ : np.ndarray
        Coefficients for features (excluding intercept)
    """

    def __init__(self) -> None:
        self.w: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit linear model using normal equations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : LinearRegression
            Fitted model
        """
        n_samples, n_features = X.shape

        # Add column of ones for intercept
        X_augmented = np.column_stack([np.ones(n_samples), X])

        # Solve (X^T X) w = X^T y using Cholesky / solve (more stable than inverse)
        # For better numerical stability, use np.linalg.lstsq or solve
        try:
            self.w = np.linalg.solve(X_augmented.T @ X_augmented, X_augmented.T @ y)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            self.w = np.linalg.lstsq(X_augmented, y, rcond=None)[0]

        self.intercept_ = self.w[0]
        self.coef_ = self.w[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values
        """
        if self.w is None:
            raise ValueError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        X_augmented = np.column_stack([np.ones(n_samples), X])
        return X_augmented @ self.w
