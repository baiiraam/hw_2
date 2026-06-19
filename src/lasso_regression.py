"""
Lasso regression (L1 regularization) using coordinate descent.
"""

import numpy as np
from typing import Optional


def soft_threshold(z: float, lambda_: float) -> float:
    """
    Soft-thresholding operator for Lasso.

    Parameters
    ----------
    z : float
        Input value (correlation residual)
    lambda_ : float
        Regularization strength

    Returns
    -------
    float
        Thresholded value
    """
    if z > lambda_:
        return z - lambda_
    elif z < -lambda_:
        return z + lambda_
    else:
        return 0.0


class LassoRegression:
    """
    Lasso regression model with L1 regularization using coordinate descent.

    Attributes
    ----------
    lambda_ : float
        Regularization strength
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    w : np.ndarray
        Weight vector (excluding intercept)
    intercept_ : float
        Intercept term
    """

    def __init__(
        self, lambda_: float = 1.0, max_iter: int = 10000, tol: float = 1e-4
    ) -> None:
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.w: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        """
        Fit Lasso model using cyclic coordinate descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : LassoRegression
            Fitted model
        """
        n_samples, n_features = X.shape

        # Standardize features
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        self.X_std_[self.X_std_ == 0] = 1
        X_scaled = (X - self.X_mean_) / self.X_std_

        # Center the target
        self.y_mean_ = y.mean()
        y_centered = y - self.y_mean_

        # Initialize coefficients
        self.w = np.zeros(n_features)

        for iteration in range(self.max_iter):
            w_old = self.w.copy()

            # Cycle through each coordinate
            for j in range(n_features):
                # Compute prediction without feature j
                residual = y_centered - X_scaled @ self.w + X_scaled[:, j] * self.w[j]

                # Compute correlation
                rho = X_scaled[:, j] @ residual

                # Apply soft-thresholding update
                denom = X_scaled[:, j] @ X_scaled[:, j]
                if denom == 0:
                    continue
                scaled_lambda = self.lambda_ * n_samples
                self.w[j] = soft_threshold(rho, scaled_lambda) / denom

            # Check convergence
            if np.max(np.abs(self.w - w_old)) < self.tol:
                break

        # Since predict manual scales X, intercept_ is simply the centered mean target value
        self.intercept_ = self.y_mean_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Lasso model.

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

        # Scale features and compute prediction
        X_scaled = (X - self.X_mean_) / self.X_std_
        return self.intercept_ + X_scaled @ self.w
