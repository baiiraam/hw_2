"""
Linear regression using batch gradient descent.
"""

import numpy as np
from typing import Optional


class LinearRegressionGD:
    """
    Linear regression model trained with batch gradient descent.

    Attributes
    ----------
    lr : float
        Learning rate
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
    w : np.ndarray
        Weight vector including intercept
    mse_history : list
        History of MSE values during training
    intercept_ : float
        Intercept term
    coef_ : np.ndarray
        Feature coefficients
    """

    def __init__(
        self, lr: float = 0.01, max_iter: int = 1000, tol: float = 1e-6
    ) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w: Optional[np.ndarray] = None
        self.mse_history: list = []
        self.intercept_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """
        Fit linear model using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : LinearRegressionGD
            Fitted model
        """
        n_samples, n_features = X.shape

        # Add column of ones for intercept
        X_augmented = np.column_stack([np.ones(n_samples), X])

        # Initialize weights to zero
        self.w = np.zeros(n_features + 1)
        self.mse_history = []

        for iteration in range(self.max_iter):
            # Compute predictions
            y_pred = X_augmented @ self.w

            # Compute MSE loss
            mse = np.mean((y - y_pred) ** 2)
            self.mse_history.append(mse)

            # Vectorized gradient: -2/N * X^T (y - Xw)
            gradient = -2 / n_samples * X_augmented.T @ (y - y_pred)

            # Update weights
            w_new = self.w - self.lr * gradient

            # Check convergence (norm of weight change)
            if np.linalg.norm(w_new - self.w) < self.tol:
                self.w = w_new
                break

            self.w = w_new

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
