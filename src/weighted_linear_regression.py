"""
Weighted Least Squares regression.

Solves: min_w Σ v_i (y_i - w^T x_i)²
where v_i are sample weights.

Closed-form: w = (X^T V X)^{-1} X^T V y
where V = diag(v)
"""

import numpy as np
from typing import Optional


class WeightedLinearRegression:
    """
    Weighted linear regression where each sample has a weight v_i > 0.

    Attributes
    ----------
    w : np.ndarray
        Weight vector including intercept as first element
    intercept_ : float
        Intercept term
    coef_ : np.ndarray
        Feature coefficients
    """

    def __init__(self) -> None:
        self.w: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.coef_: Optional[np.ndarray] = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> "WeightedLinearRegression":
        """
        Fit weighted linear regression.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values
        weights : np.ndarray of shape (n_samples,)
            Sample weights (must be positive)

        Returns
        -------
        self : WeightedLinearRegression
            Fitted model
        """
        n_samples, n_features = X.shape

        # Validate weights
        if np.any(weights <= 0):
            raise ValueError("All weights must be positive")

        # Add intercept column
        X_augmented = np.column_stack([np.ones(n_samples), X])

        # Compute weighted least squares efficiently
        # X^T V X = X^T * (weights * X)
        XtV = X_augmented.T * weights  # Shape: (n_features+1, n_samples)
        XtVX = XtV @ X_augmented
        XtVy = XtV @ y

        # Solve (X^T V X) w = X^T V y
        try:
            self.w = np.linalg.solve(XtVX, XtVy)
        except np.linalg.LinAlgError:
            self.w = np.linalg.lstsq(XtVX, XtVy, rcond=None)[0]

        self.intercept_ = self.w[0]
        self.coef_ = self.w[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the weighted linear model.

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

    def fit_with_heteroscedasticity(
        self, X: np.ndarray, y: np.ndarray
    ) -> "WeightedLinearRegression":
        """
        Fit with weights estimated from residuals (for heteroscedasticity).

        This is a two-step procedure:
        1. Fit unweighted OLS to get residuals
        2. Set weights = 1 / |residuals| (inverse variance weighting)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : WeightedLinearRegression
            Fitted model
        """
        from src.linear_regression import LinearRegression

        # Step 1: Fit unweighted OLS
        ols = LinearRegression()
        ols.fit(X, y)
        residuals = y - ols.predict(X)

        # Step 2: Compute weights as inverse of absolute residuals
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (np.abs(residuals) + 1e-6)

        # Step 3: Fit weighted regression
        return self.fit(X, y, weights)
