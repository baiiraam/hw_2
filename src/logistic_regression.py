"""
Binary and multiclass logistic regression with L2 regularization.
"""

import numpy as np
from typing import Optional, List


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    """
    Logistic regression for binary classification.
    Supports one-vs-rest for multiclass.
    """

    def __init__(
        self,
        lr: float = 0.1,
        lambda_: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.lr = lr
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.w: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_multiclass: bool = False
        self.models: List["LogisticRegression"] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit logistic regression model."""
        self.classes_ = np.unique(y)
        self.is_multiclass = len(self.classes_) > 2

        if self.is_multiclass:
            self.models = []
            for cls in self.classes_:
                y_binary = (y == cls).astype(int)
                model = LogisticRegression(
                    lr=self.lr,
                    lambda_=self.lambda_,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                model._fit_binary(X, y_binary)
                self.models.append(model)
        else:
            self._fit_binary(X, y)

        return self

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit binary logistic regression using gradient descent."""
        n_samples, n_features = X.shape
        X_augmented = np.column_stack([np.ones(n_samples), X])
        self.w = np.zeros(n_features + 1)

        for iteration in range(self.max_iter):
            z = X_augmented @ self.w
            y_pred = sigmoid(z)
            gradient = (X_augmented.T @ (y_pred - y)) / n_samples
            gradient[1:] += (self.lambda_ / n_samples) * self.w[1:]
            w_new = self.w - self.lr * gradient

            if np.linalg.norm(w_new - self.w) < self.tol:
                self.w = w_new
                break
            self.w = w_new

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.is_multiclass:
            proba = np.zeros((X.shape[0], len(self.models)))
            for i, model in enumerate(self.models):
                proba[:, i] = model.predict_proba(X)
            exp_proba = np.exp(proba - np.max(proba, axis=1, keepdims=True))
            return exp_proba / exp_proba.sum(axis=1, keepdims=True)
        else:
            if self.w is None:
                raise ValueError("Model must be fitted before prediction")
            n_samples = X.shape[0]
            X_augmented = np.column_stack([np.ones(n_samples), X])
            z = X_augmented @ self.w
            return sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.is_multiclass:
            proba = self.predict_proba(X)
            # At this point, classes_ cannot be None because fit was called
            assert self.classes_ is not None
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            proba = self.predict_proba(X)
            return (proba >= 0.5).astype(int)
