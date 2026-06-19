"""
Gaussian Naive Bayes classifier.
"""

import numpy as np
from typing import Optional


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.
    """

    def __init__(self) -> None:
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.variances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """Fit Gaussian Naive Bayes model."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            n_cls = X_cls.shape[0]
            self.priors_[i] = np.log(n_cls / len(y))
            self.means_[i] = X_cls.mean(axis=0)
            self.variances_[i] = X_cls.var(axis=0) + 1e-6

        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log-likelihood log P(X|y=c) for each class."""
        # These cannot be None because fit was called before
        assert self.means_ is not None
        assert self.variances_ is not None
        assert self.classes_ is not None

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        ll = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            diff = X - self.means_[i]
            variances = np.clip(self.variances_[i], 1e-6, None)
            log_var = np.log(variances)
            ll[:, i] = -0.5 * np.sum(
                (diff**2) / variances + log_var + np.log(2 * np.pi), axis=1
            )
        return ll

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.priors_ is None:
            raise ValueError("Model must be fitted before prediction")

        log_posterior = self._log_likelihood(X) + self.priors_
        assert self.classes_ is not None
        return self.classes_[np.argmax(log_posterior, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.priors_ is None:
            raise ValueError("Model must be fitted before prediction")

        log_posterior = self._log_likelihood(X) + self.priors_
        log_posterior_exp = log_posterior - np.max(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior_exp)
        posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
        return posterior
