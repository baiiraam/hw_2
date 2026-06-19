"""
Multinomial Naive Bayes classifier for text data with Laplace smoothing.
"""

import numpy as np
from typing import Optional


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier for count data.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.classes_: Optional[np.ndarray] = None
        self.class_log_priors_: Optional[np.ndarray] = None
        self.feature_log_probs_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        """Fit Multinomial Naive Bayes model."""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Compute class log priors
        self.class_log_priors_ = np.zeros(n_classes)
        for i, cls in enumerate(self.classes_):
            n_cls = np.sum(y == cls)
            self.class_log_priors_[i] = np.log(n_cls / n_samples)

        # Compute feature probabilities with Laplace smoothing
        self.feature_log_probs_ = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            feature_counts = X_cls.sum(axis=0)
            total_count = feature_counts.sum()

            smoothed_probs = (feature_counts + self.alpha) / (
                total_count + self.alpha * n_features
            )
            self.feature_log_probs_[i] = np.log(smoothed_probs)

        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log joint likelihood log P(x, y=c)."""
        if self.feature_log_probs_ is None or self.class_log_priors_ is None:
            raise ValueError("Model must be fitted before prediction")

        joint = X @ self.feature_log_probs_.T + self.class_log_priors_
        return joint

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        joint_log_likelihood = self._joint_log_likelihood(X)
        # Add assertion to help type checker
        assert self.classes_ is not None
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        joint_log_likelihood = self._joint_log_likelihood(X)

        # Convert to probabilities using log-sum-exp trick
        log_likelihood_exp = joint_log_likelihood - np.max(
            joint_log_likelihood, axis=1, keepdims=True
        )
        likelihood = np.exp(log_likelihood_exp)
        proba = likelihood / np.sum(likelihood, axis=1, keepdims=True)

        return proba


class MultinomialNaiveBayesForText:
    """
    Convenience wrapper for text classification with Multinomial NB.
    """

    def __init__(
        self, n_features: int = 5000, alpha: float = 1.0, use_tfidf: bool = True
    ):
        self.n_features = n_features
        self.alpha = alpha
        self.use_tfidf = use_tfidf
        self.bow = None
        self.tfidf = None
        self.nb: Optional[MultinomialNaiveBayes] = None

    def fit(self, documents: list, y: np.ndarray) -> "MultinomialNaiveBayesForText":
        """Fit the pipeline on text documents."""
        from src.text_features import BagOfWords, TfidfTransformer

        # Step 1: Bag of Words
        self.bow = BagOfWords(n_features=self.n_features)
        X_counts = self.bow.fit_transform(documents)

        # Step 2: TF-IDF (optional)
        if self.use_tfidf:
            self.tfidf = TfidfTransformer()
            X = self.tfidf.fit_transform(X_counts)
        else:
            X = X_counts

        # Step 3: Multinomial Naive Bayes
        self.nb = MultinomialNaiveBayes(alpha=self.alpha)
        self.nb.fit(X, y)

        return self

    def predict(self, documents: list) -> np.ndarray:
        """Predict classes for new documents."""
        if self.bow is None:
            raise ValueError("Model must be fitted before prediction")
        if self.nb is None:
            raise ValueError("Model must be fitted before prediction")

        X_counts = self.bow.transform(documents)

        if self.use_tfidf and self.tfidf is not None:
            X = self.tfidf.transform(X_counts)
        else:
            X = X_counts

        return self.nb.predict(X)

    def predict_proba(self, documents: list) -> np.ndarray:
        """Predict class probabilities for new documents."""
        if self.bow is None:
            raise ValueError("Model must be fitted before prediction")
        if self.nb is None:
            raise ValueError("Model must be fitted before prediction")

        X_counts = self.bow.transform(documents)

        if self.use_tfidf and self.tfidf is not None:
            X = self.tfidf.transform(X_counts)
        else:
            X = X_counts

        return self.nb.predict_proba(X)
