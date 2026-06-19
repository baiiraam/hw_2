"""
Unit tests for Multinomial Naive Bayes.
"""

import numpy as np
from src.multinomial_naive_bayes import (
    MultinomialNaiveBayes,
    MultinomialNaiveBayesForText,
)


def test_multinomial_nb_basic():
    """Test Multinomial NB on simple count data."""
    np.random.seed(42)

    X = np.random.randint(0, 5, size=(100, 10))
    y = np.random.randint(0, 3, size=100)

    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)

    assert model.classes_ is not None
    assert model.feature_log_probs_ is not None
    assert len(pred) == len(y)
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_multinomial_nb_laplace_smoothing():
    """Test that Laplace smoothing prevents zero probabilities."""
    np.random.seed(42)

    X = np.random.randint(0, 2, size=(50, 20))
    y = np.array([0] * 25 + [1] * 25)

    # Ensure feature 5 is only in class 0
    X[y == 1, 5] = 0

    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(X, y)

    # Check for -inf values - fix the type checker issue
    feature_log_probs = model.feature_log_probs_
    assert feature_log_probs is not None
    assert not np.any(np.isinf(feature_log_probs))  # Simplified assertion

    # Predict should work
    pred = model.predict(X)
    assert len(pred) == len(y)


def test_multinomial_nb_with_alpha_zero():
    """Test that alpha=0 can cause zeros."""
    np.random.seed(42)

    X = np.random.randint(0, 2, size=(50, 20))
    y = np.array([0] * 25 + [1] * 25)

    # Ensure feature 5 is only in class 0
    X[y == 1, 5] = 0

    model = MultinomialNaiveBayes(alpha=0.0)
    model.fit(X, y)

    # With alpha=0, some probabilities become -inf
    feature_log_probs = model.feature_log_probs_
    assert feature_log_probs is not None
    # Some values will be -inf, which is expected
    assert np.any(np.isinf(feature_log_probs))


def test_multinomial_vs_gaussian_on_text():
    """Compare Multinomial NB with Gaussian NB on text data."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from src.text_features import BagOfWords, TfidfTransformer
    from src.naive_bayes import GaussianNaiveBayes

    categories = ["comp.graphics", "talk.religion.misc"]
    newsgroups = fetch_20newsgroups(
        subset="all", categories=categories, random_state=42
    )
    X_text, y_text = newsgroups.data, newsgroups.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_text, test_size=0.3, random_state=42
    )

    bow = BagOfWords(n_features=1000)
    X_train_counts = bow.fit_transform(X_train)
    X_test_counts = bow.transform(X_test)

    tfidf = TfidfTransformer()
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    X_test_tfidf = tfidf.transform(X_test_counts)

    # Multinomial NB (on counts)
    mnb = MultinomialNaiveBayes(alpha=1.0)
    mnb.fit(X_train_counts, y_train)
    mnb_pred = mnb.predict(X_test_counts)
    mnb_accuracy = np.mean(mnb_pred == y_test)

    # Gaussian NB (on TF-IDF)
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train_tfidf, y_train)
    gnb_pred = gnb.predict(X_test_tfidf)
    gnb_accuracy = np.mean(gnb_pred == y_test)

    print("\n=== Multinomial NB vs Gaussian NB on Text ===")
    print(f"Multinomial NB accuracy: {mnb_accuracy:.4f}")
    print(f"Gaussian NB accuracy:    {gnb_accuracy:.4f}")

    # Multinomial should perform better on count data
    assert mnb_accuracy > 0.7


def test_multinomial_nb_pipeline():
    """Test the full pipeline wrapper."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import train_test_split

    categories = ["comp.graphics", "talk.religion.misc"]
    newsgroups = fetch_20newsgroups(
        subset="all", categories=categories, random_state=42
    )
    X_text, y_text = newsgroups.data, newsgroups.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_text, test_size=0.3, random_state=42
    )

    pipeline = MultinomialNaiveBayesForText(n_features=2000, alpha=1.0, use_tfidf=False)
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    assert len(pred) == len(y_test)
    assert proba.shape == (len(y_test), 2)
    print(f"\nPipeline accuracy: {np.mean(pred == y_test):.4f}")
