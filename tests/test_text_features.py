# tests/test_text_features.py
import numpy as np
from src.text_features import BagOfWords, TfidfTransformer


def test_bag_of_words():
    docs = ["hello world", "hello python", "python testing"]

    bow = BagOfWords(n_features=3)
    X = bow.fit_transform(docs)

    assert bow.vocabulary_ is not None
    assert len(bow.vocabulary_) == 3
    assert X.shape == (3, 3)
    assert isinstance(X, np.ndarray)


def test_tfidf_transformer():
    # Create count matrix
    X_counts = np.array([[2, 1, 0], [1, 0, 1], [0, 1, 2]])

    tfidf = TfidfTransformer()
    X_tfidf = tfidf.fit_transform(X_counts)

    assert tfidf.idf_ is not None
    assert X_tfidf.shape == X_counts.shape
    assert np.all(X_tfidf >= 0)


def test_bow_with_tokenization():
    docs = ["Hello, World! This is a test.", "Another document here."]

    bow = BagOfWords(n_features=10)
    bow.fit(docs)

    # Check that tokens are lowercase and cleaned
    tokens = bow._tokenize("Hello WORLD!!!")
    assert tokens == ["hello", "world"]
