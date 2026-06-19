"""
Bag of Words and TF-IDF text feature extraction.
"""

import numpy as np
from collections import Counter
import re
from typing import Optional


class BagOfWords:
    """
    Convert text documents to bag-of-words count vectors.

    Builds vocabulary of top n_features most frequent words.
    """

    def __init__(self, n_features: int = 5000) -> None:
        self.n_features = n_features
        self.vocabulary_: dict = {}  # word -> index mapping
        self.id_to_word_: dict = {}  # index -> word mapping

    def _tokenize(self, text: str) -> list:
        """
        Tokenize text into words (lowercase, remove non-alphanumeric).
        """
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\b[a-z]+\b", text)
        return tokens

    def fit(self, documents: list) -> "BagOfWords":
        """
        Build vocabulary from documents.

        Parameters
        ----------
        documents : list of str
            Training documents

        Returns
        -------
        self : BagOfWords
            Fitted transformer
        """
        # Count word frequencies across all documents
        word_counts = Counter()
        for doc in documents:
            tokens = self._tokenize(doc)
            word_counts.update(tokens)

        # Get top n_features most common words
        most_common = word_counts.most_common(self.n_features)

        # Build vocabulary
        self.vocabulary_ = {}
        self.id_to_word_ = {}
        for idx, (word, _) in enumerate(most_common):
            self.vocabulary_[word] = idx
            self.id_to_word_[idx] = word

        return self

    def transform(self, documents: list) -> np.ndarray:
        """
        Transform documents to count vectors.

        Parameters
        ----------
        documents : list of str
            Documents to transform

        Returns
        -------
        X : np.ndarray of shape (n_documents, n_features)
            Count matrix
        """
        if not self.vocabulary_:
            raise ValueError("BagOfWords must be fitted before transform")

        n_docs = len(documents)
        X = np.zeros((n_docs, self.n_features))

        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    idx = self.vocabulary_[token]
                    X[i, idx] += 1

        return X

    def fit_transform(self, documents: list) -> np.ndarray:
        """
        Fit vocabulary and transform documents in one step.
        """
        self.fit(documents)
        return self.transform(documents)


class TfidfTransformer:
    """
    Convert count matrix to TF-IDF weights.

    TF-IDF(t, d) = f_{t,d} * log(N / (1 + n_docs_with_t))
    """

    def __init__(self) -> None:
        self.idf_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "TfidfTransformer":
        """
        Compute IDF values from count matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_documents, n_features)
            Count matrix

        Returns
        -------
        self : TfidfTransformer
            Fitted transformer
        """
        n_docs, n_features = X.shape

        # Count documents containing each term
        doc_count = (X > 0).sum(axis=0)

        # Compute IDF: log(N / (1 + doc_count))
        self.idf_ = np.log((1 + n_docs) / (1 + doc_count)) + 1

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform count matrix to TF-IDF matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_documents, n_features)
            Count matrix

        Returns
        -------
        X_tfidf : np.ndarray of shape (n_documents, n_features)
            TF-IDF weighted matrix
        """
        if self.idf_ is None:
            raise ValueError("TfidfTransformer must be fitted before transform")

        # TF = raw counts, then multiply by IDF
        return X * self.idf_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit IDF and transform count matrix in one step.
        """
        self.fit(X)
        return self.transform(X)
