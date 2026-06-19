from src.naive_bayes import GaussianNaiveBayes
from sklearn.datasets import load_iris


def test_naive_bayes():
    data = load_iris()
    X, y = data.data, data.target

    model = GaussianNaiveBayes()
    model.fit(X, y)
    pred = model.predict(X)

    assert len(pred) == len(y)
    assert model.means_ is not None
