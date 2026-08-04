import numpy as np

from qchallenge.metrics import balanced_accuracy, binary_cross_entropy


def test_metrics():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 1])
    assert balanced_accuracy(y, pred) == 0.75
    assert binary_cross_entropy(y, np.array([0.1, 0.8, 0.7, 0.9])) > 0

