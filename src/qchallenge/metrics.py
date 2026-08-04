"""Small metric helpers with no classical predictive model dependency."""

from __future__ import annotations

import numpy as np


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if y.shape != pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    recall_0 = float(np.mean(pred[y == 0] == 0))
    recall_1 = float(np.mean(pred[y == 1] == 1))
    return (recall_0 + recall_1) / 2.0


def binary_cross_entropy(y_true: np.ndarray, p1: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    probability = np.clip(np.asarray(p1, dtype=float), 1e-8, 1 - 1e-8)
    return float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability)))

