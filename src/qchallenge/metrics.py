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


def balanced_binary_cross_entropy(y_true: np.ndarray, p1: np.ndarray) -> float:
    """Average class-conditional BCE equally; this is a loss, not a predictor."""
    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(p1, dtype=float), 1e-8, 1 - 1e-8)
    negative = -np.log(1 - probability[y == 0])
    positive = -np.log(probability[y == 1])
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("Both classes are required for balanced BCE.")
    return float((np.mean(negative) + np.mean(positive)) / 2.0)
