"""Train-only objectives computed from circuit output probabilities.

Both objectives read nothing but the measured q0 probability and the raw train
label, so neither introduces a classical predictive, surrogate or teacher model.

``balanced_bce`` is the established objective.  ``soft_balanced_accuracy`` exists
because it is not the metric the competition scores: balanced accuracy only asks
which side of the threshold a row lands on, while cross-entropy keeps paying to
push already-confident rows further out.  Measured on the trained C1 circuit,
82% of train rows sit more than 0.1 away from the threshold, so most of the
cross-entropy gradient is spent where the score cannot change.
"""

from __future__ import annotations

import numpy as np

ObjectiveResult = tuple[float, np.ndarray]


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary 0/1 values.")
    result = np.empty(len(labels), dtype=float)
    for label in (0, 1):
        mask = labels == label
        count = int(np.sum(mask))
        if count == 0:
            raise ValueError("Both classes must be present.")
        result[mask] = 0.5 / count
    return result


def balanced_bce(
    probability: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> ObjectiveResult:
    """Class-balanced binary cross-entropy and its derivative wrt probability."""
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    loss = -np.sum(
        sample_weight
        * (labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    )
    derivative = sample_weight * (
        (clipped - labels) / (clipped * (1.0 - clipped))
    )
    return float(loss), derivative


def soft_balanced_accuracy(
    probability: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    threshold: float = 0.5,
    temperature: float = 0.05,
) -> ObjectiveResult:
    """Negated smooth balanced accuracy and its derivative wrt probability.

    Each hard indicator ``1[p >= threshold]`` becomes ``sigmoid((p - t) / T)``.
    As ``T`` falls the objective converges to the scored metric, and its gradient
    concentrates on rows within roughly ``T`` of the threshold — the only rows
    whose classification can still change.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    margin = (np.asarray(probability, dtype=float) - threshold) / temperature
    # Numerically stable logistic.
    soft_positive = np.where(
        margin >= 0.0,
        1.0 / (1.0 + np.exp(-np.abs(margin))),
        np.exp(-np.abs(margin)) / (1.0 + np.exp(-np.abs(margin))),
    )
    correct = np.where(labels >= 0.5, soft_positive, 1.0 - soft_positive)
    loss = -float(np.sum(sample_weight * correct))
    slope = soft_positive * (1.0 - soft_positive) / temperature
    sign = np.where(labels >= 0.5, 1.0, -1.0)
    derivative = -sample_weight * sign * slope
    return loss, derivative


def temperature_schedule(
    start: float, stop: float, stages: int
) -> list[float]:
    """Geometric annealing schedule from ``start`` down to ``stop``."""
    if stages < 1:
        raise ValueError("Annealing needs at least one stage.")
    if start <= 0 or stop <= 0:
        raise ValueError("Temperatures must be positive.")
    if stages == 1:
        return [stop]
    return list(np.geomspace(start, stop, stages))
