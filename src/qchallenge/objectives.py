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


def smooth_auc(
    probability: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    temperature: float = 0.05,
) -> ObjectiveResult:
    """Negated smooth ROC AUC and its derivative wrt probability.

    Replaces each pairwise indicator ``1[p_pos > p_neg]`` with
    ``sigmoid((p_pos - p_neg) / T)`` and averages over all positive/negative
    pairs.  Unlike the annealed balanced-accuracy surrogate, which concentrates
    on the shrinking set of rows within ``T`` of the threshold and overfits it,
    every row keeps a share of the gradient here and no threshold appears at
    all -- the threshold is chosen afterwards from out-of-fold probabilities.

    ``sample_weight`` is accepted for interface compatibility and unused: AUC is
    already balanced by construction, since it normalizes over positive/negative
    pairs rather than over rows.

    Cost is the full ``n_pos * n_neg`` pair set, evaluated exactly.  The
    rank-based ``O(n log n)`` shortcut applies to the hard indicator, not to a
    sigmoid of the score difference, so positives are processed in blocks to
    bound the pair matrix instead.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    values = np.asarray(probability, dtype=float)
    positive = values[np.asarray(labels) >= 0.5]
    negative = values[np.asarray(labels) < 0.5]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Both classes must be present for AUC.")

    scaled_positive = positive / temperature
    scaled_negative = negative / temperature
    order = np.argsort(scaled_negative, kind="stable")
    sorted_negative = scaled_negative[order]

    # sigmoid(a - b) = exp(a) / (exp(a) + exp(b)); work in a shifted, stable form
    # by splitting each pair on which side is larger.
    total = 0.0
    gradient_positive = np.zeros(len(positive), dtype=float)
    gradient_negative = np.zeros(len(negative), dtype=float)
    # Chunk the positives so the pair matrix stays small but exact.
    chunk = max(1, 4_000_000 // max(len(negative), 1))
    for start in range(0, len(positive), chunk):
        block = scaled_positive[start : start + chunk]
        difference = block[:, None] - sorted_negative[None, :]
        sigmoid = np.where(
            difference >= 0.0,
            1.0 / (1.0 + np.exp(-np.abs(difference))),
            np.exp(-np.abs(difference)) / (1.0 + np.exp(-np.abs(difference))),
        )
        total += float(np.sum(sigmoid))
        slope = sigmoid * (1.0 - sigmoid) / temperature
        gradient_positive[start : start + chunk] = np.sum(slope, axis=1)
        # ``order`` is a permutation, so plain fancy indexing is safe here.
        gradient_negative[order] -= np.sum(slope, axis=0)

    pair_count = float(len(positive) * len(negative))
    loss = -total / pair_count
    derivative = np.empty(len(values), dtype=float)
    mask = np.asarray(labels) >= 0.5
    derivative[mask] = -gradient_positive / pair_count
    derivative[~mask] = -gradient_negative / pair_count
    return loss, derivative


def smooth_ks(
    probability: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    *,
    temperature: float = 0.05,
    grid: np.ndarray | None = None,
    sharpness: float | None = None,
) -> ObjectiveResult:
    """Negated smooth Kolmogorov-Smirnov statistic and its derivative.

    This is the objective the competition actually scores.  Balanced accuracy at
    a threshold is ``(1 + TPR(t) - FPR(t)) / 2``, and the submitted threshold is
    the one maximizing it on out-of-fold data, so the score is a monotone
    transform of ``max_t [TPR(t) - FPR(t)]`` -- the two-sample KS statistic.

    That distinguishes it from the two surrogates already here.
    ``soft_balanced_accuracy`` pins the threshold, so as T falls its gradient
    collapses onto the handful of rows beside that one threshold and overfits
    them.  ``smooth_auc`` keeps every row involved but optimizes the whole ROC
    curve, spending effort on operating points the score never uses.  Taking a
    soft maximum over a grid of thresholds keeps every row contributing at every
    threshold while still targeting only the best operating point.

    Indicators become ``sigmoid((p - t) / temperature)``; the maximum over the
    grid becomes a log-sum-exp with the given ``sharpness`` (default: the grid
    resolution), which is exact as sharpness grows.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    values = np.asarray(probability, dtype=float)
    mask = np.asarray(labels) >= 0.5
    positive_count = int(np.count_nonzero(mask))
    negative_count = len(values) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both classes must be present for KS.")
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    thresholds = np.asarray(grid, dtype=float)
    if sharpness is None:
        # Tie the softness of the max to the softness of the indicators so a
        # single annealing schedule sharpens both together.
        sharpness = 2.0 / temperature

    # rate[k] = TPR(t_k) - FPR(t_k) with smoothed indicators.
    difference = values[None, :] - thresholds[:, None]
    scaled = difference / temperature
    indicator = np.where(
        scaled >= 0.0,
        1.0 / (1.0 + np.exp(-np.abs(scaled))),
        np.exp(-np.abs(scaled)) / (1.0 + np.exp(-np.abs(scaled))),
    )
    row_weight = np.where(mask, 1.0 / positive_count, -1.0 / negative_count)
    rate = indicator @ row_weight

    # Soft maximum over thresholds, stabilized.
    shifted = sharpness * (rate - np.max(rate))
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    statistic = float(np.sum(weights * rate))

    # d(soft max)/d(rate_k) = w_k * (1 + sharpness * (rate_k - soft max))
    rate_derivative = weights * (1.0 + sharpness * (rate - statistic))
    slope = indicator * (1.0 - indicator) / temperature
    derivative = -(rate_derivative @ slope) * row_weight
    return -statistic, derivative


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
