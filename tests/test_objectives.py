import numpy as np
import pytest

from qchallenge.metrics import balanced_accuracy
from qchallenge.objectives import (
    balanced_sample_weights,
    smooth_auc,
    smooth_ks,
    soft_balanced_accuracy,
)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 1200)
    probability = np.clip(
        0.5 + 0.22 * (2 * labels - 1) + rng.normal(0, 0.25, 1200), 1e-4, 1 - 1e-4
    )
    return probability, labels, balanced_sample_weights(labels)


@pytest.mark.parametrize(
    "objective", [soft_balanced_accuracy, smooth_auc, smooth_ks]
)
def test_gradient_matches_finite_difference(scores, objective):
    probability, labels, weight = scores
    _, derivative = objective(probability, labels, weight, temperature=0.05)
    epsilon = 1e-7
    for index in (0, 11, 600, 1199):
        plus, minus = probability.copy(), probability.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite = (
            objective(plus, labels, weight, temperature=0.05)[0]
            - objective(minus, labels, weight, temperature=0.05)[0]
        ) / (2 * epsilon)
        assert abs(derivative[index] - finite) < 1e-6


def test_smooth_auc_is_exact_against_the_full_pair_sum(scores):
    probability, labels, weight = scores
    loss, _ = smooth_auc(probability, labels, weight, temperature=0.05)
    positive = probability[labels == 1][:, None]
    negative = probability[labels == 0][None, :]
    brute = -np.mean(1.0 / (1.0 + np.exp(-(positive - negative) / 0.05)))
    assert abs(loss - brute) < 1e-12


def test_smooth_ks_approaches_the_best_threshold_balanced_accuracy(scores):
    """KS is a monotone transform of max-over-threshold balanced accuracy."""
    probability, labels, weight = scores
    grid = np.linspace(0.05, 0.95, 91)
    true_best = max(
        balanced_accuracy(labels, (probability >= t).astype(int)) for t in grid
    )
    implied = [
        (1.0 - smooth_ks(probability, labels, weight, temperature=t)[0]) / 2.0
        for t in (0.10, 0.05, 0.02, 0.01)
    ]
    assert implied == sorted(implied), "sharpening must not move away from the metric"
    assert abs(implied[-1] - true_best) < 0.02


def test_objectives_reject_a_single_class(scores):
    probability, labels, weight = scores
    ones = np.ones_like(labels)
    for objective in (smooth_auc, smooth_ks):
        with pytest.raises(ValueError):
            objective(probability, ones, weight)
