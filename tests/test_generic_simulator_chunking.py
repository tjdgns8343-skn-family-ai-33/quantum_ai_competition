"""Chunking must not change any objective, including non-separable ones.

A first version called the objective once per 64-row chunk. That is only valid
for row-separable losses: smooth_auc scores pairs and smooth_ks maximizes over
the whole sample, and a chunk can hold a single class, which raised
"Both classes must be present for AUC" and killed a candidate sweep.
"""

from functools import partial

import numpy as np
import pytest

from qchallenge.candidates import build_candidate
from qchallenge.generic_simulator import value_and_gradient
from qchallenge.objectives import smooth_auc, smooth_ks, soft_balanced_accuracy

OBJECTIVES = {
    "balanced_bce": None,
    "smooth_auc": partial(smooth_auc, temperature=0.05),
    "smooth_ks": partial(smooth_ks, temperature=0.05),
    "soft_ba": partial(soft_balanced_accuracy, temperature=0.05),
}


@pytest.fixture
def problem():
    spec = build_candidate("a1b2")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 8))
    # Sorted labels put a single class inside early chunks, as real folds can.
    y = np.concatenate([np.zeros(150, dtype=int), np.ones(150, dtype=int)])
    weights = rng.uniform(-1, 1, spec.n_weights)
    return spec, x, y, weights


@pytest.mark.parametrize("name", sorted(OBJECTIVES))
def test_chunk_size_does_not_change_loss_or_gradient(problem, name):
    spec, x, y, weights = problem
    objective = OBJECTIVES[name]
    whole = value_and_gradient(spec, x, y, weights, objective, chunk_rows=len(x))
    chunked = value_and_gradient(spec, x, y, weights, objective, chunk_rows=32)
    assert abs(whole[0] - chunked[0]) < 1e-10
    assert np.max(np.abs(whole[1] - chunked[1])) < 1e-10


@pytest.mark.parametrize("name", sorted(OBJECTIVES))
def test_gradient_matches_finite_difference_under_chunking(problem, name):
    spec, x, y, weights = problem
    objective = OBJECTIVES[name]
    _, gradient = value_and_gradient(spec, x, y, weights, objective, chunk_rows=32)
    epsilon = 1e-6
    for index in (0, 5, spec.n_weights - 1):
        step = np.zeros(spec.n_weights)
        step[index] = epsilon
        plus = value_and_gradient(spec, x, y, weights + step, objective, chunk_rows=32)[0]
        minus = value_and_gradient(spec, x, y, weights - step, objective, chunk_rows=32)[0]
        assert abs(gradient[index] - (plus - minus) / (2 * epsilon)) < 1e-6
