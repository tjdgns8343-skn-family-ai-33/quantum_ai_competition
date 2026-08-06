import numpy as np
import pytest

from qchallenge.restarts import multistart_minimize, restart_seeds


def test_restart_seeds_are_deterministic_and_distinct():
    assert restart_seeds(2026, 4, 1000) == [2026, 3026, 4026, 5026]
    assert restart_seeds(2026, 1) == [2026]
    assert restart_seeds(2026, 3) == restart_seeds(2026, 3)
    with pytest.raises(ValueError):
        restart_seeds(2026, 0)
    with pytest.raises(ValueError):
        restart_seeds(2026, 2, 0)


def _double_well(point: np.ndarray) -> tuple[float, np.ndarray]:
    """Two minima: a shallow one near -2 and the global one near +2."""
    x = float(point[0])
    value = (x**2 - 4.0) ** 2 + 0.5 * x
    gradient = np.array([4.0 * x * (x**2 - 4.0) + 0.5])
    return value, gradient


_STARTS = [np.array([value]) for value in (-2.0, 0.5, 3.0)]


def test_multistart_never_does_worse_than_its_best_single_start():
    """The contract: combining starts returns the best of them, whatever the geometry."""
    bounds = [(-5.0, 5.0)]
    individual = [
        multistart_minimize(_double_well, [start], bounds=bounds, maxiter=200)[2][
            "best_observed_loss"
        ]
        for start in _STARTS
    ]
    best, initial, summary = multistart_minimize(
        _double_well, _STARTS, bounds=bounds, maxiter=200
    )
    assert min(individual) < max(individual), "starts must land in different minima"
    assert summary["best_observed_loss"] == pytest.approx(min(individual))
    assert summary["selected_restart"] == int(np.argmin(individual)) + 1
    assert initial[0] == _STARTS[int(np.argmin(individual))][0]
    assert _double_well(best)[0] == pytest.approx(min(individual))
    assert summary["restart_loss_spread"] > 0


def test_multistart_reports_every_restart_and_selects_by_training_loss_only():
    _, _, summary = multistart_minimize(
        _double_well, _STARTS, bounds=[(-5.0, 5.0)], maxiter=200
    )
    assert summary["restart_selection"] == "lowest_training_objective_only"
    assert len(summary["restarts"]) == len(_STARTS)
    losses = [record["best_observed_loss"] for record in summary["restarts"]]
    assert summary["best_observed_loss"] == pytest.approx(min(losses))
    assert summary["worst_restart_loss"] == pytest.approx(max(losses))
    assert summary["history"], "the winning restart keeps its loss history"


def test_multistart_requires_at_least_one_initial_point():
    with pytest.raises(ValueError):
        multistart_minimize(_double_well, [], bounds=[(-5.0, 5.0)], maxiter=10)


def test_single_restart_matches_the_previous_single_start_behaviour():
    weights, initial, summary = multistart_minimize(
        _double_well, [np.array([3.0])], bounds=[(-5.0, 5.0)], maxiter=200
    )
    assert summary["restart_count"] == 1
    assert summary["selected_restart"] == 1
    assert initial[0] == 3.0
    assert summary["restart_loss_spread"] == 0.0
    value, _ = _double_well(weights)
    assert value == pytest.approx(summary["best_observed_loss"])
