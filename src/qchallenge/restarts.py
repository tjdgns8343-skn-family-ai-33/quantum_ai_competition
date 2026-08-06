"""Label-independent multi-restart driver for circuit-native training.

Screening showed that repeated runs of the same architecture vary far more than
different architectures do: identical C1 settings scored 0.772 to 0.836 across
splits, and one F1 run collapsed to 0.686.  That is local-minimum noise, not an
architecture property, so a single random start is not a reliable estimate of
what a circuit can do.

Each restart begins from its own label-independent random point and is scored
only by the training objective it just minimized.  No validation, out-of-fold,
public-test or label statistic beyond the training loss takes part in the
choice, so a restart used inside a cross-validation fold cannot leak.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

ValueAndGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


def restart_seeds(seed: int, n_restarts: int, stride: int = 1) -> list[int]:
    """Derive restart seeds from a base seed with no reference to labels."""
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1.")
    if stride < 1:
        raise ValueError("restart seed stride must be at least 1.")
    return [seed + stride * index for index in range(n_restarts)]


def multistart_minimize(
    value_and_gradient: ValueAndGradient,
    initial_points: Sequence[np.ndarray],
    *,
    bounds: list[tuple[float, float]],
    maxiter: int,
    record_history_for_best: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Minimize from every initial point and keep the lowest training loss.

    Returns the best weights, the initial point that produced them, and a
    summary describing every restart.
    """
    if not initial_points:
        raise ValueError("multistart_minimize needs at least one initial point.")

    best_loss = float("inf")
    best_weights: np.ndarray | None = None
    best_initial: np.ndarray | None = None
    best_history: list[dict] = []
    best_index = -1
    records: list[dict] = []

    for restart_index, initial in enumerate(initial_points):
        history: list[dict] = []
        observed_loss = float("inf")
        observed_weights = np.asarray(initial, dtype=float).copy()

        def tracked(weight_values: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal observed_loss, observed_weights
            value, gradient = value_and_gradient(weight_values)
            history.append(
                {
                    "evaluation": len(history) + 1,
                    "loss": float(value),
                    "gradient_norm": float(np.linalg.norm(gradient)),
                }
            )
            if value < observed_loss:
                observed_loss = float(value)
                observed_weights = np.asarray(weight_values, dtype=float).copy()
            return float(value), gradient

        initial_loss, initial_gradient = value_and_gradient(initial)
        result = minimize(
            tracked,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "maxls": 30, "ftol": 1e-10, "gtol": 1e-7},
        )
        records.append(
            {
                "restart": restart_index + 1,
                "initial_loss": float(initial_loss),
                "initial_gradient_norm": float(np.linalg.norm(initial_gradient)),
                "terminal_loss": float(result.fun),
                "best_observed_loss": float(observed_loss),
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "gradient_evaluations": int(result.njev),
            }
        )
        if observed_loss < best_loss:
            best_loss = observed_loss
            best_weights = observed_weights
            best_initial = np.asarray(initial, dtype=float).copy()
            best_history = history
            best_index = restart_index

    assert best_weights is not None and best_initial is not None
    losses = [record["best_observed_loss"] for record in records]
    summary = {
        "restart_selection": "lowest_training_objective_only",
        "restart_count": len(records),
        "selected_restart": best_index + 1,
        "best_observed_loss": float(best_loss),
        "worst_restart_loss": float(np.max(losses)),
        "median_restart_loss": float(np.median(losses)),
        "restart_loss_spread": float(np.max(losses) - np.min(losses)),
        "restarts": records,
        "history": best_history if record_history_for_best else [],
    }
    return best_weights, best_initial, summary
