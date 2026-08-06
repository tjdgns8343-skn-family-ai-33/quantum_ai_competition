"""The restart count must actually reach the optimizer from every entry point.

A first wiring pass patched screen and final training but silently missed the
cross-validation call site, so a run requested with four restarts quietly used
one.  The only visible symptom was a restart loss spread of exactly 0.0, which
is easy to read past.  These tests assert the plumbing end to end instead.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qchallenge.c1_training import (
    C1CrossValidationConfig,
    C1TrainConfig,
    run_c1_cross_validation,
    run_c1_screen,
)
from qchallenge.f1_training import F1TrainConfig, run_f1_screen

RESTARTS = 3


@pytest.fixture
def tiny_train_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = 60
    frame = pd.DataFrame(
        rng.normal(size=(rows, 8)), columns=[f"x{i}" for i in range(1, 9)]
    )
    frame["label"] = np.tile([0, 1], rows // 2)
    path = tmp_path / "public_train.csv"
    frame.to_csv(path, index=False)
    return path


def _restart_counts(payload: dict) -> list[int]:
    if "optimization" in payload:
        return [payload["optimization"]["restart_count"]]
    return [fold["optimization"]["restart_count"] for fold in payload["fold_results"]]


def test_c1_screen_uses_every_requested_restart(tiny_train_csv, tmp_path):
    run_dir = run_c1_screen(
        C1TrainConfig(
            train_csv=tiny_train_csv,
            artifacts_dir=tmp_path / "artifacts",
            maxiter=2,
            n_restarts=RESTARTS,
        )
    )
    payload = json.loads((run_dir / "screen_metrics.json").read_text(encoding="utf-8"))
    assert _restart_counts(payload) == [RESTARTS]
    assert len(payload["optimization"]["restarts"]) == RESTARTS
    assert len(set(payload["optimization"]["restart_seeds"])) == RESTARTS


def test_c1_cross_validation_uses_every_requested_restart(tiny_train_csv, tmp_path):
    run_dir = run_c1_cross_validation(
        C1CrossValidationConfig(
            train_csv=tiny_train_csv,
            artifacts_dir=tmp_path / "artifacts",
            maxiter=2,
            folds=2,
            n_restarts=RESTARTS,
        )
    )
    payload = json.loads(
        (run_dir / "cross_validation_metrics.json").read_text(encoding="utf-8")
    )
    assert _restart_counts(payload) == [RESTARTS, RESTARTS]


def test_f1_screen_uses_every_requested_restart(tiny_train_csv, tmp_path):
    run_dir = run_f1_screen(
        F1TrainConfig(
            train_csv=tiny_train_csv,
            artifacts_dir=tmp_path / "artifacts",
            n_blocks=1,
            maxiter=2,
            n_restarts=RESTARTS,
        )
    )
    payload = json.loads((run_dir / "screen_metrics.json").read_text(encoding="utf-8"))
    assert _restart_counts(payload) == [RESTARTS]


def test_distinct_restart_seeds_produce_distinct_initial_points(tiny_train_csv, tmp_path):
    """Guards against restarts that are wired but all start from the same point."""
    run_dir = run_c1_screen(
        C1TrainConfig(
            train_csv=tiny_train_csv,
            artifacts_dir=tmp_path / "artifacts",
            maxiter=2,
            n_restarts=RESTARTS,
        )
    )
    payload = json.loads((run_dir / "screen_metrics.json").read_text(encoding="utf-8"))
    initial_losses = [
        record["initial_loss"] for record in payload["optimization"]["restarts"]
    ]
    assert len(set(initial_losses)) == RESTARTS
