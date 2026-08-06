"""Reporting-only evaluation of a trained artifact on a labelled CSV.

The competition rules permit the public test set to be used to *test* a trained
circuit but forbid using any part of it for training.  This module therefore
only measures and records; nothing here feeds initialization, the objective, the
optimizer, threshold selection or candidate selection.  Those stay train-only,
so a public-test number produced here is an independent generalization check
rather than a quantity anything was fitted to.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .c1_circuit import C1_ARCHITECTURE
from .c1_simulator import c1_exact_probabilities
from .data import load_raw_train
from .f1_circuit import F1_ARCHITECTURE
from .f1_simulator import f1_exact_probabilities
from .metrics import (
    balanced_accuracy,
    balanced_binary_cross_entropy,
    binary_cross_entropy,
)

REPORTING_ONLY_NOTE = (
    "Measured for reporting only. Training, threshold selection and candidate "
    "selection use public_train.csv exclusively."
)


def _probability_function(architecture: str, n_blocks: int | None):
    if architecture == C1_ARCHITECTURE:
        return lambda x, w: c1_exact_probabilities(x, w)
    if architecture == F1_ARCHITECTURE:
        if n_blocks is None:
            raise ValueError("F1 evaluation requires the block count.")
        return lambda x, w: f1_exact_probabilities(x, w, n_blocks)
    raise ValueError(f"Unsupported architecture for evaluation: {architecture}")


def load_artifact_weights(directory: Path) -> tuple[np.ndarray, str, int | None, float]:
    """Read weights, architecture, block count and threshold from an artifact."""
    weights_path = directory / "weights.json"
    if weights_path.is_file():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        threshold = float(payload.get("threshold", 0.5))
        theta = [
            payload[key]
            for key in sorted(
                (name for name in payload if name.startswith("theta_")),
                key=lambda name: int(name.split("_")[1]),
            )
        ]
        provenance = json.loads(
            (directory / "parameter_provenance.json").read_text(encoding="utf-8")
        )
        architecture = str(provenance["architecture"])
        metrics_path = directory / "training_metrics.json"
        n_blocks = None
        if metrics_path.is_file():
            n_blocks = json.loads(metrics_path.read_text(encoding="utf-8")).get(
                "n_blocks"
            )
        return np.asarray(theta, dtype=float), architecture, n_blocks, threshold

    screen_path = directory / "screen_metrics.json"
    if screen_path.is_file():
        payload = json.loads(screen_path.read_text(encoding="utf-8"))
        return (
            np.asarray(payload["selected_weights"], dtype=float),
            str(payload["architecture"]),
            payload.get("n_blocks"),
            float(payload.get("config", {}).get("decision_threshold", 0.5)),
        )
    raise FileNotFoundError(f"No weights.json or screen_metrics.json in {directory}")


def evaluate_artifact_on_csv(
    directory: Path,
    csv_path: Path,
    *,
    shots: int = 1024,
    shot_seed: int = 20260806,
    threshold: float | None = None,
) -> dict:
    weights, architecture, n_blocks, artifact_threshold = load_artifact_weights(
        directory
    )
    decision_threshold = artifact_threshold if threshold is None else threshold
    x, y, data_info = load_raw_train(csv_path)
    probability = _probability_function(architecture, n_blocks)(x, weights)
    rng = np.random.default_rng(shot_seed)
    shot_probability = rng.binomial(shots, probability) / shots
    return {
        "purpose": "reporting_only_generalization_check",
        "note": REPORTING_ONLY_NOTE,
        "used_for_training": False,
        "used_for_threshold_selection": False,
        "used_for_candidate_selection": False,
        "artifact_dir": str(directory),
        "architecture": architecture,
        "n_blocks": n_blocks,
        "weight_count": int(len(weights)),
        "threshold": decision_threshold,
        "data": data_info,
        "rows": int(len(y)),
        "exact_balanced_accuracy": balanced_accuracy(
            y, (probability >= decision_threshold).astype(int)
        ),
        "shot_balanced_accuracy": balanced_accuracy(
            y, (shot_probability >= decision_threshold).astype(int)
        ),
        "exact_binary_cross_entropy": binary_cross_entropy(y, probability),
        "exact_balanced_binary_cross_entropy": balanced_binary_cross_entropy(
            y, probability
        ),
        "mean_exact_probability": float(np.mean(probability)),
        "rows_within_two_shot_sigma_of_threshold": int(
            np.sum(
                np.abs(probability - decision_threshold)
                < 2.0 * np.sqrt(probability * (1.0 - probability) / shots)
            )
        ),
        "shots": shots,
        "shot_seed": shot_seed,
    }
