"""Quantum-only beam search over the number and identity of raw features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .data import load_raw_train
from .metrics import balanced_accuracy, balanced_binary_cross_entropy
from .training import (
    exact_quantum_probabilities,
    make_exact_qnn,
    optimize_quantum_loss,
    stratified_holdout,
)


@dataclass(frozen=True)
class FeatureSearchConfig:
    train_csv: Path
    artifacts_dir: Path = Path("artifacts")
    seed: int = 2026
    init_scale: float = 0.05
    maxiter: int = 24
    screen_rows: int = 1200
    validation_fraction: float = 0.25
    beam_width: int = 3
    max_features: int = 5
    feature_layout: str = "sequential"


def _stratified_subsample(y: np.ndarray, rows: int, seed: int) -> np.ndarray:
    if rows <= 0:
        raise ValueError("screen_rows must be positive.")
    if rows >= len(y):
        return np.arange(len(y), dtype=int)
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    remaining = rows
    for position, label in enumerate((0, 1)):
        indices = np.flatnonzero(y == label)
        if position == 0:
            count = int(round(rows * len(indices) / len(y)))
            count = min(max(1, count), len(indices))
            remaining -= count
        else:
            count = min(max(1, remaining), len(indices))
        parts.append(rng.choice(indices, size=count, replace=False))
    return np.sort(np.concatenate(parts))


def _exact_metrics(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    selected: tuple[int, ...],
    config: FeatureSearchConfig,
) -> dict:
    qnn = make_exact_qnn(
        config.seed, config.feature_layout, selected, True
    )
    probability = exact_quantum_probabilities(qnn, x[:, selected], weights)
    return {
        "balanced_accuracy_at_0_5": balanced_accuracy(
            y, (probability >= 0.5).astype(int)
        ),
        "balanced_binary_cross_entropy": balanced_binary_cross_entropy(
            y, probability
        ),
    }


def run_feature_search(config: FeatureSearchConfig) -> Path:
    if config.beam_width <= 0:
        raise ValueError("beam_width must be positive.")
    if not 1 <= config.max_features <= 5:
        raise ValueError(
            "max_features must be between 1 and 5 because only five B1 data slots affect q0."
        )
    if config.maxiter < 18:
        raise ValueError(
            "maxiter must be at least 18 for the 16-parameter COBYLA screen."
        )

    x, y, data_info = load_raw_train(config.train_csv)
    sampled = _stratified_subsample(y, config.screen_rows, config.seed + 1000)
    screen_x, screen_y = x[sampled], y[sampled]
    fit, valid = stratified_holdout(
        screen_y, config.validation_fraction, config.seed
    )
    cache: dict[tuple[int, ...], dict] = {}

    def evaluate(selected: tuple[int, ...]) -> dict:
        selected = tuple(sorted(selected))
        if selected in cache:
            return cache[selected]
        weights, optimization = optimize_quantum_loss(
            screen_x[np.ix_(fit, selected)],
            screen_y[fit],
            seed=config.seed,
            init_scale=config.init_scale,
            maxiter=config.maxiter,
            shots=1024,
            feature_layout=config.feature_layout,
            selected_features=selected,
            pack_selected_features=True,
        )
        result = {
            "feature_count": len(selected),
            "features_zero_based": list(selected),
            "features_csv": [f"x{index + 1}" for index in selected],
            "fit": _exact_metrics(
                screen_x[fit], screen_y[fit], weights, selected, config
            ),
            "validation": _exact_metrics(
                screen_x[valid], screen_y[valid], weights, selected, config
            ),
            "optimization": {
                "evaluations": optimization["evaluations"],
                "initial_loss": optimization["initial_loss"],
                "final_loss": optimization["final_loss"],
                "best_observed_loss": optimization["best_observed_loss"],
            },
        }
        cache[selected] = result
        return result

    beam: list[tuple[int, ...]] = [tuple()]
    best_by_count: dict[int, dict] = {}
    for target_count in range(1, config.max_features + 1):
        candidate_sets = {
            (*selected, added)
            for selected in beam
            for added in range(8)
            if added not in selected
        }
        rows = [evaluate(selected) for selected in sorted(candidate_sets)]
        rows.sort(
            key=lambda row: (
                -row["validation"]["balanced_accuracy_at_0_5"],
                row["validation"]["balanced_binary_cross_entropy"],
                row["features_zero_based"],
            )
        )
        beam = [
            tuple(row["features_zero_based"])
            for row in rows[: config.beam_width]
        ]
        best_by_count[target_count] = rows[0]

    ranked = sorted(
        cache.values(),
        key=lambda row: (
            -row["validation"]["balanced_accuracy_at_0_5"],
            row["validation"]["balanced_binary_cross_entropy"],
            row["feature_count"],
            row["features_zero_based"],
        ),
    )
    payload = {
        "mode": "quantum_only_forward_packed_feature_beam_search",
        "submission_created": False,
        "config": {
            **asdict(config),
            "train_csv": str(config.train_csv),
            "artifacts_dir": str(config.artifacts_dir),
        },
        "data": data_info,
        "screen_sample_rows": int(len(sampled)),
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(valid)),
        "candidate_count": len(cache),
        "same_initialization_for_every_candidate": True,
        "active_slot_order": ["RY_q0", "RY_q1", "RY_q2", "RY_q3", "RZ_q0"],
        "feature_order_is_part_of_candidate": True,
        "all_candidates_packed_into_causally_active_slots": True,
        "classical_predictive_model_used": False,
        "best_by_feature_count": {
            str(count): best_by_count[count] for count in sorted(best_by_count)
        },
        "top_candidates": ranked[: min(20, len(ranked))],
    }
    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_"
        f"feature_beam_{config.feature_layout}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "feature_search_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return run_dir
