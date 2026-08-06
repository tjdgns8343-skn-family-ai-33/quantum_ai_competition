"""End-to-end training for the compliant C1 quantum circuit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .objectives import (
    balanced_sample_weights,
    smooth_auc,
    soft_balanced_accuracy,
    temperature_schedule,
)
from .restarts import multistart_minimize, restart_seeds
from .c1_circuit import (
    C1_ARCHITECTURE,
    C1_N_FEATURES,
    C1_N_WEIGHTS,
    C1_REUPLOAD_BLOCKS,
    build_c1_unitary,
    export_c1_submission_qasm,
)
from .c1_simulator import (
    c1_balanced_bce_value_and_gradient,
    c1_exact_probabilities,
    c1_value_and_gradient,
)
from .data import load_raw_train
from .encoding_note import build_encoding_note
from .metrics import (
    balanced_accuracy,
    balanced_binary_cross_entropy,
    binary_cross_entropy,
)
from .training import FIXED_THRESHOLD, stratified_holdout


@dataclass(frozen=True)
class C1TrainConfig:
    train_csv: Path
    artifacts_dir: Path = Path("artifacts")
    seed: int = 2026
    split_seed: int | None = None
    init_scale: float = 0.05
    affine_scale_jitter: float = 0.1
    maxiter: int = 80
    n_restarts: int = 1
    shots: int = 1024
    validation_fraction: float = 0.2
    max_rows: int | None = None
    decision_threshold: float = FIXED_THRESHOLD
    objective: str = "balanced_bce"
    temperature_start: float = 0.30
    temperature_stop: float = 0.015
    anneal_stages: int = 10
    stage_maxiter: int = 40
    # Which annealing stage to submit.  Annealing is a path, not a point: the
    # cross-validation selected "stage k of this schedule", so the final run
    # replays the same schedule and takes the weights from that stage rather
    # than re-annealing straight to the chosen temperature.
    select_stage: int | None = None


@dataclass(frozen=True)
class C1CrossValidationConfig:
    train_csv: Path
    artifacts_dir: Path = Path("artifacts")
    split_seed: int = 2026
    init_seed: int = 2026
    init_scale: float = 0.05
    affine_scale_jitter: float = 0.1
    maxiter: int = 80
    n_restarts: int = 1
    folds: int = 5
    shots: int = 1024
    objective: str = "balanced_bce"
    temperature_start: float = 0.20
    temperature_stop: float = 0.02
    anneal_stages: int = 6
    stage_maxiter: int = 50


def c1_random_initial_point(
    seed: int, *, local_scale: float, affine_scale_jitter: float
) -> np.ndarray:
    """Label-independent initialization around direct raw-angle encoding."""
    if local_scale < 0 or affine_scale_jitter < 0:
        raise ValueError("C1 initialization scales must be non-negative.")
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-local_scale, local_scale, C1_N_WEIGHTS)
    for block_index in range(len(C1_REUPLOAD_BLOCKS)):
        offset = 16 * block_index
        weights[offset : offset + 4] = rng.uniform(
            1.0 - affine_scale_jitter,
            1.0 + affine_scale_jitter,
            4,
        )
    return weights


def _stratified_subsample(
    y: np.ndarray, max_rows: int | None, seed: int
) -> np.ndarray:
    if max_rows is None or max_rows >= len(y):
        return np.arange(len(y), dtype=int)
    if max_rows < 20:
        raise ValueError("C1 max_rows must be at least 20.")
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    remaining = max_rows
    for position, label in enumerate((0, 1)):
        indices = np.flatnonzero(y == label)
        if position == 0:
            count = int(round(max_rows * len(indices) / len(y)))
            count = max(1, min(count, len(indices)))
            remaining -= count
        else:
            count = max(1, min(remaining, len(indices)))
        parts.append(rng.choice(indices, size=count, replace=False))
    return np.sort(np.concatenate(parts))


def _c1_bounds() -> list[tuple[float, float]]:
    bounds = [(-np.pi, np.pi) for _ in range(C1_N_WEIGHTS)]
    for block_index in range(len(C1_REUPLOAD_BLOCKS)):
        offset = 16 * block_index
        for qubit in range(4):
            bounds[offset + qubit] = (-3.0, 3.0)
    return bounds


def optimize_c1_quantum_loss(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    init_scale: float,
    affine_scale_jitter: float,
    maxiter: int,
    n_restarts: int = 1,
    restart_seed_stride: int = 1000,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Optimize analytic gradients of the C1 statevector probability only."""
    seeds = restart_seeds(seed, n_restarts, restart_seed_stride)
    initial_points = [
        c1_random_initial_point(
            restart_seed,
            local_scale=init_scale,
            affine_scale_jitter=affine_scale_jitter,
        )
        for restart_seed in seeds
    ]

    def value_and_gradient(weight_values: np.ndarray) -> tuple[float, np.ndarray]:
        return c1_balanced_bce_value_and_gradient(x, y, weight_values)

    weights, initial, summary = multistart_minimize(
        value_and_gradient,
        initial_points,
        bounds=_c1_bounds(),
        maxiter=maxiter,
    )
    return weights, {
        "optimizer": "L-BFGS-B_with_exact_adjoint_gradient",
        "training_emulator": "vectorized_exact_4_qubit_statevector",
        "qiskit_equivalence_required": True,
        "objective": "balanced_binary_cross_entropy_of_C1_q0_probability",
        "restart_seeds": seeds,
        **summary,
    }, initial


def optimize_c1_annealed(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    init_scale: float,
    affine_scale_jitter: float,
    maxiter: int,
    surrogate: str = "soft_balanced_accuracy",
    n_restarts: int = 1,
    threshold: float = FIXED_THRESHOLD,
    temperature_start: float = 0.20,
    temperature_stop: float = 0.02,
    anneal_stages: int = 5,
    stage_maxiter: int = 60,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Warm up on balanced BCE, then anneal a surrogate of the scored metric.

    Both surrogates are flat or nearly flat somewhere, so optimizing either from
    a random start would stall; starting from the cross-entropy solution and
    lowering the temperature avoids that.  Every stage still reads only circuit
    probabilities and raw train labels.

    ``soft_balanced_accuracy`` converges to the scored metric but concentrates
    on the shrinking set of rows within ``T`` of the threshold, which is why it
    overfits them.  ``smooth_auc`` instead scores every positive/negative pair,
    so the gradient stays spread across all rows and no threshold is involved --
    the threshold is chosen afterwards from out-of-fold probabilities.
    """
    if surrogate not in ("soft_balanced_accuracy", "smooth_auc"):
        raise ValueError(f"Unknown C1 surrogate: {surrogate!r}")
    weights, warm_start_summary, initial = optimize_c1_quantum_loss(
        x,
        y,
        seed=seed,
        init_scale=init_scale,
        affine_scale_jitter=affine_scale_jitter,
        maxiter=maxiter,
        n_restarts=n_restarts,
    )
    sample_weight = balanced_sample_weights(np.asarray(y, dtype=int))
    stages: list[dict] = []
    # Stage 0 is the cross-entropy warm start, so a caller can read the whole
    # temperature curve -- including "no annealing at all" -- from one run and
    # pick the stop temperature on out-of-fold data rather than on train.
    stage_weights: list[list[float]] = [[float(v) for v in weights]]
    stage_temperatures: list[float | None] = [None]

    def build_objective(temperature: float):
        if surrogate == "smooth_auc":
            return partial(smooth_auc, temperature=temperature)
        return partial(
            soft_balanced_accuracy, threshold=threshold, temperature=temperature
        )

    def surrogate_value(weight_values: np.ndarray, temperature: float) -> float:
        probability = c1_exact_probabilities(x, weight_values)
        loss, _ = build_objective(temperature)(
            probability, np.asarray(y, dtype=float), sample_weight
        )
        return -loss

    for temperature in temperature_schedule(
        temperature_start, temperature_stop, anneal_stages
    ):
        objective = build_objective(temperature)
        before = balanced_accuracy(
            y, (c1_exact_probabilities(x, weights) >= threshold).astype(int)
        )
        result = minimize(
            lambda values: c1_value_and_gradient(x, y, values, objective),
            weights,
            method="L-BFGS-B",
            jac=True,
            bounds=_c1_bounds(),
            options={"maxiter": stage_maxiter, "maxls": 30, "ftol": 1e-12},
        )
        weights = np.asarray(result.x, dtype=float)
        stages.append(
            {
                "temperature": float(temperature),
                "surrogate_value": surrogate_value(weights, temperature),
                "train_balanced_accuracy_before": before,
                "train_balanced_accuracy_after": balanced_accuracy(
                    y, (c1_exact_probabilities(x, weights) >= threshold).astype(int)
                ),
                "iterations": int(result.nit),
                "message": str(result.message),
            }
        )
        stage_weights.append([float(value) for value in weights])
        stage_temperatures.append(float(temperature))

    return weights, {
        "optimizer": "L-BFGS-B_with_exact_adjoint_gradient",
        "training_emulator": "vectorized_exact_4_qubit_statevector",
        "qiskit_equivalence_required": True,
        "objective": f"balanced_bce_warm_start_then_annealed_{surrogate}",
        "anneal_threshold": float(threshold),
        "anneal_stages": stages,
        "stage_weights": stage_weights,
        "stage_temperatures": stage_temperatures,
        "warm_start": warm_start_summary,
        "history": warm_start_summary.get("history", []),
        "best_observed_loss": warm_start_summary.get("best_observed_loss"),
        "restart_count": warm_start_summary.get("restart_count", n_restarts),
    }, initial


def _metrics(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    shots: int,
    seed: int,
    threshold: float = FIXED_THRESHOLD,
) -> dict:
    exact = c1_exact_probabilities(x, weights)
    rng = np.random.default_rng(seed)
    shot_probability = rng.binomial(shots, exact) / shots
    exact_prediction = (exact >= threshold).astype(int)
    shot_prediction = (shot_probability >= threshold).astype(int)
    return {
        "rows": int(len(y)),
        "exact_balanced_accuracy_at_threshold": balanced_accuracy(
            y, exact_prediction
        ),
        "shot_balanced_accuracy_at_threshold": balanced_accuracy(
            y, shot_prediction
        ),
        "exact_binary_cross_entropy": binary_cross_entropy(y, exact),
        "exact_balanced_binary_cross_entropy": balanced_binary_cross_entropy(
            y, exact
        ),
        "shot_binary_cross_entropy": binary_cross_entropy(y, shot_probability),
        "mean_exact_probability": float(np.mean(exact)),
        "shots": shots,
        "shot_seed": seed,
        "threshold": threshold,
    }


def c1_qiskit_equivalence_report(
    x: np.ndarray, weights: np.ndarray, *, max_rows: int = 16
) -> dict:
    sample = np.asarray(x[:max_rows], dtype=float)
    custom = c1_exact_probabilities(sample, weights)
    circuit, features, parameters = build_c1_unitary()
    qiskit_probability: list[float] = []
    for row in sample:
        bindings = {
            **dict(zip(features, row, strict=True)),
            **dict(zip(parameters, weights, strict=True)),
        }
        state = Statevector.from_instruction(circuit.assign_parameters(bindings))
        qiskit_probability.append(float(state.probabilities([0])[1]))
    difference = np.abs(custom - np.asarray(qiskit_probability))
    return {
        "rows": int(len(sample)),
        "maximum_absolute_probability_difference": float(np.max(difference)),
        "passes_at_1e_10": bool(np.max(difference) <= 1e-10),
    }


def c1_feature_causal_report(
    x: np.ndarray, weights: np.ndarray, *, perturbation: float = 1e-5
) -> dict:
    sample = np.asarray(x[: min(128, len(x))], dtype=float)
    rows: list[dict] = []
    for feature_index in range(C1_N_FEATURES):
        plus = sample.copy()
        minus = sample.copy()
        plus[:, feature_index] += perturbation
        minus[:, feature_index] -= perturbation
        sensitivity = np.abs(
            (c1_exact_probabilities(plus, weights) - c1_exact_probabilities(minus, weights))
            / (2.0 * perturbation)
        )
        rows.append(
            {
                "feature": f"x{feature_index + 1}",
                "mean_absolute_dp_dx": float(np.mean(sensitivity)),
                "maximum_absolute_dp_dx": float(np.max(sensitivity)),
            }
        )
    return {
        "perturbation": perturbation,
        "rows": rows,
        "all_features_nonzero_at_1e_10": all(
            row["maximum_absolute_dp_dx"] > 1e-10 for row in rows
        ),
    }


def _config_payload(config: C1TrainConfig) -> dict:
    return {
        **asdict(config),
        "train_csv": str(config.train_csv),
        "artifacts_dir": str(config.artifacts_dir),
    }


def run_c1_screen(config: C1TrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    split_seed = config.seed if config.split_seed is None else config.split_seed
    selected_rows = _stratified_subsample(y, config.max_rows, split_seed)
    screen_x, screen_y = x[selected_rows], y[selected_rows]
    fit, valid = stratified_holdout(
        screen_y, config.validation_fraction, split_seed
    )
    weights, optimization, initial = optimize_c1_quantum_loss(
        screen_x[fit],
        screen_y[fit],
        seed=config.seed,
        init_scale=config.init_scale,
        affine_scale_jitter=config.affine_scale_jitter,
        maxiter=config.maxiter,
        n_restarts=config.n_restarts,
    )
    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_screen_c1_"
        f"split{split_seed}_init{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "mode": "train_only_holdout_C1_direct_quantum_gradient_screen",
        "architecture": C1_ARCHITECTURE,
        "submission_created": False,
        "split_seed": split_seed,
        "init_seed": config.seed,
        "config": _config_payload(config),
        "data": data_info,
        "screen_rows": int(len(screen_x)),
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(valid)),
        "optimizer_uses_only_quantum_probability_loss": True,
        "classical_predictive_model_used": False,
        "optimization": optimization,
        "fit_metrics": _metrics(
            screen_x[fit], screen_y[fit], weights, shots=config.shots, seed=split_seed + 1,
            threshold=config.decision_threshold,
        ),
        "validation_metrics": _metrics(
            screen_x[valid], screen_y[valid], weights, shots=config.shots, seed=split_seed + 2,
            threshold=config.decision_threshold,
        ),
        "qiskit_equivalence": c1_qiskit_equivalence_report(screen_x[valid], weights),
        "initial_feature_causal_report": c1_feature_causal_report(screen_x, initial),
        "trained_feature_causal_report": c1_feature_causal_report(screen_x, weights),
        "selected_weights": [float(value) for value in weights],
    }
    (run_dir / "screen_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return run_dir


def _stratified_folds(
    y: np.ndarray, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("C1 cross-validation requires at least two folds.")
    rng = np.random.default_rng(seed)
    validation_parts: list[list[np.ndarray]] = [[] for _ in range(folds)]
    for label in (0, 1):
        shuffled = rng.permutation(np.flatnonzero(y == label))
        for fold_index, part in enumerate(np.array_split(shuffled, folds)):
            validation_parts[fold_index].append(part)
    all_indices = np.arange(len(y), dtype=int)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for parts in validation_parts:
        valid = np.sort(np.concatenate(parts))
        fit_mask = np.ones(len(y), dtype=bool)
        fit_mask[valid] = False
        result.append((all_indices[fit_mask], valid))
    return result


def run_c1_cross_validation(config: C1CrossValidationConfig) -> Path:
    """Generate genuine C1 OOF probabilities and a train-only threshold."""
    x, y, data_info = load_raw_train(config.train_csv)
    splits = _stratified_folds(y, config.folds, config.split_seed)
    oof_probability = np.full(len(y), np.nan, dtype=float)
    fold_results: list[dict] = []
    stage_oof: list[np.ndarray] | None = None
    stage_temperatures: list[float | None] = []
    equivalence_reports: list[dict] = []

    for fold_index, (fit, valid) in enumerate(splits):
        shared = dict(
            seed=config.init_seed,
            init_scale=config.init_scale,
            affine_scale_jitter=config.affine_scale_jitter,
            maxiter=config.maxiter,
            n_restarts=config.n_restarts,
        )
        if config.objective == "balanced_bce":
            weights, optimization, _ = optimize_c1_quantum_loss(
                x[fit], y[fit], **shared
            )
        elif config.objective in ("soft_balanced_accuracy", "smooth_auc"):
            weights, optimization, _ = optimize_c1_annealed(
                x[fit],
                y[fit],
                surrogate=config.objective,
                temperature_start=config.temperature_start,
                temperature_stop=config.temperature_stop,
                anneal_stages=config.anneal_stages,
                stage_maxiter=config.stage_maxiter,
                **shared,
            )
        else:
            raise ValueError(f"Unknown C1 objective: {config.objective!r}")
        if "stage_weights" in optimization:
            if stage_oof is None:
                stage_oof = [
                    np.full(len(y), np.nan, dtype=float)
                    for _ in optimization["stage_weights"]
                ]
                stage_temperatures = optimization["stage_temperatures"]
            for position, values in enumerate(optimization["stage_weights"]):
                stage_oof[position][valid] = c1_exact_probabilities(
                    x[valid], np.asarray(values, dtype=float)
                )
            # The per-fold weight dump is large and already summarized.
            optimization = {
                key: value
                for key, value in optimization.items()
                if key != "stage_weights"
            }
        probability = c1_exact_probabilities(x[valid], weights)
        oof_probability[valid] = probability
        fold_results.append(
            {
                "fold": fold_index + 1,
                "fit_rows": int(len(fit)),
                "validation_rows": int(len(valid)),
                "balanced_accuracy_at_0_5": balanced_accuracy(
                    y[valid], probability >= 0.5
                ),
                "balanced_binary_cross_entropy": balanced_binary_cross_entropy(
                    y[valid], probability
                ),
                "optimization": optimization,
            }
        )
        equivalence_reports.append(c1_qiskit_equivalence_report(x[valid], weights))

    if not np.isfinite(oof_probability).all():
        raise RuntimeError("C1 OOF generation left one or more rows unpredicted.")
    threshold_grid = np.linspace(0.25, 0.75, 201)
    threshold_scores = np.asarray(
        [balanced_accuracy(y, oof_probability >= threshold) for threshold in threshold_grid]
    )
    best_index = int(np.argmax(threshold_scores))
    selected_threshold = float(threshold_grid[best_index])
    rng = np.random.default_rng(config.split_seed + 1000)
    shot_probability = rng.binomial(config.shots, oof_probability) / config.shots

    # Out-of-fold score at every annealing temperature, so the stop temperature
    # is chosen on held-out rows instead of on the training fit it overfits.
    temperature_curve: list[dict] = []
    if stage_oof is not None:
        for position, probabilities in enumerate(stage_oof):
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Stage OOF generation left rows unpredicted.")
            scores = np.asarray(
                [balanced_accuracy(y, probabilities >= t) for t in threshold_grid]
            )
            best = int(np.argmax(scores))
            shot = rng.binomial(config.shots, probabilities) / config.shots
            temperature_curve.append(
                {
                    "stage": position,
                    "temperature": stage_temperatures[position],
                    "is_warm_start_only": position == 0,
                    # Per fold, so a gain can be checked for consistency rather
                    # than read off a single pooled number.
                    "per_fold_balanced_accuracy_at_selected_threshold": [
                        balanced_accuracy(
                            y[valid_rows], probabilities[valid_rows] >= threshold_grid[best]
                        )
                        for _, valid_rows in splits
                    ],
                    "balanced_accuracy_at_0_5": balanced_accuracy(
                        y, probabilities >= 0.5
                    ),
                    "selected_threshold": float(threshold_grid[best]),
                    "balanced_accuracy_at_selected_threshold": float(scores[best]),
                    "shot_balanced_accuracy_at_selected_threshold": balanced_accuracy(
                        y, shot >= threshold_grid[best]
                    ),
                    "balanced_binary_cross_entropy": balanced_binary_cross_entropy(
                        y, probabilities
                    ),
                }
            )

    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_cv_c1_"
        f"split{config.split_seed}_init{config.init_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "mode": "C1_quantum_only_out_of_fold_threshold_selection",
        "architecture": C1_ARCHITECTURE,
        "submission_created": False,
        "config": {
            **asdict(config),
            "train_csv": str(config.train_csv),
            "artifacts_dir": str(config.artifacts_dir),
        },
        "data": data_info,
        "classical_predictive_model_used": False,
        "parameter_transfer_used": False,
        "threshold_source": "out_of_fold_C1_quantum_probabilities_only",
        "balanced_accuracy_at_0_5": balanced_accuracy(y, oof_probability >= 0.5),
        "selected_threshold": selected_threshold,
        "balanced_accuracy_at_selected_threshold": float(threshold_scores[best_index]),
        "shot_balanced_accuracy_at_selected_threshold": balanced_accuracy(
            y, shot_probability >= selected_threshold
        ),
        "balanced_binary_cross_entropy": balanced_binary_cross_entropy(
            y, oof_probability
        ),
        "annealing_temperature_curve": temperature_curve,
        "fold_results": fold_results,
        "qiskit_equivalence_reports": equivalence_reports,
        "threshold_grid": [
            {"threshold": float(threshold), "balanced_accuracy": float(score)}
            for threshold, score in zip(threshold_grid, threshold_scores, strict=True)
        ],
        "oof_probability": [float(value) for value in oof_probability],
    }
    (run_dir / "cross_validation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return run_dir


def run_c1_final(config: C1TrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    if config.max_rows is not None and config.max_rows < len(y):
        raise ValueError("C1 final training must use the complete public train set.")
    if not 0.0 < config.decision_threshold < 1.0:
        raise ValueError("C1 decision_threshold must be in (0, 1).")
    shared = dict(
        seed=config.seed,
        init_scale=config.init_scale,
        affine_scale_jitter=config.affine_scale_jitter,
        maxiter=config.maxiter,
        n_restarts=config.n_restarts,
    )
    if config.objective == "balanced_bce":
        weights, optimization, initial = optimize_c1_quantum_loss(x, y, **shared)
    elif config.objective in ("soft_balanced_accuracy", "smooth_auc"):
        weights, optimization, initial = optimize_c1_annealed(
            x,
            y,
            surrogate=config.objective,
            temperature_start=config.temperature_start,
            temperature_stop=config.temperature_stop,
            anneal_stages=config.anneal_stages,
            stage_maxiter=config.stage_maxiter,
            **shared,
        )
        if config.select_stage is not None:
            stage_weights = optimization["stage_weights"]
            if not 0 <= config.select_stage < len(stage_weights):
                raise ValueError(
                    f"select_stage must be in 0..{len(stage_weights) - 1}."
                )
            weights = np.asarray(
                stage_weights[config.select_stage], dtype=float
            )
            optimization = {
                **optimization,
                "submitted_stage": config.select_stage,
                "submitted_temperature": optimization["stage_temperatures"][
                    config.select_stage
                ],
                "stage_selection": "index_chosen_from_train_only_out_of_fold_curve",
            }
        optimization = {
            key: value
            for key, value in optimization.items()
            if key != "stage_weights"
        }
    else:
        raise ValueError(f"Unknown C1 objective: {config.objective!r}")
    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_final_c1_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    constraint = export_c1_submission_qasm(run_dir / "classifier.qasm")
    weights_payload = {
        **{f"theta_{index}": float(value) for index, value in enumerate(weights)},
        "threshold": config.decision_threshold,
    }
    (run_dir / "weights.json").write_text(
        json.dumps(weights_payload, indent=2), encoding="utf-8"
    )
    expected_feature_uses = {
        f"x_{feature_index}": sum(
            feature_index in block for block in C1_REUPLOAD_BLOCKS
        )
        for feature_index in range(C1_N_FEATURES)
    }
    provenance = {
        "architecture": C1_ARCHITECTURE,
        "weight_count": C1_N_WEIGHTS,
        "parameter_generation": "label_independent_random_initialization_then_direct_C1_quantum_probability_gradient_optimization",
        "initialization": {
            "seed": config.seed,
            "affine_scales": f"uniform({1-config.affine_scale_jitter},{1+config.affine_scale_jitter})",
            "bias_and_mixer_parameters": f"uniform({-config.init_scale},{config.init_scale})",
            "label_dependent": False,
        },
        "training_quantum_emulator": "vectorized exact 4-qubit statevector verified against qiskit.quantum_info.Statevector",
        "loss_source": "C1 q0 measurement probability versus raw public_train label",
        "loss": {
            "balanced_bce": "balanced binary cross-entropy",
            "soft_balanced_accuracy": (
                "balanced binary cross-entropy warm start, then annealed "
                "sigmoid((p - threshold)/T) surrogate of balanced accuracy"
            ),
            "smooth_auc": (
                "balanced binary cross-entropy warm start, then annealed "
                "sigmoid((p_pos - p_neg)/T) surrogate of ROC AUC over all "
                "positive/negative train pairs"
            ),
        }[config.objective],
        "loss_inputs": "circuit q0 probability and raw public_train label only",
        "annealing": None
        if config.objective == "balanced_bce"
        else {
            "temperature_start": config.temperature_start,
            "temperature_stop": config.temperature_stop,
            "stages": config.anneal_stages,
            "submitted_stage": config.select_stage,
            "stage_and_temperature_selection": (
                "train-only 5-fold out-of-fold balanced accuracy on "
                "public_train.csv; public_test.csv was not used"
            ),
        },
        "optimizer": "L-BFGS-B with exact adjoint quantum-circuit gradient",
        "selected_raw_features_zero_based": list(range(C1_N_FEATURES)),
        "selected_raw_features_csv": [f"x{index + 1}" for index in range(C1_N_FEATURES)],
        "expected_feature_uses": expected_feature_uses,
        "feature_processing": "none",
        "single_feature_affine_encoding": True,
        "classical_predictive_model": None,
        "surrogate_model": None,
        "teacher_model": None,
        "warm_start": False,
        "transferred_coefficients": False,
        "decision_threshold": {
            "value": config.decision_threshold,
            "selection": "pre_registered_train_only_quantum_OOF_or_fixed",
        },
    }
    (run_dir / "parameter_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    metrics = {
        "mode": "full_train_C1_direct_quantum_gradient_optimization",
        "architecture": C1_ARCHITECTURE,
        "config": _config_payload(config),
        "data": data_info,
        "optimization": optimization,
        "full_train_metrics": _metrics(
            x, y, weights, shots=config.shots, seed=config.seed + 1,
            threshold=config.decision_threshold,
        ),
        "constraint_report": constraint,
        "qiskit_equivalence": c1_qiskit_equivalence_report(x, weights),
        "initial_feature_causal_report": c1_feature_causal_report(x, initial),
        "trained_feature_causal_report": c1_feature_causal_report(x, weights),
    }
    (run_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    note = build_encoding_note(
        architecture=C1_ARCHITECTURE,
        encoding_description=(
            "Raw features enter as RY rotations. Re-uploading blocks 1 and 3 encode "
            "x1-x4 on q0-q3 and blocks 2 and 4 encode x5-x8 on q0-q3, each as one "
            "RY(theta_scale * x_i + theta_bias) gate carrying a single raw feature."
        ),
        entanglement_description=(
            "Each block applies trainable RZ and RY mixers, then the causal CX funnel "
            "CX(3->2), CX(2->1), CX(1->0). A final trainable RY on the readout qubit "
            "turns accumulated phase into a Z-basis probability."
        ),
        readout_qubit=0,
        used_features=range(C1_N_FEATURES),
        uploads_per_feature=expected_feature_uses,
        qubits=constraint["qubits"],
        depth=constraint["depth"],
        two_qubit_gates=constraint["two_qubit_gate_count"],
        weight_count=C1_N_WEIGHTS,
        objective=config.objective,
        threshold=config.decision_threshold,
        threshold_source=(
            "balanced accuracy of train-only 5-fold out-of-fold circuit probabilities "
            "on public_train.csv"
        ),
        initialization=(
            f"Every theta starts from a label-independent uniform random draw with seed "
            f"{config.seed}: affine scales near 1, biases and mixers near 0. Nothing "
            "about the labels enters the initialization."
        ),
        annealing=provenance["annealing"],
    )
    (run_dir / "encoding_note.txt").write_text(note, encoding="utf-8")
    return run_dir
