"""End-to-end training for the compliant C1 quantum circuit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

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
)
from .data import load_raw_train
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
    shots: int = 1024
    validation_fraction: float = 0.2
    max_rows: int | None = None
    decision_threshold: float = FIXED_THRESHOLD


@dataclass(frozen=True)
class C1CrossValidationConfig:
    train_csv: Path
    artifacts_dir: Path = Path("artifacts")
    split_seed: int = 2026
    init_seed: int = 2026
    init_scale: float = 0.05
    affine_scale_jitter: float = 0.1
    maxiter: int = 80
    folds: int = 5
    shots: int = 1024


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
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Optimize analytic gradients of the C1 statevector probability only."""
    initial = c1_random_initial_point(
        seed,
        local_scale=init_scale,
        affine_scale_jitter=affine_scale_jitter,
    )
    history: list[dict] = []
    best_loss = float("inf")
    best_weights = initial.copy()

    def value_and_gradient(weight_values: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best_loss, best_weights
        value, gradient = c1_balanced_bce_value_and_gradient(
            x, y, weight_values
        )
        gradient_norm = float(np.linalg.norm(gradient))
        history.append(
            {
                "evaluation": len(history) + 1,
                "loss": float(value),
                "gradient_norm": gradient_norm,
            }
        )
        if value < best_loss:
            best_loss = float(value)
            best_weights = np.asarray(weight_values, dtype=float).copy()
        return float(value), gradient

    initial_loss, initial_gradient = c1_balanced_bce_value_and_gradient(
        x, y, initial
    )
    result = minimize(
        value_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=_c1_bounds(),
        options={"maxiter": maxiter, "maxls": 30, "ftol": 1e-10, "gtol": 1e-7},
    )
    # scipy's terminal point is normally the best point, but preserve the
    # lowest directly observed quantum loss in case a line search terminates.
    selected = best_weights
    return selected, {
        "optimizer": "L-BFGS-B_with_exact_adjoint_gradient",
        "training_emulator": "vectorized_exact_4_qubit_statevector",
        "qiskit_equivalence_required": True,
        "objective": "balanced_binary_cross_entropy_of_C1_q0_probability",
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
        "initial_loss": float(initial_loss),
        "initial_gradient_norm": float(np.linalg.norm(initial_gradient)),
        "terminal_loss": float(result.fun),
        "best_observed_loss": float(best_loss),
        "history": history,
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
    equivalence_reports: list[dict] = []

    for fold_index, (fit, valid) in enumerate(splits):
        weights, optimization, _ = optimize_c1_quantum_loss(
            x[fit],
            y[fit],
            seed=config.init_seed,
            init_scale=config.init_scale,
            affine_scale_jitter=config.affine_scale_jitter,
            maxiter=config.maxiter,
        )
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
    weights, optimization, initial = optimize_c1_quantum_loss(
        x,
        y,
        seed=config.seed,
        init_scale=config.init_scale,
        affine_scale_jitter=config.affine_scale_jitter,
        maxiter=config.maxiter,
    )
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
        "loss": "balanced binary cross-entropy",
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
    note = (
        "C1 causal data-reuploading VQC uses all raw CSV features with no preprocessing or augmentation. "
        "Blocks 1/3 encode x1-x4 and blocks 2/4 encode x5-x8 as single-feature affine "
        "RY(theta_scale*x_i + theta_bias) gates. Each block then applies trainable RZ/RY mixers "
        "and the causal CX funnel CX(3->2), CX(2->1), CX(1->0); only q0 is measured. "
        "All theta values use label-independent random initialization and are optimized only against "
        "balanced cross-entropy of the exact C1 q0 statevector probability on public_train.csv. "
        "The vectorized adjoint statevector and Qiskit Statevector probabilities agree within 1e-10. "
        "No classical predictive, surrogate, teacher, warm-start, transferred coefficient, PCA, scaling, "
        "imputation, feature product, kernel, or augmentation is used. "
        f"Threshold is {config.decision_threshold:g} and must be selected only from train-only quantum predictions."
    )
    (run_dir / "encoding_note.txt").write_text(note + "\n", encoding="utf-8")
    return run_dir
