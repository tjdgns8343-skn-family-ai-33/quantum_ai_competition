"""End-to-end training for the compliant F1 eight-qubit quantum circuit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from .restarts import multistart_minimize, restart_seeds
from .data import load_raw_train
from .f1_circuit import (
    F1_ARCHITECTURE,
    F1_DEFAULT_BLOCKS,
    F1_N_FEATURES,
    F1_N_QUBITS,
    F1_PARAMETERS_PER_BLOCK,
    build_f1_unitary,
    export_f1_submission_qasm,
    f1_reupload_blocks,
    f1_weight_count,
)
from .f1_simulator import (
    f1_balanced_bce_value_and_gradient,
    f1_exact_probabilities,
)
from .metrics import (
    balanced_accuracy,
    balanced_binary_cross_entropy,
    binary_cross_entropy,
)
from .training import FIXED_THRESHOLD, stratified_holdout


@dataclass(frozen=True)
class F1TrainConfig:
    train_csv: Path
    artifacts_dir: Path = Path("artifacts")
    seed: int = 2026
    split_seed: int | None = None
    n_blocks: int = F1_DEFAULT_BLOCKS
    init_scale: float = 0.05
    affine_scale_center: float = 1.0
    affine_scale_jitter: float = 0.1
    maxiter: int = 200
    n_restarts: int = 1
    shots: int = 1024
    validation_fraction: float = 0.2
    max_rows: int | None = None
    decision_threshold: float = FIXED_THRESHOLD


@dataclass(frozen=True)
class F1CrossValidationConfig:
    train_csv: Path
    artifacts_dir: Path = Path("artifacts")
    split_seed: int = 2026
    init_seed: int = 2026
    n_blocks: int = F1_DEFAULT_BLOCKS
    init_scale: float = 0.05
    affine_scale_center: float = 1.0
    affine_scale_jitter: float = 0.1
    maxiter: int = 200
    n_restarts: int = 1
    folds: int = 5
    shots: int = 1024


def f1_random_initial_point(
    seed: int,
    *,
    local_scale: float,
    affine_scale_jitter: float,
    affine_scale_center: float = 1.0,
    n_blocks: int = F1_DEFAULT_BLOCKS,
) -> np.ndarray:
    """Label-independent initialization around direct raw-angle encoding.

    Mixers start near identity, which keeps the deep circuit away from the
    barren-plateau regime that uniformly random angles would produce.
    """
    if local_scale < 0 or affine_scale_jitter < 0:
        raise ValueError("F1 initialization scales must be non-negative.")
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-local_scale, local_scale, f1_weight_count(n_blocks))
    for block_index in range(n_blocks):
        offset = F1_PARAMETERS_PER_BLOCK * block_index
        weights[offset : offset + F1_N_QUBITS] = rng.uniform(
            affine_scale_center - affine_scale_jitter,
            affine_scale_center + affine_scale_jitter,
            F1_N_QUBITS,
        )
    return weights


def _stratified_subsample(
    y: np.ndarray, max_rows: int | None, seed: int
) -> np.ndarray:
    if max_rows is None or max_rows >= len(y):
        return np.arange(len(y), dtype=int)
    if max_rows < 20:
        raise ValueError("F1 max_rows must be at least 20.")
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


def _f1_bounds(n_blocks: int) -> list[tuple[float, float]]:
    bounds = [(-np.pi, np.pi) for _ in range(f1_weight_count(n_blocks))]
    for block_index in range(n_blocks):
        offset = F1_PARAMETERS_PER_BLOCK * block_index
        for qubit in range(F1_N_QUBITS):
            bounds[offset + qubit] = (-3.0, 3.0)
    return bounds


def optimize_f1_quantum_loss(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_blocks: int,
    init_scale: float,
    affine_scale_center: float,
    affine_scale_jitter: float,
    maxiter: int,
    n_restarts: int = 1,
    restart_seed_stride: int = 1000,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Optimize analytic gradients of the F1 statevector probability only."""
    seeds = restart_seeds(seed, n_restarts, restart_seed_stride)
    initial_points = [
        f1_random_initial_point(
            restart_seed,
            local_scale=init_scale,
            affine_scale_jitter=affine_scale_jitter,
            affine_scale_center=affine_scale_center,
            n_blocks=n_blocks,
        )
        for restart_seed in seeds
    ]

    def value_and_gradient(weight_values: np.ndarray) -> tuple[float, np.ndarray]:
        return f1_balanced_bce_value_and_gradient(x, y, weight_values, n_blocks)

    weights, initial, summary = multistart_minimize(
        value_and_gradient,
        initial_points,
        bounds=_f1_bounds(n_blocks),
        maxiter=maxiter,
    )
    return weights, {
        "optimizer": "L-BFGS-B_with_exact_adjoint_gradient",
        "training_emulator": "vectorized_exact_8_qubit_statevector",
        "qiskit_equivalence_required": True,
        "objective": "balanced_binary_cross_entropy_of_F1_q0_probability",
        "n_blocks": int(n_blocks),
        "weight_count": f1_weight_count(n_blocks),
        "restart_seeds": seeds,
        **summary,
    }, initial


def _metrics(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_blocks: int,
    *,
    shots: int,
    seed: int,
    threshold: float = FIXED_THRESHOLD,
) -> dict:
    exact = f1_exact_probabilities(x, weights, n_blocks)
    rng = np.random.default_rng(seed)
    shot_probability = rng.binomial(shots, exact) / shots
    return {
        "rows": int(len(y)),
        "exact_balanced_accuracy_at_threshold": balanced_accuracy(
            y, (exact >= threshold).astype(int)
        ),
        "shot_balanced_accuracy_at_threshold": balanced_accuracy(
            y, (shot_probability >= threshold).astype(int)
        ),
        "exact_binary_cross_entropy": binary_cross_entropy(y, exact),
        "exact_balanced_binary_cross_entropy": balanced_binary_cross_entropy(y, exact),
        "shot_binary_cross_entropy": binary_cross_entropy(y, shot_probability),
        "mean_exact_probability": float(np.mean(exact)),
        "shots": shots,
        "shot_seed": seed,
        "threshold": threshold,
    }


def f1_qiskit_equivalence_report(
    x: np.ndarray, weights: np.ndarray, n_blocks: int, *, max_rows: int = 16
) -> dict:
    sample = np.asarray(x[:max_rows], dtype=float)
    custom = f1_exact_probabilities(sample, weights, n_blocks)
    circuit, features, parameters = build_f1_unitary(n_blocks)
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


def f1_feature_causal_report(
    x: np.ndarray, weights: np.ndarray, n_blocks: int, *, perturbation: float = 1e-5
) -> dict:
    sample = np.asarray(x[: min(128, len(x))], dtype=float)
    rows: list[dict] = []
    for feature_index in range(F1_N_FEATURES):
        plus = sample.copy()
        minus = sample.copy()
        plus[:, feature_index] += perturbation
        minus[:, feature_index] -= perturbation
        sensitivity = np.abs(
            (
                f1_exact_probabilities(plus, weights, n_blocks)
                - f1_exact_probabilities(minus, weights, n_blocks)
            )
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


def _config_payload(config) -> dict:
    return {
        **asdict(config),
        "train_csv": str(config.train_csv),
        "artifacts_dir": str(config.artifacts_dir),
    }


def run_f1_screen(config: F1TrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    split_seed = config.seed if config.split_seed is None else config.split_seed
    selected_rows = _stratified_subsample(y, config.max_rows, split_seed)
    screen_x, screen_y = x[selected_rows], y[selected_rows]
    fit, valid = stratified_holdout(screen_y, config.validation_fraction, split_seed)
    weights, optimization, initial = optimize_f1_quantum_loss(
        screen_x[fit],
        screen_y[fit],
        seed=config.seed,
        n_blocks=config.n_blocks,
        init_scale=config.init_scale,
        affine_scale_center=config.affine_scale_center,
        affine_scale_jitter=config.affine_scale_jitter,
        maxiter=config.maxiter,
        n_restarts=config.n_restarts,
    )
    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_screen_f1_"
        f"b{config.n_blocks}_split{split_seed}_init{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "mode": "train_only_holdout_F1_direct_quantum_gradient_screen",
        "architecture": F1_ARCHITECTURE,
        "submission_created": False,
        "n_blocks": config.n_blocks,
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
            screen_x[fit],
            screen_y[fit],
            weights,
            config.n_blocks,
            shots=config.shots,
            seed=split_seed + 1,
            threshold=config.decision_threshold,
        ),
        "validation_metrics": _metrics(
            screen_x[valid],
            screen_y[valid],
            weights,
            config.n_blocks,
            shots=config.shots,
            seed=split_seed + 2,
            threshold=config.decision_threshold,
        ),
        "qiskit_equivalence": f1_qiskit_equivalence_report(
            screen_x[valid], weights, config.n_blocks
        ),
        "initial_feature_causal_report": f1_feature_causal_report(
            screen_x, initial, config.n_blocks
        ),
        "trained_feature_causal_report": f1_feature_causal_report(
            screen_x, weights, config.n_blocks
        ),
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
        raise ValueError("F1 cross-validation requires at least two folds.")
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


def run_f1_cross_validation(config: F1CrossValidationConfig) -> Path:
    """Generate genuine F1 OOF probabilities and a train-only threshold."""
    x, y, data_info = load_raw_train(config.train_csv)
    splits = _stratified_folds(y, config.folds, config.split_seed)
    oof_probability = np.full(len(y), np.nan, dtype=float)
    fold_results: list[dict] = []
    equivalence_reports: list[dict] = []

    for fold_index, (fit, valid) in enumerate(splits):
        weights, optimization, _ = optimize_f1_quantum_loss(
            x[fit],
            y[fit],
            seed=config.init_seed,
            n_blocks=config.n_blocks,
            init_scale=config.init_scale,
            affine_scale_center=config.affine_scale_center,
            affine_scale_jitter=config.affine_scale_jitter,
            maxiter=config.maxiter,
            n_restarts=config.n_restarts,
        )
        probability = f1_exact_probabilities(x[valid], weights, config.n_blocks)
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
        equivalence_reports.append(
            f1_qiskit_equivalence_report(x[valid], weights, config.n_blocks)
        )

    if not np.isfinite(oof_probability).all():
        raise RuntimeError("F1 OOF generation left one or more rows unpredicted.")
    threshold_grid = np.linspace(0.25, 0.75, 201)
    threshold_scores = np.asarray(
        [balanced_accuracy(y, oof_probability >= threshold) for threshold in threshold_grid]
    )
    best_index = int(np.argmax(threshold_scores))
    selected_threshold = float(threshold_grid[best_index])
    rng = np.random.default_rng(config.split_seed + 1000)
    shot_probability = rng.binomial(config.shots, oof_probability) / config.shots

    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_cv_f1_"
        f"b{config.n_blocks}_split{config.split_seed}_init{config.init_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "mode": "F1_quantum_only_out_of_fold_threshold_selection",
        "architecture": F1_ARCHITECTURE,
        "submission_created": False,
        "n_blocks": config.n_blocks,
        "config": _config_payload(config),
        "data": data_info,
        "classical_predictive_model_used": False,
        "parameter_transfer_used": False,
        "threshold_source": "out_of_fold_F1_quantum_probabilities_only",
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


def run_f1_final(config: F1TrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    if config.max_rows is not None and config.max_rows < len(y):
        raise ValueError("F1 final training must use the complete public train set.")
    if not 0.0 < config.decision_threshold < 1.0:
        raise ValueError("F1 decision_threshold must be in (0, 1).")
    weights, optimization, initial = optimize_f1_quantum_loss(
        x,
        y,
        seed=config.seed,
        n_blocks=config.n_blocks,
        init_scale=config.init_scale,
        affine_scale_center=config.affine_scale_center,
        affine_scale_jitter=config.affine_scale_jitter,
        maxiter=config.maxiter,
        n_restarts=config.n_restarts,
    )
    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_final_f1_"
        f"b{config.n_blocks}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    constraint = export_f1_submission_qasm(
        run_dir / "classifier.qasm", config.n_blocks
    )
    weights_payload = {
        **{f"theta_{index}": float(value) for index, value in enumerate(weights)},
        "threshold": config.decision_threshold,
    }
    (run_dir / "weights.json").write_text(
        json.dumps(weights_payload, indent=2), encoding="utf-8"
    )
    blocks = f1_reupload_blocks(config.n_blocks)
    expected_feature_uses = {
        f"x_{feature_index}": sum(block.count(feature_index) for block in blocks)
        for feature_index in range(F1_N_FEATURES)
    }
    provenance = {
        "architecture": F1_ARCHITECTURE,
        "weight_count": f1_weight_count(config.n_blocks),
        "parameter_generation": "label_independent_random_initialization_then_direct_F1_quantum_probability_gradient_optimization",
        "initialization": {
            "seed": config.seed,
            "affine_scales": (
                f"uniform({config.affine_scale_center - config.affine_scale_jitter},"
                f"{config.affine_scale_center + config.affine_scale_jitter})"
            ),
            "bias_and_mixer_parameters": f"uniform({-config.init_scale},{config.init_scale})",
            "label_dependent": False,
        },
        "training_quantum_emulator": "vectorized exact 8-qubit statevector verified against qiskit.quantum_info.Statevector",
        "loss_source": "F1 q0 measurement probability versus raw public_train label",
        "loss": "balanced binary cross-entropy",
        "optimizer": "L-BFGS-B with exact adjoint quantum-circuit gradient",
        "selected_raw_features_zero_based": list(range(F1_N_FEATURES)),
        "selected_raw_features_csv": [f"x{index + 1}" for index in range(F1_N_FEATURES)],
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
        "mode": "full_train_F1_direct_quantum_gradient_optimization",
        "architecture": F1_ARCHITECTURE,
        "n_blocks": config.n_blocks,
        "config": _config_payload(config),
        "data": data_info,
        "optimization": optimization,
        "full_train_metrics": _metrics(
            x,
            y,
            weights,
            config.n_blocks,
            shots=config.shots,
            seed=config.seed + 1,
            threshold=config.decision_threshold,
        ),
        "constraint_report": constraint,
        "qiskit_equivalence": f1_qiskit_equivalence_report(x, weights, config.n_blocks),
        "initial_feature_causal_report": f1_feature_causal_report(
            x, initial, config.n_blocks
        ),
        "trained_feature_causal_report": f1_feature_causal_report(
            x, weights, config.n_blocks
        ),
    }
    (run_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    note = (
        f"F1 eight-qubit tree-funnel data-reuploading VQC uses all raw CSV features with no "
        f"preprocessing or augmentation. Each of the {config.n_blocks} blocks encodes every raw "
        f"feature once as a single-feature affine RY(theta_scale*x_i + theta_bias) gate, with block b "
        f"placing feature (q + b) mod 8 on qubit q. Each block then applies trainable RZ/RY mixers "
        f"and the CX tree funnel CX(1->0), CX(3->2), CX(5->4), CX(7->6), CX(2->0), CX(6->4), CX(4->0), "
        f"which brings all eight qubits into the q0 causal cone within a single block; only q0 is measured. "
        "All theta values use label-independent random initialization and are optimized only against "
        "balanced cross-entropy of the exact F1 q0 statevector probability on public_train.csv. "
        "The vectorized adjoint statevector and Qiskit Statevector probabilities agree within 1e-10. "
        "No classical predictive, surrogate, teacher, warm-start, transferred coefficient, PCA, scaling, "
        "imputation, feature product, kernel, or augmentation is used. "
        f"Threshold is {config.decision_threshold:g} and must be selected only from train-only quantum predictions."
    )
    (run_dir / "encoding_note.txt").write_text(note + "\n", encoding="utf-8")
    return run_dir
