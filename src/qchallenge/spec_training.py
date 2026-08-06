"""Training and evaluation for any ``CircuitSpec``.

Generic on purpose: earlier architectures each carried a near-identical copy of
this file, which is how the restart-count wiring bug reached cross-validation
while screen and final training were already fixed.  One implementation means
one place to fix.

Selection stays train-only.  Restarts are chosen by training loss alone, the
threshold comes from out-of-fold circuit probabilities, and public_test.csv is
never read here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from functools import partial
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .data import load_raw_train
from .gatespec import CircuitSpec, build_circuit
from .generic_simulator import exact_probabilities, value_and_gradient
from .objectives import (
    smooth_auc,
    smooth_ks,
    soft_balanced_accuracy,
    temperature_schedule,
)
from .metrics import (
    balanced_accuracy,
    balanced_binary_cross_entropy,
    binary_cross_entropy,
)
from .restarts import multistart_minimize, restart_seeds
from .training import FIXED_THRESHOLD, stratified_holdout


@dataclass(frozen=True)
class SpecTrainConfig:
    train_csv: Path
    artifacts_dir: Path = Path("artifacts")
    label: str = "spec"
    seed: int = 2026
    split_seed: int | None = None
    init_scale: float = 1.0
    affine_scale_center: float = 1.0
    affine_scale_jitter: float = 0.3
    maxiter: int = 200
    n_restarts: int = 4
    folds: int = 5
    shots: int = 1024
    validation_fraction: float = 0.2
    max_rows: int | None = None
    decision_threshold: float = FIXED_THRESHOLD
    objective: str = "balanced_bce"
    temperature_start: float = 0.30
    temperature_stop: float = 0.015
    anneal_stages: int = 10
    stage_maxiter: int = 40


def scale_indices(spec: CircuitSpec) -> tuple[int, ...]:
    return tuple(
        sorted({gate.scale_index for gate in spec.gates if gate.is_data})
    )


def random_initial_point(
    spec: CircuitSpec,
    seed: int,
    *,
    init_scale: float,
    affine_scale_center: float,
    affine_scale_jitter: float,
) -> np.ndarray:
    """Label-independent initialization; affine scales start near one.

    The organizers already scaled every raw feature into [-pi, pi], so a scale
    of about one spans the encoding range without wrapping.
    """
    if init_scale < 0 or affine_scale_jitter < 0:
        raise ValueError("Initialization scales must be non-negative.")
    rng = np.random.default_rng(seed)
    weights = rng.uniform(-init_scale, init_scale, spec.n_weights)
    indices = np.asarray(scale_indices(spec), dtype=int)
    if len(indices):
        weights[indices] = rng.uniform(
            affine_scale_center - affine_scale_jitter,
            affine_scale_center + affine_scale_jitter,
            len(indices),
        )
    return weights


def spec_bounds(spec: CircuitSpec) -> list[tuple[float, float]]:
    bounds = [(-np.pi, np.pi) for _ in range(spec.n_weights)]
    for index in scale_indices(spec):
        bounds[index] = (-3.0, 3.0)
    return bounds


def optimize_spec(
    spec: CircuitSpec,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    init_scale: float,
    affine_scale_center: float,
    affine_scale_jitter: float,
    maxiter: int,
    n_restarts: int,
    restart_seed_stride: int = 1000,
) -> tuple[np.ndarray, dict, np.ndarray]:
    seeds = restart_seeds(seed, n_restarts, restart_seed_stride)
    initial_points = [
        random_initial_point(
            spec,
            restart_seed,
            init_scale=init_scale,
            affine_scale_center=affine_scale_center,
            affine_scale_jitter=affine_scale_jitter,
        )
        for restart_seed in seeds
    ]
    weights, initial, summary = multistart_minimize(
        lambda values: value_and_gradient(spec, x, y, values),
        initial_points,
        bounds=spec_bounds(spec),
        maxiter=maxiter,
    )
    return weights, {
        "optimizer": "L-BFGS-B_with_exact_adjoint_gradient",
        "training_emulator": f"vectorized_exact_{spec.n_qubits}_qubit_statevector",
        "qiskit_equivalence_required": True,
        "objective": "balanced_binary_cross_entropy_of_readout_probability",
        "weight_count": spec.n_weights,
        "restart_seeds": seeds,
        **summary,
    }, initial


def optimize_spec_annealed(
    spec: CircuitSpec,
    x: np.ndarray,
    y: np.ndarray,
    *,
    surrogate: str,
    seed: int,
    init_scale: float,
    affine_scale_center: float,
    affine_scale_jitter: float,
    maxiter: int,
    n_restarts: int,
    threshold: float = FIXED_THRESHOLD,
    temperature_start: float = 0.30,
    temperature_stop: float = 0.015,
    anneal_stages: int = 10,
    stage_maxiter: int = 40,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Balanced-BCE warm start, then anneal a surrogate; same shape as C1's.

    Weights are recorded at every stage so one cross-validation run yields the
    whole temperature curve and the stop temperature is chosen out-of-fold.
    """
    builders = {
        "smooth_auc": lambda t: partial(smooth_auc, temperature=t),
        "smooth_ks": lambda t: partial(smooth_ks, temperature=t),
        "soft_balanced_accuracy": lambda t: partial(
            soft_balanced_accuracy, threshold=threshold, temperature=t
        ),
    }
    if surrogate not in builders:
        raise ValueError(f"Unknown surrogate: {surrogate!r}")

    weights, warm_start_summary, initial = optimize_spec(
        spec,
        x,
        y,
        seed=seed,
        init_scale=init_scale,
        affine_scale_center=affine_scale_center,
        affine_scale_jitter=affine_scale_jitter,
        maxiter=maxiter,
        n_restarts=n_restarts,
    )
    stage_weights = [[float(v) for v in weights]]
    stage_temperatures: list[float | None] = [None]
    stages: list[dict] = []

    for temperature in temperature_schedule(
        temperature_start, temperature_stop, anneal_stages
    ):
        objective = builders[surrogate](temperature)
        result = minimize(
            lambda values: value_and_gradient(spec, x, y, values, objective),
            weights,
            method="L-BFGS-B",
            jac=True,
            bounds=spec_bounds(spec),
            options={"maxiter": stage_maxiter, "maxls": 30, "ftol": 1e-12},
        )
        weights = np.asarray(result.x, dtype=float)
        stage_weights.append([float(v) for v in weights])
        stage_temperatures.append(float(temperature))
        stages.append(
            {
                "temperature": float(temperature),
                "train_balanced_accuracy": balanced_accuracy(
                    y, (exact_probabilities(spec, x, weights) >= threshold).astype(int)
                ),
                "iterations": int(result.nit),
            }
        )

    return weights, {
        "optimizer": "L-BFGS-B_with_exact_adjoint_gradient",
        "objective": f"balanced_bce_warm_start_then_annealed_{surrogate}",
        "weight_count": spec.n_weights,
        "anneal_stages": stages,
        "stage_weights": stage_weights,
        "stage_temperatures": stage_temperatures,
        "warm_start": warm_start_summary,
        "restart_count": warm_start_summary.get("restart_count", n_restarts),
    }, initial


def metrics(
    spec: CircuitSpec,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    shots: int,
    seed: int,
    threshold: float,
) -> dict:
    exact = exact_probabilities(spec, x, weights)
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
        "mean_exact_probability": float(np.mean(exact)),
        "shots": shots,
        "threshold": threshold,
    }


def qiskit_equivalence_report(
    spec: CircuitSpec, x: np.ndarray, weights: np.ndarray, *, max_rows: int = 16
) -> dict:
    sample = np.asarray(x[:max_rows], dtype=float)
    custom = exact_probabilities(spec, sample, weights)
    circuit, features, parameters = build_circuit(spec)
    reference: list[float] = []
    for row in sample:
        bindings = {
            **{
                parameter: row[int(str(parameter).split("_")[1])]
                for parameter in features
            },
            **dict(zip(parameters, weights, strict=True)),
        }
        state = Statevector.from_instruction(circuit.assign_parameters(bindings))
        reference.append(float(state.probabilities([spec.readout_qubit])[1]))
    difference = np.abs(custom - np.asarray(reference))
    return {
        "rows": int(len(sample)),
        "maximum_absolute_probability_difference": float(np.max(difference)),
        "passes_at_1e_10": bool(np.max(difference) <= 1e-10),
    }


def feature_causal_report(
    spec: CircuitSpec, x: np.ndarray, weights: np.ndarray, *, perturbation: float = 1e-5
) -> dict:
    sample = np.asarray(x[: min(128, len(x))], dtype=float)
    rows: list[dict] = []
    for feature_index in range(8):
        plus, minus = sample.copy(), sample.copy()
        plus[:, feature_index] += perturbation
        minus[:, feature_index] -= perturbation
        sensitivity = np.abs(
            (
                exact_probabilities(spec, plus, weights)
                - exact_probabilities(spec, minus, weights)
            )
            / (2.0 * perturbation)
        )
        rows.append(
            {
                "feature": f"x{feature_index + 1}",
                "used_in_circuit": feature_index in spec.used_features(),
                "mean_absolute_dp_dx": float(np.mean(sensitivity)),
                "maximum_absolute_dp_dx": float(np.max(sensitivity)),
            }
        )
    return {"perturbation": perturbation, "rows": rows}


def _subsample(y: np.ndarray, max_rows: int | None, seed: int) -> np.ndarray:
    if max_rows is None or max_rows >= len(y):
        return np.arange(len(y), dtype=int)
    rng = np.random.default_rng(seed)
    parts, remaining = [], max_rows
    for position, label in enumerate((0, 1)):
        indices = np.flatnonzero(y == label)
        if position == 0:
            count = max(1, min(int(round(max_rows * len(indices) / len(y))), len(indices)))
            remaining -= count
        else:
            count = max(1, min(remaining, len(indices)))
        parts.append(rng.choice(indices, size=count, replace=False))
    return np.sort(np.concatenate(parts))


def _payload_config(config: SpecTrainConfig) -> dict:
    return {
        **asdict(config),
        "train_csv": str(config.train_csv),
        "artifacts_dir": str(config.artifacts_dir),
    }


def run_spec_screen(spec: CircuitSpec, config: SpecTrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    split_seed = config.seed if config.split_seed is None else config.split_seed
    selected = _subsample(y, config.max_rows, split_seed)
    screen_x, screen_y = x[selected], y[selected]
    fit, valid = stratified_holdout(screen_y, config.validation_fraction, split_seed)
    weights, optimization, initial = optimize_spec(
        spec,
        screen_x[fit],
        screen_y[fit],
        seed=config.seed,
        init_scale=config.init_scale,
        affine_scale_center=config.affine_scale_center,
        affine_scale_jitter=config.affine_scale_jitter,
        maxiter=config.maxiter,
        n_restarts=config.n_restarts,
    )
    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_screen_{config.label}_"
        f"split{split_seed}_init{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "mode": "train_only_holdout_spec_screen",
        "architecture": spec.name,
        "label": config.label,
        "submission_created": False,
        "public_test_used": False,
        "classical_predictive_model_used": False,
        "split_seed": split_seed,
        "init_seed": config.seed,
        "config": _payload_config(config),
        "data": data_info,
        "circuit": {
            "qubits": spec.n_qubits,
            "weight_count": spec.n_weights,
            "two_qubit_gate_count": spec.two_qubit_count(),
            "used_features_zero_based": list(spec.used_features()),
            "feature_uses": dict(sorted(spec.feature_uses().items())),
        },
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(valid)),
        "optimization": optimization,
        "fit_metrics": metrics(
            spec, screen_x[fit], screen_y[fit], weights,
            shots=config.shots, seed=split_seed + 1,
            threshold=config.decision_threshold,
        ),
        "validation_metrics": metrics(
            spec, screen_x[valid], screen_y[valid], weights,
            shots=config.shots, seed=split_seed + 2,
            threshold=config.decision_threshold,
        ),
        "qiskit_equivalence": qiskit_equivalence_report(spec, screen_x[valid], weights),
        "trained_feature_causal_report": feature_causal_report(spec, screen_x, weights),
        "selected_weights": [float(value) for value in weights],
    }
    (run_dir / "screen_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return run_dir


def _stratified_folds(y: np.ndarray, folds: int, seed: int):
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[] for _ in range(folds)]
    for label in (0, 1):
        shuffled = rng.permutation(np.flatnonzero(y == label))
        for index, part in enumerate(np.array_split(shuffled, folds)):
            parts[index].append(part)
    all_indices = np.arange(len(y), dtype=int)
    result = []
    for group in parts:
        valid = np.sort(np.concatenate(group))
        mask = np.ones(len(y), dtype=bool)
        mask[valid] = False
        result.append((all_indices[mask], valid))
    return result


def run_spec_cross_validation(spec: CircuitSpec, config: SpecTrainConfig) -> Path:
    """Genuine out-of-fold probabilities and a train-only threshold."""
    x, y, data_info = load_raw_train(config.train_csv)
    split_seed = config.seed if config.split_seed is None else config.split_seed
    oof = np.full(len(y), np.nan, dtype=float)
    fold_results = []
    stage_oof: list[np.ndarray] | None = None
    stage_temperatures: list[float | None] = []
    splits = _stratified_folds(y, config.folds, split_seed)
    shared = dict(
        seed=config.seed,
        init_scale=config.init_scale,
        affine_scale_center=config.affine_scale_center,
        affine_scale_jitter=config.affine_scale_jitter,
        maxiter=config.maxiter,
        n_restarts=config.n_restarts,
    )
    for index, (fit, valid) in enumerate(splits):
        if config.objective == "balanced_bce":
            weights, optimization, _ = optimize_spec(spec, x[fit], y[fit], **shared)
        else:
            weights, optimization, _ = optimize_spec_annealed(
                spec,
                x[fit],
                y[fit],
                surrogate=config.objective,
                threshold=config.decision_threshold,
                temperature_start=config.temperature_start,
                temperature_stop=config.temperature_stop,
                anneal_stages=config.anneal_stages,
                stage_maxiter=config.stage_maxiter,
                **shared,
            )
        if "stage_weights" in optimization:
            if stage_oof is None:
                stage_oof = [
                    np.full(len(y), np.nan, dtype=float)
                    for _ in optimization["stage_weights"]
                ]
                stage_temperatures = optimization["stage_temperatures"]
            for position, values in enumerate(optimization["stage_weights"]):
                stage_oof[position][valid] = exact_probabilities(
                    spec, x[valid], np.asarray(values, dtype=float)
                )
            optimization = {
                k: v for k, v in optimization.items() if k != "stage_weights"
            }
        probability = exact_probabilities(spec, x[valid], weights)
        oof[valid] = probability
        fold_results.append(
            {
                "fold": index + 1,
                "fit_rows": int(len(fit)),
                "validation_rows": int(len(valid)),
                "balanced_accuracy_at_0_5": balanced_accuracy(y[valid], probability >= 0.5),
                "balanced_binary_cross_entropy": balanced_binary_cross_entropy(
                    y[valid], probability
                ),
                "optimization": optimization,
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("Cross-validation left rows unpredicted.")

    grid = np.linspace(0.25, 0.75, 201)
    scores = np.asarray([balanced_accuracy(y, oof >= t) for t in grid])
    best = int(np.argmax(scores))
    rng = np.random.default_rng(split_seed + 1000)
    shot_probability = rng.binomial(config.shots, oof) / config.shots

    # Out-of-fold score at every annealing temperature, so the stop temperature
    # is chosen on held-out rows rather than on the training fit.
    temperature_curve: list[dict] = []
    if stage_oof is not None:
        for position, probabilities in enumerate(stage_oof):
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Stage OOF generation left rows unpredicted.")
            stage_scores = np.asarray(
                [balanced_accuracy(y, probabilities >= t) for t in grid]
            )
            top = int(np.argmax(stage_scores))
            shot = rng.binomial(config.shots, probabilities) / config.shots
            temperature_curve.append(
                {
                    "stage": position,
                    "temperature": stage_temperatures[position],
                    "is_warm_start_only": position == 0,
                    "per_fold_balanced_accuracy_at_selected_threshold": [
                        balanced_accuracy(
                            y[rows], probabilities[rows] >= grid[top]
                        )
                        for _, rows in splits
                    ],
                    "balanced_accuracy_at_0_5": balanced_accuracy(y, probabilities >= 0.5),
                    "selected_threshold": float(grid[top]),
                    "balanced_accuracy_at_selected_threshold": float(stage_scores[top]),
                    "shot_balanced_accuracy_at_selected_threshold": balanced_accuracy(
                        y, shot >= grid[top]
                    ),
                }
            )

    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_cv_{config.label}_"
        f"split{split_seed}_init{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "mode": "quantum_only_out_of_fold_threshold_selection",
        "architecture": spec.name,
        "label": config.label,
        "submission_created": False,
        "public_test_used": False,
        "classical_predictive_model_used": False,
        "parameter_transfer_used": False,
        "threshold_source": "out_of_fold_circuit_probabilities_only",
        "config": _payload_config(config),
        "data": data_info,
        "circuit": {
            "qubits": spec.n_qubits,
            "weight_count": spec.n_weights,
            "two_qubit_gate_count": spec.two_qubit_count(),
            "used_features_zero_based": list(spec.used_features()),
        },
        "balanced_accuracy_at_0_5": balanced_accuracy(y, oof >= 0.5),
        "selected_threshold": float(grid[best]),
        "balanced_accuracy_at_selected_threshold": float(scores[best]),
        "shot_balanced_accuracy_at_selected_threshold": balanced_accuracy(
            y, shot_probability >= grid[best]
        ),
        "balanced_binary_cross_entropy": balanced_binary_cross_entropy(y, oof),
        "annealing_temperature_curve": temperature_curve,
        "fold_results": fold_results,
        "oof_probability": [float(value) for value in oof],
    }
    (run_dir / "cross_validation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return run_dir
