"""Direct optimization of quantum measurement loss; no classical pre-training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from scipy.optimize import minimize

from .circuit import N_WEIGHTS, READOUT_QUBIT, build_training_circuit, export_submission_qasm
from .data import load_raw_train
from .metrics import balanced_accuracy, binary_cross_entropy

FIXED_THRESHOLD = 0.5


@dataclass(frozen=True)
class TrainConfig:
    train_csv: Path
    artifacts_dir: Path = Path("artifacts")
    seed: int = 2026
    init_scale: float = 0.05
    maxiter: int = 120
    shots: int = 1024
    validation_fraction: float = 0.2


def random_initial_point(seed: int, scale: float) -> np.ndarray:
    """Generate label-independent parameters; no warm start input is accepted."""
    if scale < 0:
        raise ValueError("init_scale must be non-negative.")
    return np.random.default_rng(seed).uniform(-scale, scale, N_WEIGHTS)


def stratified_holdout(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5).")
    rng = np.random.default_rng(seed)
    valid_parts: list[np.ndarray] = []
    fit_parts: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        shuffled = rng.permutation(indices)
        count = max(1, int(round(len(indices) * fraction)))
        valid_parts.append(shuffled[:count])
        fit_parts.append(shuffled[count:])
    return np.sort(np.concatenate(fit_parts)), np.sort(np.concatenate(valid_parts))


def make_qnn(shots: int, seed: int) -> SamplerQNN:
    circuit, inputs, weights = build_training_circuit()
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    return SamplerQNN(
        circuit=circuit,
        sampler=sampler,
        input_params=inputs,
        weight_params=weights,
        interpret=lambda bitstring: (bitstring >> READOUT_QUBIT) & 1,
        output_shape=2,
    )


def quantum_probabilities(qnn: SamplerQNN, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.asarray(qnn.forward(x, weights), dtype=float)
    if result.shape != (len(x), 2):
        raise RuntimeError(f"Unexpected SamplerQNN output shape: {result.shape}")
    return result[:, 1]


def optimize_quantum_loss(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    init_scale: float,
    maxiter: int,
    shots: int,
) -> tuple[np.ndarray, dict]:
    """Optimize only BCE computed from SamplerQNN measurement probabilities."""
    qnn = make_qnn(shots, seed)
    initial = random_initial_point(seed, init_scale)
    history: list[float] = []

    def objective(weight_values: np.ndarray) -> float:
        value = binary_cross_entropy(y, quantum_probabilities(qnn, x, weight_values))
        history.append(value)
        return value

    result = minimize(
        objective,
        initial,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )
    return np.asarray(result.x, dtype=float), {
        "optimizer": "COBYLA",
        "objective": "binary_cross_entropy_of_SamplerQNN_measurement_probability",
        "success": bool(result.success),
        "message": str(result.message),
        "evaluations": int(result.nfev),
        "initial_loss": float(history[0]),
        "final_loss": float(history[-1]),
        "best_observed_loss": float(min(history)),
        "loss_history": history,
    }


def _evaluate(x: np.ndarray, y: np.ndarray, weights: np.ndarray, shots: int, seed: int) -> dict:
    probability = quantum_probabilities(make_qnn(shots, seed), x, weights)
    prediction = (probability >= FIXED_THRESHOLD).astype(int)
    return {
        "rows": int(len(y)),
        "balanced_accuracy_at_fixed_0_5": balanced_accuracy(y, prediction),
        "binary_cross_entropy": binary_cross_entropy(y, probability),
        "mean_probability": float(np.mean(probability)),
    }


def run_screen(config: TrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    fit, valid = stratified_holdout(y, config.validation_fraction, config.seed)
    weights, optimization = optimize_quantum_loss(
        x[fit], y[fit], seed=config.seed, init_scale=config.init_scale,
        maxiter=config.maxiter, shots=config.shots,
    )
    run_dir = config.artifacts_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_screen_b1_seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "mode": "train_only_holdout_screen",
        "config": {**asdict(config), "train_csv": str(config.train_csv), "artifacts_dir": str(config.artifacts_dir)},
        "data": data_info,
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(valid)),
        "optimization": optimization,
        "fit_metrics": _evaluate(x[fit], y[fit], weights, config.shots, config.seed + 1),
        "validation_metrics": _evaluate(x[valid], y[valid], weights, config.shots, config.seed + 2),
        "submission_created": False,
    }
    (run_dir / "screen_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir


def run_final(config: TrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    weights, optimization = optimize_quantum_loss(
        x, y, seed=config.seed, init_scale=config.init_scale,
        maxiter=config.maxiter, shots=config.shots,
    )
    run_dir = config.artifacts_dir / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_final_b1_seed{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    constraint = export_submission_qasm(run_dir / "classifier.qasm")
    weights_payload = {f"theta_{index}": float(value) for index, value in enumerate(weights)}
    weights_payload["threshold"] = FIXED_THRESHOLD
    (run_dir / "weights.json").write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")
    provenance = {
        "parameter_generation": "random_initialization_then_direct_quantum_measurement_loss_optimization",
        "initialization": {"distribution": "uniform", "seed": config.seed, "range": [-config.init_scale, config.init_scale]},
        "quantum_emulator": "qiskit.primitives.StatevectorSampler",
        "loss_source": "SamplerQNN measurement probability versus raw train label",
        "feature_processing": "none",
        "classical_predictive_model": None,
        "surrogate_model": None,
        "teacher_model": None,
        "warm_start": False,
        "transferred_coefficients": False,
        "decision_threshold": {"value": FIXED_THRESHOLD, "selection": "fixed_not_fitted"},
    }
    (run_dir / "parameter_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    metrics = {
        "mode": "full_train_direct_quantum_optimization",
        "config": {**asdict(config), "train_csv": str(config.train_csv), "artifacts_dir": str(config.artifacts_dir)},
        "data": data_info,
        "optimization": optimization,
        "full_train_metrics": _evaluate(x, y, weights, config.shots, config.seed + 1),
        "constraint_report": constraint,
    }
    (run_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    note = (
        "Raw CSV features x1-x4 are mapped directly to x_0-x_3 and encoded with RY on q0-q3. "
        "Raw x5-x8 are mapped directly to x_4-x_7 and encoded with RZ on q0-q3. "
        "Each encoding gate contains exactly one unmodified raw feature. Two CX-tree variational blocks "
        "use theta_0-theta_15 and only q0 is measured. All theta values started from a label-independent "
        "uniform random initialization and were optimized exclusively against cross-entropy computed from "
        "StatevectorSampler/SamplerQNN measurement probabilities on public_train.csv. No classical predictive, "
        "surrogate, teacher, warm-start, transferred coefficient, scaling, PCA, imputation, or augmentation was used. "
        "The decision threshold is fixed at 0.5."
    )
    (run_dir / "encoding_note.txt").write_text(note + "\n", encoding="utf-8")
    return run_dir

