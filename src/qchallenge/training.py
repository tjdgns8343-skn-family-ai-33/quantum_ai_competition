"""Direct optimization of quantum measurement loss; no classical pre-training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from scipy.optimize import minimize

from .circuit import (
    N_QUBITS,
    N_WEIGHTS,
    READOUT_QUBIT,
    FEATURE_LAYOUTS,
    build_training_circuit,
    build_unitary,
    export_submission_qasm,
    normalize_selected_features,
)
from .data import load_raw_train
from .metrics import balanced_accuracy, balanced_binary_cross_entropy, binary_cross_entropy

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
    feature_layout: str = "sequential"
    selected_features: tuple[int, ...] = tuple(range(8))
    pack_selected_features: bool = False


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


def make_qnn(
    shots: int,
    seed: int,
    feature_layout: str = "sequential",
    selected_features: tuple[int, ...] | None = None,
    pack_selected_features: bool = False,
) -> SamplerQNN:
    circuit, inputs, weights = build_training_circuit(
        feature_layout, selected_features, pack_selected_features
    )
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    return SamplerQNN(
        circuit=circuit,
        sampler=sampler,
        input_params=inputs,
        weight_params=weights,
        interpret=lambda bitstring: (bitstring >> READOUT_QUBIT) & 1,
        output_shape=2,
    )


def make_exact_qnn(
    seed: int,
    feature_layout: str = "sequential",
    selected_features: tuple[int, ...] | None = None,
    pack_selected_features: bool = False,
) -> EstimatorQNN:
    """Exact statevector expectation of Z on q0 for the same quantum circuit."""
    circuit, inputs, weights = build_unitary(
        feature_layout, selected_features, pack_selected_features
    )
    pauli = ["I"] * N_QUBITS
    pauli[N_QUBITS - 1 - READOUT_QUBIT] = "Z"
    observable = SparsePauliOp.from_list([("".join(pauli), 1.0)])
    return EstimatorQNN(
        circuit=circuit,
        estimator=StatevectorEstimator(default_precision=0.0, seed=seed),
        observables=observable,
        input_params=inputs,
        weight_params=weights,
    )


def quantum_probabilities(qnn: SamplerQNN, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.asarray(qnn.forward(x, weights), dtype=float)
    if result.shape != (len(x), 2):
        raise RuntimeError(f"Unexpected SamplerQNN output shape: {result.shape}")
    return result[:, 1]


def exact_quantum_probabilities(
    qnn: EstimatorQNN, x: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    expectation = np.asarray(qnn.forward(x, weights), dtype=float)
    if expectation.shape != (len(x), 1):
        raise RuntimeError(f"Unexpected EstimatorQNN output shape: {expectation.shape}")
    return np.clip((1.0 - expectation[:, 0]) / 2.0, 0.0, 1.0)


def optimize_quantum_loss(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    init_scale: float,
    maxiter: int,
    shots: int,
    feature_layout: str,
    selected_features: tuple[int, ...],
    pack_selected_features: bool = False,
) -> tuple[np.ndarray, dict]:
    """Optimize only BCE computed from SamplerQNN measurement probabilities."""
    selected = normalize_selected_features(selected_features)
    if x.shape[1] != len(selected):
        raise ValueError(
            f"Expected {len(selected)} selected input columns, got {x.shape[1]}."
        )
    qnn = make_exact_qnn(
        seed, feature_layout, selected, pack_selected_features
    )
    initial = random_initial_point(seed, init_scale)
    history: list[float] = []

    def objective(weight_values: np.ndarray) -> float:
        value = balanced_binary_cross_entropy(
            y, exact_quantum_probabilities(qnn, x, weight_values)
        )
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
        "training_emulator": "qiskit.primitives.StatevectorEstimator",
        "training_probability": "exact_q0_probability_from_Z_expectation",
        "feature_layout": feature_layout,
        "selected_features_zero_based": list(selected),
        "pack_selected_features": pack_selected_features,
        "objective": "balanced_binary_cross_entropy_of_quantum_circuit_probability",
        "success": bool(result.success),
        "message": str(result.message),
        "evaluations": int(result.nfev),
        "initial_loss": float(history[0]),
        "final_loss": float(history[-1]),
        "best_observed_loss": float(min(history)),
        "loss_history": history,
    }


def _evaluate(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    shots: int,
    seed: int,
    feature_layout: str,
    selected_features: tuple[int, ...],
    pack_selected_features: bool = False,
) -> dict:
    probability = quantum_probabilities(
        make_qnn(
            shots, seed, feature_layout, selected_features,
            pack_selected_features,
        ),
        x,
        weights,
    )
    prediction = (probability >= FIXED_THRESHOLD).astype(int)
    return {
        "rows": int(len(y)),
        "balanced_accuracy_at_fixed_0_5": balanced_accuracy(y, prediction),
        "binary_cross_entropy": binary_cross_entropy(y, probability),
        "mean_probability": float(np.mean(probability)),
    }


def run_screen(config: TrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    selected = normalize_selected_features(config.selected_features)
    selected_x = x[:, selected]
    fit, valid = stratified_holdout(y, config.validation_fraction, config.seed)
    weights, optimization = optimize_quantum_loss(
        selected_x[fit], y[fit], seed=config.seed, init_scale=config.init_scale,
        maxiter=config.maxiter, shots=config.shots,
        feature_layout=config.feature_layout,
        selected_features=selected,
        pack_selected_features=config.pack_selected_features,
    )
    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_screen_b1_"
        f"{config.feature_layout}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "mode": "train_only_holdout_screen",
        "config": {**asdict(config), "train_csv": str(config.train_csv), "artifacts_dir": str(config.artifacts_dir)},
        "data": data_info,
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(valid)),
        "optimization": optimization,
        "fit_metrics": _evaluate(
            selected_x[fit], y[fit], weights, config.shots, config.seed + 1,
            config.feature_layout, selected,
            config.pack_selected_features,
        ),
        "validation_metrics": _evaluate(
            selected_x[valid], y[valid], weights, config.shots, config.seed + 2,
            config.feature_layout, selected,
            config.pack_selected_features,
        ),
        "submission_created": False,
    }
    (run_dir / "screen_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_dir


def run_final(config: TrainConfig) -> Path:
    x, y, data_info = load_raw_train(config.train_csv)
    selected = normalize_selected_features(config.selected_features)
    selected_x = x[:, selected]
    weights, optimization = optimize_quantum_loss(
        selected_x, y, seed=config.seed, init_scale=config.init_scale,
        maxiter=config.maxiter, shots=config.shots,
        feature_layout=config.feature_layout,
        selected_features=selected,
        pack_selected_features=config.pack_selected_features,
    )
    run_dir = config.artifacts_dir / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_final_b1_"
        f"{config.feature_layout}_seed{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    constraint = export_submission_qasm(
        run_dir / "classifier.qasm", config.feature_layout, selected,
        config.pack_selected_features,
    )
    weights_payload = {f"theta_{index}": float(value) for index, value in enumerate(weights)}
    weights_payload["threshold"] = FIXED_THRESHOLD
    (run_dir / "weights.json").write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")
    provenance = {
        "parameter_generation": "random_initialization_then_direct_quantum_measurement_loss_optimization",
        "initialization": {"distribution": "uniform", "seed": config.seed, "range": [-config.init_scale, config.init_scale]},
        "training_quantum_emulator": "qiskit.primitives.StatevectorEstimator",
        "validation_quantum_emulator": "qiskit.primitives.StatevectorSampler",
        "loss_source": "exact EstimatorQNN q0 probability versus raw train label",
        "loss": "balanced binary cross-entropy; equal mean loss per class",
        "feature_layout": {
            "name": config.feature_layout,
            "zero_based_indices": FEATURE_LAYOUTS[config.feature_layout],
            "selection": "pre_registered_train_only_quantum_validation",
        },
        "selected_raw_features_zero_based": list(selected),
        "selected_raw_features_csv": [f"x{index + 1}" for index in selected],
        "pack_selected_features": config.pack_selected_features,
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
        "full_train_metrics": _evaluate(
            selected_x, y, weights, config.shots, config.seed + 1,
            config.feature_layout, selected,
            config.pack_selected_features,
        ),
        "constraint_report": constraint,
    }
    (run_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if config.pack_selected_features:
        first_layout, second_layout = selected[:4], selected[4:]
    else:
        first_layout, second_layout = FEATURE_LAYOUTS[config.feature_layout]
    selected_set = set(selected)
    first_names = ",".join(
        f"x{index + 1}->q{qubit}"
        for qubit, index in enumerate(first_layout)
        if index in selected_set
    ) or "none"
    second_names = ",".join(
        f"x{index + 1}->q{qubit}"
        for qubit, index in enumerate(second_layout)
        if index in selected_set
    ) or "none"
    unused_names = ",".join(
        f"x{index + 1}" for index in range(8) if index not in selected_set
    ) or "none"
    note = (
        f"Feature layout {config.feature_layout}: raw mappings ({first_names}) are encoded "
        f"directly with RY, and raw mappings ({second_names}) are encoded directly with RZ. "
        f"Unused raw features: {unused_names}. "
        "Each encoding gate contains exactly one unmodified raw feature. Two CX-tree variational blocks "
        "use theta_0-theta_15 and only q0 is measured. All theta values started from a label-independent "
        "uniform random initialization and were optimized exclusively against balanced cross-entropy computed from "
        "StatevectorEstimator/EstimatorQNN q0 probabilities on public_train.csv. No classical predictive, "
        "surrogate, teacher, warm-start, transferred coefficient, scaling, PCA, imputation, or augmentation was used. "
        "The decision threshold is fixed at 0.5."
    )
    (run_dir / "encoding_note.txt").write_text(note + "\n", encoding="utf-8")
    return run_dir
