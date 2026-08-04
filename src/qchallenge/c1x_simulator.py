"""Vectorized exact statevector and adjoint gradients for the C1X circuit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .c1x_circuit import (
    C1X_CAUSAL_FUNNEL,
    C1X_N_FEATURES,
    C1X_N_QUBITS,
    C1X_N_WEIGHTS,
    C1X_READOUT_QUBIT,
    C1X_REUPLOAD_BLOCKS,
)


@dataclass(frozen=True)
class _Operation:
    kind: str
    before: np.ndarray
    angle: np.ndarray | float | None = None
    qubit: int | None = None
    control: int | None = None
    target: int | None = None
    # Each tuple is (weight index, per-row chain-rule factor or None for 1).
    parameter_factors: tuple[tuple[int, np.ndarray | None], ...] = ()


_DIMENSION = 1 << C1X_N_QUBITS
_BASIS = np.arange(_DIMENSION, dtype=np.int64)
_PAIR_INDICES = {
    qubit: (
        _BASIS[((_BASIS >> qubit) & 1) == 0],
        _BASIS[((_BASIS >> qubit) & 1) == 1],
    )
    for qubit in range(C1X_N_QUBITS)
}
_Z_READOUT_SIGN = np.where(
    ((_BASIS >> C1X_READOUT_QUBIT) & 1) == 0, 1.0, -1.0
)


def _angle_column(angle: np.ndarray | float) -> np.ndarray | float:
    if np.ndim(angle) == 0:
        return np.asarray(angle).item()
    return np.asarray(angle)[:, None]


def _apply_ry(
    state: np.ndarray, angle: np.ndarray | float, qubit: int, *, inverse: bool = False
) -> np.ndarray:
    effective = -np.asarray(angle, dtype=float) if inverse else np.asarray(angle, dtype=float)
    cosine = _angle_column(np.cos(effective / 2.0))
    sine = _angle_column(np.sin(effective / 2.0))
    zero, one = _PAIR_INDICES[qubit]
    a0 = state[:, zero]
    a1 = state[:, one]
    result = state.copy()
    result[:, zero] = cosine * a0 - sine * a1
    result[:, one] = sine * a0 + cosine * a1
    return result


def _differentiate_ry(
    state: np.ndarray, angle: np.ndarray | float, qubit: int
) -> np.ndarray:
    cosine = _angle_column(np.cos(np.asarray(angle, dtype=float) / 2.0))
    sine = _angle_column(np.sin(np.asarray(angle, dtype=float) / 2.0))
    zero, one = _PAIR_INDICES[qubit]
    a0 = state[:, zero]
    a1 = state[:, one]
    result = np.zeros_like(state)
    result[:, zero] = -0.5 * sine * a0 - 0.5 * cosine * a1
    result[:, one] = 0.5 * cosine * a0 - 0.5 * sine * a1
    return result


def _apply_rz(
    state: np.ndarray, angle: np.ndarray | float, qubit: int, *, inverse: bool = False
) -> np.ndarray:
    effective = -np.asarray(angle, dtype=float) if inverse else np.asarray(angle, dtype=float)
    negative = _angle_column(np.exp(-0.5j * effective))
    positive = _angle_column(np.exp(0.5j * effective))
    zero, one = _PAIR_INDICES[qubit]
    result = state.copy()
    result[:, zero] = negative * state[:, zero]
    result[:, one] = positive * state[:, one]
    return result


def _differentiate_rz(
    state: np.ndarray, angle: np.ndarray | float, qubit: int
) -> np.ndarray:
    values = np.asarray(angle, dtype=float)
    negative = _angle_column((-0.5j) * np.exp(-0.5j * values))
    positive = _angle_column((0.5j) * np.exp(0.5j * values))
    zero, one = _PAIR_INDICES[qubit]
    result = np.zeros_like(state)
    result[:, zero] = negative * state[:, zero]
    result[:, one] = positive * state[:, one]
    return result


def _apply_cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
    low = _BASIS[
        (((_BASIS >> control) & 1) == 1)
        & (((_BASIS >> target) & 1) == 0)
    ]
    high = low | (1 << target)
    result = state.copy()
    result[:, low] = state[:, high]
    result[:, high] = state[:, low]
    return result


def _validate_inputs(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(x, dtype=float)
    parameters = np.asarray(weights, dtype=float)
    if features.ndim != 2 or features.shape[1] != C1X_N_FEATURES:
        raise ValueError(f"C1X expects shape (n, {C1X_N_FEATURES}); got {features.shape}.")
    if parameters.shape != (C1X_N_WEIGHTS,):
        raise ValueError(f"C1X expects {C1X_N_WEIGHTS} weights; got {parameters.shape}.")
    if not np.isfinite(features).all() or not np.isfinite(parameters).all():
        raise ValueError("C1X inputs and weights must be finite.")
    return features, parameters


def _forward_with_operations(
    x: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, list[_Operation]]:
    features, parameters = _validate_inputs(x, weights)
    state = np.zeros((len(features), _DIMENSION), dtype=np.complex128)
    state[:, 0] = 1.0
    operations: list[_Operation] = []

    for block_index, block_features in enumerate(C1X_REUPLOAD_BLOCKS):
        offset = 16 * block_index
        for qubit, feature_index in enumerate(block_features):
            scale_index = offset + qubit
            bias_index = offset + 4 + qubit
            angle = parameters[scale_index] * features[:, feature_index] + parameters[bias_index]
            before = state
            state = _apply_ry(state, angle, qubit)
            operations.append(
                _Operation(
                    kind="ry",
                    before=before,
                    angle=angle,
                    qubit=qubit,
                    parameter_factors=(
                        (scale_index, features[:, feature_index]),
                        (bias_index, None),
                    ),
                )
            )
        for qubit in range(C1X_N_QUBITS):
            parameter_index = offset + 8 + qubit
            angle = float(parameters[parameter_index])
            before = state
            state = _apply_rz(state, angle, qubit)
            operations.append(
                _Operation(
                    kind="rz",
                    before=before,
                    angle=angle,
                    qubit=qubit,
                    parameter_factors=((parameter_index, None),),
                )
            )
        for qubit in range(C1X_N_QUBITS):
            parameter_index = offset + 12 + qubit
            angle = float(parameters[parameter_index])
            before = state
            state = _apply_ry(state, angle, qubit)
            operations.append(
                _Operation(
                    kind="ry",
                    before=before,
                    angle=angle,
                    qubit=qubit,
                    parameter_factors=((parameter_index, None),),
                )
            )
        for control, target in C1X_CAUSAL_FUNNEL:
            before = state
            state = _apply_cx(state, control, target)
            operations.append(
                _Operation(
                    kind="cx", before=before, control=control, target=target
                )
            )

    before = state
    readout_parameter_index = C1X_N_WEIGHTS - 1
    angle = float(parameters[readout_parameter_index])
    state = _apply_ry(state, angle, C1X_READOUT_QUBIT)
    operations.append(
        _Operation(
            kind="ry",
            before=before,
            angle=angle,
            qubit=C1X_READOUT_QUBIT,
            parameter_factors=((readout_parameter_index, None),),
        )
    )
    return state, operations


def c1x_exact_probabilities(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    state, _ = _forward_with_operations(x, weights)
    expectation = np.sum(np.abs(state) ** 2 * _Z_READOUT_SIGN[None, :], axis=1)
    return np.clip((1.0 - expectation.real) / 2.0, 0.0, 1.0)


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    result = np.empty(len(labels), dtype=float)
    for label in (0, 1):
        mask = labels == label
        count = int(np.sum(mask))
        if count == 0:
            raise ValueError("Both classes must be present for balanced BCE.")
        result[mask] = 0.5 / count
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("C1X labels must be binary 0/1 values.")
    return result


def c1x_balanced_bce_value_and_gradient(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, *, epsilon: float = 1e-9
) -> tuple[float, np.ndarray]:
    """Return exact circuit BCE and its analytic adjoint gradient."""
    labels = np.asarray(y, dtype=float)
    state, operations = _forward_with_operations(x, weights)
    expectation = np.sum(np.abs(state) ** 2 * _Z_READOUT_SIGN[None, :], axis=1).real
    probability = np.clip((1.0 - expectation) / 2.0, epsilon, 1.0 - epsilon)
    sample_weight = _balanced_sample_weights(labels.astype(int))
    loss = -np.sum(
        sample_weight
        * (labels * np.log(probability) + (1.0 - labels) * np.log1p(-probability))
    )

    dloss_dprobability = sample_weight * (
        (probability - labels) / (probability * (1.0 - probability))
    )
    dloss_dexpectation = -0.5 * dloss_dprobability
    gradient = np.zeros(C1X_N_WEIGHTS, dtype=float)

    # For E=<psi|Z_q0|psi>, the reverse state starts at Z_q0|psi>.
    adjoint = state * _Z_READOUT_SIGN[None, :]
    for operation in reversed(operations):
        if operation.kind == "cx":
            adjoint = _apply_cx(
                adjoint, int(operation.control), int(operation.target)
            )
            continue
        if operation.kind == "ry":
            derivative = _differentiate_ry(
                operation.before, operation.angle, int(operation.qubit)
            )
            angle_derivative = 2.0 * np.real(
                np.sum(np.conjugate(adjoint) * derivative, axis=1)
            )
            adjoint = _apply_ry(
                adjoint, operation.angle, int(operation.qubit), inverse=True
            )
        elif operation.kind == "rz":
            derivative = _differentiate_rz(
                operation.before, operation.angle, int(operation.qubit)
            )
            angle_derivative = 2.0 * np.real(
                np.sum(np.conjugate(adjoint) * derivative, axis=1)
            )
            adjoint = _apply_rz(
                adjoint, operation.angle, int(operation.qubit), inverse=True
            )
        else:
            raise RuntimeError(f"Unknown C1X operation: {operation.kind}")

        weighted_angle_derivative = dloss_dexpectation * angle_derivative
        for parameter_index, factor in operation.parameter_factors:
            if factor is None:
                gradient[parameter_index] += float(np.sum(weighted_angle_derivative))
            else:
                gradient[parameter_index] += float(
                    np.sum(weighted_angle_derivative * factor)
                )
    return float(loss), gradient


