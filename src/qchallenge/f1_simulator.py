"""Vectorized exact statevector and adjoint gradients for the F1 circuit.

Two choices keep the eight-qubit circuit trainable at full data size.

First, the state is stored as ``(2,) * 8 + (rows,)`` so the batch axis is the
fastest-varying one.  Every gate then reads and writes long contiguous runs
instead of the stride-2 interleaving a leading batch axis would force on the
low-index qubits, and per-row angles broadcast without any reshaping.

Second, the eight-qubit state is 16 times larger than C1's, so caching one
pre-gate state per operation would need gigabytes.  This module therefore uses
textbook adjoint differentiation: the forward pass keeps only the final state,
and the backward pass reconstructs each pre-gate state by applying the inverse
gate.  Peak memory stays at two statevectors regardless of circuit depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .f1_circuit import (
    F1_DEFAULT_BLOCKS,
    F1_N_FEATURES,
    F1_N_QUBITS,
    F1_PARAMETERS_PER_BLOCK,
    F1_READOUT_QUBIT,
    F1_TREE_FUNNEL,
    f1_reupload_blocks,
    f1_weight_count,
)

_DIMENSION = 1 << F1_N_QUBITS
_BASIS = np.arange(_DIMENSION, dtype=np.int64)
_Z_READOUT_SIGN = np.where(((_BASIS >> F1_READOUT_QUBIT) & 1) == 0, 1.0, -1.0)
_TENSOR_SHAPE = (2,) * F1_N_QUBITS

# Rows are processed in chunks so the whole statevector stays cache-resident
# across the hundreds of gate applications in one forward/backward sweep.
# Loss and gradient are plain sums over rows, so chunking is exact.  Measured
# on the full 6,000-row train set, 64 rows per chunk was about four times
# faster than running every row in one batch.
F1_CHUNK_ROWS = 64


def _row_chunks(values: np.ndarray, chunk_rows: int) -> list[np.ndarray]:
    if chunk_rows < 1:
        raise ValueError("F1 chunk_rows must be positive.")
    return [
        values[start : start + chunk_rows]
        for start in range(0, max(len(values), 1), chunk_rows)
    ]


@dataclass(frozen=True)
class _Operation:
    kind: str
    angle: np.ndarray | float | None = None
    qubit: int | None = None
    control: int | None = None
    target: int | None = None
    # Each tuple is (weight index, per-row chain-rule factor or None for 1).
    parameter_factors: tuple[tuple[int, np.ndarray | None], ...] = ()


def _axis(qubit: int) -> int:
    """Tensor axis of a qubit; the trailing axis is the batch.

    A flat Qiskit index is ``sum_q b_q 2**q``, so reshaping the qubit axes to
    ``(2,) * n`` puts the most significant qubit first and qubit ``q`` on axis
    ``n - 1 - q``.
    """
    return F1_N_QUBITS - 1 - qubit


def _half(state: np.ndarray, axis: int, bit: int) -> np.ndarray:
    """Return a writable view of the ``bit`` branch of ``axis``."""
    return state[(slice(None),) * axis + (bit,)]


def _trig(angle: np.ndarray | float, inverse: bool) -> tuple:
    values = np.asarray(angle, dtype=float)
    if inverse:
        values = -values
    return np.cos(values / 2.0), np.sin(values / 2.0)


def _phase(angle: np.ndarray | float, inverse: bool) -> tuple:
    values = np.asarray(angle, dtype=float)
    if inverse:
        values = -values
    return np.exp(-0.5j * values), np.exp(0.5j * values)


def _apply_ry(
    state: np.ndarray, angle: np.ndarray | float, qubit: int, *, inverse: bool = False
) -> np.ndarray:
    cosine, sine = _trig(angle, inverse)
    axis = _axis(qubit)
    a0 = _half(state, axis, 0)
    a1 = _half(state, axis, 1)
    new0 = cosine * a0 - sine * a1
    new1 = sine * a0 + cosine * a1
    a0[...] = new0
    a1[...] = new1
    return state


def _apply_rz(
    state: np.ndarray, angle: np.ndarray | float, qubit: int, *, inverse: bool = False
) -> np.ndarray:
    negative, positive = _phase(angle, inverse)
    axis = _axis(qubit)
    _half(state, axis, 0)[...] *= negative
    _half(state, axis, 1)[...] *= positive
    return state


def _apply_cx(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """CX is its own inverse, so the same routine undoes it."""
    control_axis = _axis(control)
    target_axis = _axis(target)
    branch = _half(state, control_axis, 1)
    # Indexing the control axis removes it, shifting every later axis down.
    reduced = target_axis if target_axis < control_axis else target_axis - 1
    t0 = _half(branch, reduced, 0)
    t1 = _half(branch, reduced, 1)
    swap = t0.copy()
    t0[...] = t1
    t1[...] = swap
    return state


def _branch_sum(values: np.ndarray, rows: int) -> np.ndarray:
    """Sum every qubit branch of a half-state, leaving one value per row."""
    return values.reshape(-1, rows).sum(axis=0)


def _ry_angle_derivative(
    state: np.ndarray,
    adjoint: np.ndarray,
    angle: np.ndarray | float,
    qubit: int,
    rows: int,
) -> np.ndarray:
    """Return ``2 Re <adjoint| dRY/dtheta |state>`` per row.

    ``state`` must already be the pre-gate state.  Only the two branches of the
    rotated qubit are touched, so no full-size intermediate is allocated.
    """
    cosine, sine = _trig(angle, False)
    axis = _axis(qubit)
    a0 = _half(state, axis, 0)
    a1 = _half(state, axis, 1)
    b0 = _half(adjoint, axis, 0)
    b1 = _half(adjoint, axis, 1)
    derivative0 = -0.5 * (sine * a0 + cosine * a1)
    derivative1 = 0.5 * (cosine * a0 - sine * a1)
    overlap = np.conjugate(b0) * derivative0
    overlap += np.conjugate(b1) * derivative1
    return 2.0 * _branch_sum(overlap.real, rows)


def _rz_angle_derivative(
    state: np.ndarray,
    adjoint: np.ndarray,
    angle: np.ndarray | float,
    qubit: int,
    rows: int,
) -> np.ndarray:
    negative, positive = _phase(angle, False)
    axis = _axis(qubit)
    a0 = _half(state, axis, 0)
    a1 = _half(state, axis, 1)
    b0 = _half(adjoint, axis, 0)
    b1 = _half(adjoint, axis, 1)
    overlap = np.conjugate(b0) * ((-0.5j * negative) * a0)
    overlap += np.conjugate(b1) * ((0.5j * positive) * a1)
    return 2.0 * _branch_sum(overlap.real, rows)


def _validate_inputs(
    x: np.ndarray, weights: np.ndarray, n_blocks: int
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(x, dtype=float)
    parameters = np.asarray(weights, dtype=float)
    expected = f1_weight_count(n_blocks)
    if features.ndim != 2 or features.shape[1] != F1_N_FEATURES:
        raise ValueError(f"F1 expects shape (n, {F1_N_FEATURES}); got {features.shape}.")
    if parameters.shape != (expected,):
        raise ValueError(f"F1 expects {expected} weights; got {parameters.shape}.")
    if not np.isfinite(features).all() or not np.isfinite(parameters).all():
        raise ValueError("F1 inputs and weights must be finite.")
    return features, parameters


def _forward(
    x: np.ndarray, weights: np.ndarray, n_blocks: int
) -> tuple[np.ndarray, list[_Operation]]:
    features, parameters = _validate_inputs(x, weights, n_blocks)
    rows = len(features)
    state = np.zeros((*_TENSOR_SHAPE, rows), dtype=np.complex128)
    state[(0,) * F1_N_QUBITS] = 1.0
    operations: list[_Operation] = []

    for block_index, block_features in enumerate(f1_reupload_blocks(n_blocks)):
        offset = F1_PARAMETERS_PER_BLOCK * block_index
        for qubit, feature_index in enumerate(block_features):
            scale_index = offset + qubit
            bias_index = offset + F1_N_QUBITS + qubit
            column = np.ascontiguousarray(features[:, feature_index])
            angle = parameters[scale_index] * column + parameters[bias_index]
            _apply_ry(state, angle, qubit)
            operations.append(
                _Operation(
                    kind="ry",
                    angle=angle,
                    qubit=qubit,
                    parameter_factors=((scale_index, column), (bias_index, None)),
                )
            )
        for qubit in range(F1_N_QUBITS):
            parameter_index = offset + 2 * F1_N_QUBITS + qubit
            angle = float(parameters[parameter_index])
            _apply_rz(state, angle, qubit)
            operations.append(
                _Operation(
                    kind="rz",
                    angle=angle,
                    qubit=qubit,
                    parameter_factors=((parameter_index, None),),
                )
            )
        for qubit in range(F1_N_QUBITS):
            parameter_index = offset + 3 * F1_N_QUBITS + qubit
            angle = float(parameters[parameter_index])
            _apply_ry(state, angle, qubit)
            operations.append(
                _Operation(
                    kind="ry",
                    angle=angle,
                    qubit=qubit,
                    parameter_factors=((parameter_index, None),),
                )
            )
        for control, target in F1_TREE_FUNNEL:
            _apply_cx(state, control, target)
            operations.append(_Operation(kind="cx", control=control, target=target))

    readout_index = F1_PARAMETERS_PER_BLOCK * n_blocks
    angle = float(parameters[readout_index])
    _apply_ry(state, angle, F1_READOUT_QUBIT)
    operations.append(
        _Operation(
            kind="ry",
            angle=angle,
            qubit=F1_READOUT_QUBIT,
            parameter_factors=((readout_index, None),),
        )
    )
    return state, operations


def _readout_expectation(state: np.ndarray) -> np.ndarray:
    rows = state.shape[-1]
    flat = state.reshape(_DIMENSION, rows)
    return _Z_READOUT_SIGN @ (flat.real**2 + flat.imag**2)


def f1_exact_probabilities(
    x: np.ndarray,
    weights: np.ndarray,
    n_blocks: int = F1_DEFAULT_BLOCKS,
    *,
    chunk_rows: int = F1_CHUNK_ROWS,
) -> np.ndarray:
    features = np.asarray(x, dtype=float)
    parts = [
        (1.0 - _readout_expectation(_forward(chunk, weights, n_blocks)[0])) / 2.0
        for chunk in _row_chunks(features, chunk_rows)
    ]
    return np.clip(np.concatenate(parts), 0.0, 1.0)


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("F1 labels must be binary 0/1 values.")
    result = np.empty(len(labels), dtype=float)
    for label in (0, 1):
        mask = labels == label
        count = int(np.sum(mask))
        if count == 0:
            raise ValueError("Both classes must be present for balanced BCE.")
        result[mask] = 0.5 / count
    return result


def _chunk_value_and_gradient(
    x: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    weights: np.ndarray,
    n_blocks: int,
    epsilon: float,
) -> tuple[float, np.ndarray]:
    """Loss and gradient contributions of one row chunk."""
    state, operations = _forward(x, weights, n_blocks)
    rows = state.shape[-1]
    expectation = _readout_expectation(state)
    probability = np.clip((1.0 - expectation) / 2.0, epsilon, 1.0 - epsilon)
    loss = -np.sum(
        sample_weight
        * (labels * np.log(probability) + (1.0 - labels) * np.log1p(-probability))
    )

    dloss_dprobability = sample_weight * (
        (probability - labels) / (probability * (1.0 - probability))
    )
    dloss_dexpectation = -0.5 * dloss_dprobability
    gradient = np.zeros(f1_weight_count(n_blocks), dtype=float)

    # For E=<psi|Z_q0|psi>, the reverse state starts at Z_q0|psi>.  ``state`` is
    # rewound in lockstep so each pre-gate state is recovered without caching.
    adjoint = (
        state.reshape(_DIMENSION, rows) * _Z_READOUT_SIGN[:, None]
    ).reshape(state.shape)
    for operation in reversed(operations):
        if operation.kind == "cx":
            control, target = int(operation.control), int(operation.target)
            _apply_cx(state, control, target)
            _apply_cx(adjoint, control, target)
            continue

        qubit = int(operation.qubit)
        if operation.kind == "ry":
            _apply_ry(state, operation.angle, qubit, inverse=True)
            angle_derivative = _ry_angle_derivative(
                state, adjoint, operation.angle, qubit, rows
            )
            _apply_ry(adjoint, operation.angle, qubit, inverse=True)
        elif operation.kind == "rz":
            _apply_rz(state, operation.angle, qubit, inverse=True)
            angle_derivative = _rz_angle_derivative(
                state, adjoint, operation.angle, qubit, rows
            )
            _apply_rz(adjoint, operation.angle, qubit, inverse=True)
        else:
            raise RuntimeError(f"Unknown F1 operation: {operation.kind}")

        weighted_angle_derivative = dloss_dexpectation * angle_derivative
        for parameter_index, factor in operation.parameter_factors:
            if factor is None:
                gradient[parameter_index] += float(np.sum(weighted_angle_derivative))
            else:
                gradient[parameter_index] += float(
                    np.sum(weighted_angle_derivative * factor)
                )
    return float(loss), gradient


def f1_balanced_bce_value_and_gradient(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_blocks: int = F1_DEFAULT_BLOCKS,
    *,
    epsilon: float = 1e-9,
    chunk_rows: int = F1_CHUNK_ROWS,
) -> tuple[float, np.ndarray]:
    """Return exact circuit BCE and its analytic adjoint gradient.

    Class balancing is computed once over all rows, so summing the per-chunk
    contributions reproduces the whole-batch value and gradient exactly.
    """
    features = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=float)
    if len(features) != len(labels):
        raise ValueError("F1 requires one label per feature row.")
    sample_weight = _balanced_sample_weights(labels.astype(int))

    loss = 0.0
    gradient = np.zeros(f1_weight_count(n_blocks), dtype=float)
    for start in range(0, len(features), chunk_rows):
        stop = start + chunk_rows
        chunk_loss, chunk_gradient = _chunk_value_and_gradient(
            features[start:stop],
            labels[start:stop],
            sample_weight[start:stop],
            weights,
            n_blocks,
            epsilon,
        )
        loss += chunk_loss
        gradient += chunk_gradient
    return float(loss), gradient
