"""Exact statevector and adjoint gradients for any ``CircuitSpec``.

Carries over the two decisions that made the eight-qubit simulator practical:
the batch axis is stored last so every gate touches long contiguous runs, and
the backward pass rebuilds each pre-gate state by applying the inverse gate
instead of caching one state per gate.  Rows are processed in cache-sized
chunks; loss and gradient are sums over rows, so chunking is exact.
"""

from __future__ import annotations

import numpy as np

from .gatespec import CircuitSpec, validate
from .objectives import balanced_bce, balanced_sample_weights

CHUNK_ROWS = 64
N_FEATURES = 8


def _axis(spec: CircuitSpec, qubit: int) -> int:
    """Tensor axis of a qubit; the trailing axis is the batch."""
    return spec.n_qubits - 1 - qubit


def _half(state: np.ndarray, axis: int, bit: int) -> np.ndarray:
    return state[(slice(None),) * axis + (bit,)]


def _trig(angle, inverse: bool):
    values = np.asarray(angle, dtype=float)
    if inverse:
        values = -values
    return np.cos(values / 2.0), np.sin(values / 2.0)


def _phase(angle, inverse: bool):
    values = np.asarray(angle, dtype=float)
    if inverse:
        values = -values
    return np.exp(-0.5j * values), np.exp(0.5j * values)


def _apply_ry(state, angle, axis, *, inverse=False):
    cosine, sine = _trig(angle, inverse)
    a0 = _half(state, axis, 0)
    a1 = _half(state, axis, 1)
    new0 = cosine * a0 - sine * a1
    new1 = sine * a0 + cosine * a1
    a0[...] = new0
    a1[...] = new1
    return state


def _apply_rz(state, angle, axis, *, inverse=False):
    negative, positive = _phase(angle, inverse)
    _half(state, axis, 0)[...] *= negative
    _half(state, axis, 1)[...] *= positive
    return state


def _apply_cx(state, control_axis, target_axis):
    """CX is its own inverse, so the same routine undoes it."""
    branch = _half(state, control_axis, 1)
    reduced = target_axis if target_axis < control_axis else target_axis - 1
    t0 = _half(branch, reduced, 0)
    t1 = _half(branch, reduced, 1)
    swap = t0.copy()
    t0[...] = t1
    t1[...] = swap
    return state


def _branch_sum(values: np.ndarray, rows: int) -> np.ndarray:
    return values.reshape(-1, rows).sum(axis=0)


def _ry_derivative(state, adjoint, angle, axis, rows):
    cosine, sine = _trig(angle, False)
    a0, a1 = _half(state, axis, 0), _half(state, axis, 1)
    b0, b1 = _half(adjoint, axis, 0), _half(adjoint, axis, 1)
    overlap = np.conjugate(b0) * (-0.5 * (sine * a0 + cosine * a1))
    overlap += np.conjugate(b1) * (0.5 * (cosine * a0 - sine * a1))
    return 2.0 * _branch_sum(overlap.real, rows)


def _rz_derivative(state, adjoint, angle, axis, rows):
    negative, positive = _phase(angle, False)
    a0, a1 = _half(state, axis, 0), _half(state, axis, 1)
    b0, b1 = _half(adjoint, axis, 0), _half(adjoint, axis, 1)
    overlap = np.conjugate(b0) * ((-0.5j * negative) * a0)
    overlap += np.conjugate(b1) * ((0.5j * positive) * a1)
    return 2.0 * _branch_sum(overlap.real, rows)


def _plan(spec: CircuitSpec, features: np.ndarray, weights: np.ndarray) -> list:
    """Resolve every gate to (kind, axes, angle, parameter factors)."""
    plan = []
    for gate in spec.gates:
        if gate.kind == "cx":
            plan.append(
                (
                    "cx",
                    _axis(spec, gate.control),
                    _axis(spec, gate.target),
                    None,
                    (),
                )
            )
            continue
        axis = _axis(spec, gate.qubit)
        if gate.is_data:
            column = np.ascontiguousarray(features[:, gate.feature])
            angle = weights[gate.scale_index] * column + weights[gate.bias_index]
            factors = ((gate.scale_index, column), (gate.bias_index, None))
        else:
            angle = float(weights[gate.param_index])
            factors = ((gate.param_index, None),)
        plan.append((gate.kind, axis, None, angle, factors))
    return plan


def _readout_sign(spec: CircuitSpec) -> np.ndarray:
    dimension = 1 << spec.n_qubits
    basis = np.arange(dimension, dtype=np.int64)
    return np.where(((basis >> spec.readout_qubit) & 1) == 0, 1.0, -1.0)


def _forward(spec, features, weights):
    rows = len(features)
    state = np.zeros(((2,) * spec.n_qubits) + (rows,), dtype=np.complex128)
    state[(0,) * spec.n_qubits] = 1.0
    plan = _plan(spec, features, weights)
    for kind, first, second, angle, _ in plan:
        if kind == "cx":
            _apply_cx(state, first, second)
        elif kind == "ry":
            _apply_ry(state, angle, first)
        else:
            _apply_rz(state, angle, first)
    return state, plan


def _expectation(spec, state):
    rows = state.shape[-1]
    flat = state.reshape(1 << spec.n_qubits, rows)
    return _readout_sign(spec) @ (flat.real**2 + flat.imag**2)


def _validate_inputs(spec, x, weights):
    features = np.asarray(x, dtype=float)
    parameters = np.asarray(weights, dtype=float)
    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(f"{spec.name} expects shape (n, {N_FEATURES}).")
    if parameters.shape != (spec.n_weights,):
        raise ValueError(
            f"{spec.name} expects {spec.n_weights} weights; got {parameters.shape}."
        )
    if not np.isfinite(features).all() or not np.isfinite(parameters).all():
        raise ValueError(f"{spec.name} inputs and weights must be finite.")
    return features, parameters


def exact_probabilities(
    spec: CircuitSpec,
    x: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> np.ndarray:
    validate(spec)
    features, parameters = _validate_inputs(spec, x, weights)
    parts = []
    for start in range(0, len(features), chunk_rows):
        chunk = features[start : start + chunk_rows]
        state, _ = _forward(spec, chunk, parameters)
        parts.append((1.0 - _expectation(spec, state)) / 2.0)
    return np.clip(np.concatenate(parts), 0.0, 1.0)


def _chunk_value_and_gradient(
    spec, features, labels, sample_weight, weights, objective, epsilon
):
    state, plan = _forward(spec, features, weights)
    rows = state.shape[-1]
    probability = np.clip(
        (1.0 - _expectation(spec, state)) / 2.0, epsilon, 1.0 - epsilon
    )
    loss, dloss_dprobability = objective(probability, labels, sample_weight)
    dloss_dexpectation = -0.5 * dloss_dprobability
    gradient = np.zeros(spec.n_weights, dtype=float)

    adjoint = (
        state.reshape(1 << spec.n_qubits, rows) * _readout_sign(spec)[:, None]
    ).reshape(state.shape)
    for kind, first, second, angle, factors in reversed(plan):
        if kind == "cx":
            _apply_cx(state, first, second)
            _apply_cx(adjoint, first, second)
            continue
        if kind == "ry":
            _apply_ry(state, angle, first, inverse=True)
            angle_derivative = _ry_derivative(state, adjoint, angle, first, rows)
            _apply_ry(adjoint, angle, first, inverse=True)
        else:
            _apply_rz(state, angle, first, inverse=True)
            angle_derivative = _rz_derivative(state, adjoint, angle, first, rows)
            _apply_rz(adjoint, angle, first, inverse=True)
        weighted = dloss_dexpectation * angle_derivative
        for parameter_index, factor in factors:
            gradient[parameter_index] += float(
                np.sum(weighted if factor is None else weighted * factor)
            )
    return float(loss), gradient


def value_and_gradient(
    spec: CircuitSpec,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    objective=None,
    *,
    epsilon: float = 1e-9,
    chunk_rows: int = CHUNK_ROWS,
) -> tuple[float, np.ndarray]:
    """Objective of the exact readout probability and its adjoint gradient."""
    validate(spec)
    features, parameters = _validate_inputs(spec, x, weights)
    labels = np.asarray(y, dtype=float)
    if len(features) != len(labels):
        raise ValueError(f"{spec.name} requires one label per feature row.")
    sample_weight = balanced_sample_weights(labels.astype(int))
    if objective is None:
        objective = lambda p, l, w: balanced_bce(p, l, w, epsilon=epsilon)

    loss = 0.0
    gradient = np.zeros(spec.n_weights, dtype=float)
    for start in range(0, len(features), chunk_rows):
        stop = start + chunk_rows
        chunk_loss, chunk_gradient = _chunk_value_and_gradient(
            spec,
            features[start:stop],
            labels[start:stop],
            sample_weight[start:stop],
            parameters,
            objective,
            epsilon,
        )
        loss += chunk_loss
        gradient += chunk_gradient
    return float(loss), gradient
