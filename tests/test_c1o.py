import numpy as np
from qiskit.quantum_info import Statevector

from qchallenge.c1o_circuit import (
    C1O_N_FEATURES,
    C1O_N_WEIGHTS,
    build_c1o_submission_circuit,
    build_c1o_unitary,
    c1o_constraint_report,
)
from qchallenge.c1o_simulator import (
    c1o_balanced_bce_value_and_gradient,
    c1o_exact_probabilities,
)
from qchallenge.c1o_training import c1o_feature_causal_report, c1o_random_initial_point


def _random_problem(seed: int = 14):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, C1O_N_FEATURES))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    weights = c1o_random_initial_point(
        seed, local_scale=0.05, affine_scale_jitter=0.1
    )
    return x, y, weights


def test_c1o_constraints_and_direct_affine_parameters():
    circuit, features, weights = build_c1o_submission_circuit()
    report = c1o_constraint_report(circuit)
    assert report["passes"]
    assert report["qubits"] == 4
    assert report["depth"] <= 50
    assert report["two_qubit_gate_count"] == 6
    assert len(features) == C1O_N_FEATURES
    assert len(weights) == C1O_N_WEIGHTS

    seen: list[str] = []
    unitary, _, _ = build_c1o_unitary()
    for instruction in unitary.data:
        for expression in instruction.operation.params:
            raw = [
                parameter
                for parameter in getattr(expression, "parameters", set())
                if str(parameter).startswith("x_")
            ]
            assert len(raw) <= 1
            if raw:
                seen.append(str(raw[0]))
                derivative = expression.gradient(raw[0])
                second_derivative = (
                    derivative.gradient(raw[0])
                    if hasattr(derivative, "gradient")
                    else 0.0
                )
                assert str(second_derivative) in {"0", "0.0"}
    assert sorted(seen) == sorted([f"x_{index}" for index in range(8)])


def test_c1o_vectorized_statevector_matches_qiskit():
    x, _, weights = _random_problem()
    expected = c1o_exact_probabilities(x, weights)
    circuit, features, parameters = build_c1o_unitary()
    actual = []
    for row in x:
        bindings = {
            **dict(zip(features, row, strict=True)),
            **dict(zip(parameters, weights, strict=True)),
        }
        state = Statevector.from_instruction(circuit.assign_parameters(bindings))
        actual.append(state.probabilities([0])[1])
    assert np.max(np.abs(expected - np.asarray(actual))) < 1e-10


def test_c1o_adjoint_gradient_matches_finite_difference():
    x, y, weights = _random_problem()
    _, gradient = c1o_balanced_bce_value_and_gradient(x, y, weights)
    epsilon = 1e-6
    for index in (0, 4, 8, 12, 16, 31, 32):
        direction = np.zeros(C1O_N_WEIGHTS)
        direction[index] = epsilon
        plus, _ = c1o_balanced_bce_value_and_gradient(x, y, weights + direction)
        minus, _ = c1o_balanced_bce_value_and_gradient(x, y, weights - direction)
        finite_difference = (plus - minus) / (2.0 * epsilon)
        assert abs(gradient[index] - finite_difference) < 1e-6


def test_c1o_all_features_are_in_q0_causal_cone_at_initialization():
    x, _, weights = _random_problem()
    report = c1o_feature_causal_report(x, weights)
    assert report["all_features_nonzero_at_1e_10"]


def test_c1o_initialization_is_reproducible_and_label_independent():
    first = c1o_random_initial_point(
        2026, local_scale=0.05, affine_scale_jitter=0.1
    )
    second = c1o_random_initial_point(
        2026, local_scale=0.05, affine_scale_jitter=0.1
    )
    third = c1o_random_initial_point(
        2027, local_scale=0.05, affine_scale_jitter=0.1
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


