import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qchallenge.f1_circuit import (
    F1_DEFAULT_BLOCKS,
    F1_N_FEATURES,
    build_f1_submission_circuit,
    build_f1_unitary,
    f1_constraint_report,
    f1_reupload_blocks,
    f1_weight_count,
)
from qchallenge.f1_simulator import (
    f1_balanced_bce_value_and_gradient,
    f1_exact_probabilities,
)
from qchallenge.f1_training import f1_feature_causal_report, f1_random_initial_point


def _random_problem(seed: int = 14, n_blocks: int = F1_DEFAULT_BLOCKS):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, F1_N_FEATURES))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    weights = f1_random_initial_point(
        seed, local_scale=0.05, affine_scale_jitter=0.1, n_blocks=n_blocks
    )
    return x, y, weights


def test_f1_uses_the_full_competition_budget_without_exceeding_it():
    circuit, features, weights = build_f1_submission_circuit()
    report = f1_constraint_report(circuit)
    assert report["passes"]
    assert report["qubits"] == 8
    assert report["depth"] <= 50
    assert report["two_qubit_gate_count"] == 56
    assert report["two_qubit_gate_count"] <= 80
    assert report["measurement_count"] == 1
    assert len(features) == F1_N_FEATURES
    assert len(weights) == f1_weight_count()


def test_f1_data_gates_are_single_feature_affine():
    seen: list[str] = []
    unitary, _, _ = build_f1_unitary()
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
    expected = [f"x_{index}" for index in range(8)] * F1_DEFAULT_BLOCKS
    assert sorted(seen) == sorted(expected)


def test_f1_every_block_uses_every_feature_exactly_once():
    for block in f1_reupload_blocks():
        assert sorted(block) == list(range(F1_N_FEATURES))


@pytest.mark.parametrize("n_blocks", [1, 4, F1_DEFAULT_BLOCKS])
def test_f1_vectorized_statevector_matches_qiskit(n_blocks):
    x, _, weights = _random_problem(n_blocks=n_blocks)
    expected = f1_exact_probabilities(x, weights, n_blocks)
    circuit, features, parameters = build_f1_unitary(n_blocks)
    actual = []
    for row in x:
        bindings = {
            **dict(zip(features, row, strict=True)),
            **dict(zip(parameters, weights, strict=True)),
        }
        state = Statevector.from_instruction(circuit.assign_parameters(bindings))
        actual.append(state.probabilities([0])[1])
    assert np.max(np.abs(expected - np.asarray(actual))) < 1e-10


def test_f1_adjoint_gradient_matches_finite_difference():
    x, y, weights = _random_problem()
    count = f1_weight_count()
    _, gradient = f1_balanced_bce_value_and_gradient(x, y, weights)
    epsilon = 1e-6
    for index in (0, 7, 8, 16, 24, 31, 64, 129, 200, count - 1):
        direction = np.zeros(count)
        direction[index] = epsilon
        plus, _ = f1_balanced_bce_value_and_gradient(x, y, weights + direction)
        minus, _ = f1_balanced_bce_value_and_gradient(x, y, weights - direction)
        finite_difference = (plus - minus) / (2.0 * epsilon)
        assert abs(gradient[index] - finite_difference) < 1e-6


def test_f1_row_chunking_does_not_change_loss_or_gradient():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, F1_N_FEATURES))
    y = rng.integers(0, 2, 200)
    weights = f1_random_initial_point(5, local_scale=0.3, affine_scale_jitter=0.1)
    whole_loss, whole_gradient = f1_balanced_bce_value_and_gradient(
        x, y, weights, chunk_rows=len(x)
    )
    chunked_loss, chunked_gradient = f1_balanced_bce_value_and_gradient(
        x, y, weights, chunk_rows=32
    )
    assert abs(whole_loss - chunked_loss) < 1e-12
    assert np.max(np.abs(whole_gradient - chunked_gradient)) < 1e-12
    whole = f1_exact_probabilities(x, weights, chunk_rows=len(x))
    chunked = f1_exact_probabilities(x, weights, chunk_rows=32)
    assert np.max(np.abs(whole - chunked)) < 1e-12


def test_f1_all_features_reach_q0_within_a_single_block():
    """One block must already put every feature in the q0 causal cone."""
    x, _, weights = _random_problem(n_blocks=1)
    report = f1_feature_causal_report(x, weights, 1)
    assert report["all_features_nonzero_at_1e_10"]


def test_f1_initialization_is_reproducible_and_label_independent():
    first = f1_random_initial_point(2026, local_scale=0.05, affine_scale_jitter=0.1)
    second = f1_random_initial_point(2026, local_scale=0.05, affine_scale_jitter=0.1)
    third = f1_random_initial_point(2027, local_scale=0.05, affine_scale_jitter=0.1)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)
