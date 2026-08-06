"""Candidate specs must build legal circuits that the simulator reproduces.

CZ matters here: no architecture in this repository used it until now, so the
spec builder, the QASM export and the statevector simulator all had to learn it
at once. A diagonal entangler is easy to get silently wrong -- it changes no
computational-basis amplitude magnitude on its own, so a broken implementation
can still look plausible until it is checked against Qiskit.
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qchallenge.c1_simulator import c1_exact_probabilities
from qchallenge.candidates import CANDIDATES, build_candidate
from qchallenge.gatespec import build_circuit, constraint_report
from qchallenge.generic_simulator import exact_probabilities, value_and_gradient


@pytest.mark.parametrize("name", sorted(CANDIDATES))
def test_candidate_satisfies_competition_constraints(name):
    circuit, _, _ = build_circuit(build_candidate(name), measured=True)
    report = constraint_report(circuit)
    assert report["passes"], report
    assert 2 <= report["qubits"] <= 8
    assert report["depth"] <= 50
    assert 1 <= report["two_qubit_gate_count"] <= 80
    assert report["measurement_count"] == 1
    assert not report["unsupported_gates"]


@pytest.mark.parametrize("name", ["c1ref", "c1ref_cz", "c1ref_cxcz", "l1b2", "a1b2"])
def test_simulator_matches_qiskit(name):
    spec = build_candidate(name)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(5, 8))
    weights = rng.uniform(-1.0, 1.0, spec.n_weights)
    circuit, features, parameters = build_circuit(spec)
    expected = []
    for row in x:
        bindings = {
            **{p: row[int(str(p).split("_")[1])] for p in features},
            **dict(zip(parameters, weights, strict=True)),
        }
        state = Statevector.from_instruction(circuit.assign_parameters(bindings))
        expected.append(state.probabilities([spec.readout_qubit])[1])
    assert np.max(np.abs(exact_probabilities(spec, x, weights) - expected)) < 1e-10


@pytest.mark.parametrize("name", ["c1ref_cz", "c1ref_cxcz", "l1b3"])
def test_adjoint_gradient_matches_finite_difference(name):
    spec = build_candidate(name)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(40, 8))
    y = rng.integers(0, 2, 40)
    weights = rng.uniform(-1.0, 1.0, spec.n_weights)
    _, gradient = value_and_gradient(spec, x, y, weights)
    epsilon = 1e-6
    for index in (0, 3, spec.n_weights - 1):
        step = np.zeros(spec.n_weights)
        step[index] = epsilon
        plus = value_and_gradient(spec, x, y, weights + step)[0]
        minus = value_and_gradient(spec, x, y, weights - step)[0]
        assert abs(gradient[index] - (plus - minus) / (2 * epsilon)) < 1e-6


def test_c1ref_reproduces_the_submitted_circuit():
    """c1ref is the incumbent, so screens compare on one instrument.

    The spec numbers its thetas per qubit while the original groups all scales
    then all biases; the permutation below is the whole difference.
    """
    spec = build_candidate("c1ref")
    rng = np.random.default_rng(9)
    x = rng.normal(size=(6, 8))
    original = rng.uniform(-1.0, 1.0, 65)
    reordered = original.copy()
    for block in range(4):
        for qubit in range(4):
            reordered[16 * block + 2 * qubit] = original[16 * block + qubit]
            reordered[16 * block + 2 * qubit + 1] = original[16 * block + 4 + qubit]
    assert np.max(
        np.abs(exact_probabilities(spec, x, reordered) - c1_exact_probabilities(x, original))
    ) < 1e-12


def test_cz_variant_differs_from_cx_but_keeps_the_same_budget():
    cx, cz = build_candidate("c1ref"), build_candidate("c1ref_cz")
    assert cx.n_weights == cz.n_weights
    assert cx.two_qubit_count() == cz.two_qubit_count()
    rng = np.random.default_rng(13)
    x = rng.normal(size=(8, 8))
    weights = rng.uniform(-1.0, 1.0, cx.n_weights)
    difference = np.abs(
        exact_probabilities(cx, x, weights) - exact_probabilities(cz, x, weights)
    )
    assert difference.max() > 1e-3, "the two entanglers must not be equivalent"
