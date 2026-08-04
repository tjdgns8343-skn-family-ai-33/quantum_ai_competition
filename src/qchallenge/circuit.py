"""Single-feature direct encoding and a circuit-native B1 variational ansatz."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit.circuit import Parameter

N_QUBITS = 4
N_FEATURES = 8
N_WEIGHTS = 16
READOUT_QUBIT = 0
ALLOWED_GATES = {"x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "cx", "cz", "measure"}
ENTANGLERS = ((1, 0), (3, 2), (2, 1))
FEATURE_LAYOUTS = {
    "sequential": ((0, 1, 2, 3), (4, 5, 6, 7)),
    "proposed_1347_2568": ((0, 2, 3, 6), (1, 4, 5, 7)),
    "odd_even": ((0, 2, 4, 6), (1, 3, 5, 7)),
    "block_swap": ((4, 5, 6, 7), (0, 1, 2, 3)),
}


def normalize_selected_features(
    selected_features: tuple[int, ...] | None,
) -> tuple[int, ...]:
    selected = tuple(range(N_FEATURES)) if selected_features is None else tuple(selected_features)
    if not selected:
        raise ValueError("At least one raw feature must be selected.")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_features must not contain duplicates.")
    if any(index < 0 or index >= N_FEATURES for index in selected):
        raise ValueError("selected_features must contain zero-based indices from 0 to 7.")
    return selected


def input_parameters() -> list[Parameter]:
    return [Parameter(f"x_{index}") for index in range(N_FEATURES)]


def weight_parameters() -> list[Parameter]:
    return [Parameter(f"theta_{index}") for index in range(N_WEIGHTS)]


def _variational_block(
    circuit: QuantumCircuit, weights: list[Parameter], offset: int
) -> None:
    for control, target in ENTANGLERS:
        circuit.cx(control, target)
    for qubit in range(N_QUBITS):
        circuit.ry(weights[offset + qubit], qubit)
    for qubit in range(N_QUBITS):
        circuit.rz(weights[offset + 4 + qubit], qubit)


def build_unitary(
    feature_layout: str = "sequential",
    selected_features: tuple[int, ...] | None = None,
    pack_selected_features: bool = False,
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build B1; each data gate contains exactly one untouched raw feature."""
    if feature_layout not in FEATURE_LAYOUTS:
        raise ValueError(
            f"Unknown feature_layout {feature_layout}; choose one of {tuple(FEATURE_LAYOUTS)}"
        )
    selected = normalize_selected_features(selected_features)
    selected_set = set(selected)
    if pack_selected_features:
        first_features = selected[:4]
        second_features = selected[4:]
    else:
        first_features, second_features = FEATURE_LAYOUTS[feature_layout]
    features = input_parameters()
    weights = weight_parameters()
    circuit = QuantumCircuit(N_QUBITS, name=f"compliant_b1_{feature_layout}")
    for qubit, feature_index in enumerate(first_features):
        if pack_selected_features or feature_index in selected_set:
            circuit.ry(features[feature_index], qubit)
    _variational_block(circuit, weights, 0)
    for qubit, feature_index in enumerate(second_features):
        if pack_selected_features or feature_index in selected_set:
            circuit.rz(features[feature_index], qubit)
    _variational_block(circuit, weights, 8)
    return circuit, [features[index] for index in selected], weights


def build_training_circuit(
    feature_layout: str = "sequential",
    selected_features: tuple[int, ...] | None = None,
    pack_selected_features: bool = False,
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_unitary(
        feature_layout, selected_features, pack_selected_features
    )
    circuit.measure_all()
    return circuit, features, weights


def build_submission_circuit(
    feature_layout: str = "sequential",
    selected_features: tuple[int, ...] | None = None,
    pack_selected_features: bool = False,
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_unitary(
        feature_layout, selected_features, pack_selected_features
    )
    circuit.add_register(ClassicalRegister(1, "c"))
    circuit.measure(READOUT_QUBIT, circuit.clbits[0])
    return circuit, features, weights


def constraint_report(circuit: QuantumCircuit) -> dict:
    counts = Counter(circuit.count_ops())
    unsupported = sorted(set(counts) - ALLOWED_GATES)
    two_qubit = int(counts["cx"] + counts["cz"])
    report = {
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "operation_counts": dict(counts),
        "two_qubit_gate_count": two_qubit,
        "measurement_count": int(counts["measure"]),
        "unsupported_gates": unsupported,
    }
    report["passes"] = bool(
        2 <= circuit.num_qubits <= 8
        and circuit.depth() <= 50
        and 1 <= two_qubit <= 80
        and counts["measure"] == 1
        and not unsupported
    )
    return report


def export_submission_qasm(
    destination: Path,
    feature_layout: str = "sequential",
    selected_features: tuple[int, ...] | None = None,
    pack_selected_features: bool = False,
) -> dict:
    circuit, _, _ = build_submission_circuit(
        feature_layout, selected_features, pack_selected_features
    )
    report = constraint_report(circuit)
    if not report["passes"]:
        raise ValueError(f"Circuit constraint failure: {report}")
    qasm = qasm3.dumps(circuit)
    destination.write_text(
        "// Raw x1..x8 are mapped directly to x_0..x_7; no preprocessing.\n" + qasm,
        encoding="utf-8",
    )
    return report
