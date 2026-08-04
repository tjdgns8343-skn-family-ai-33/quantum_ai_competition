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


def build_unitary() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build B1; each data gate contains exactly one untouched raw feature."""
    features = input_parameters()
    weights = weight_parameters()
    circuit = QuantumCircuit(N_QUBITS, name="compliant_b1_direct_raw")
    for qubit in range(N_QUBITS):
        circuit.ry(features[qubit], qubit)
    _variational_block(circuit, weights, 0)
    for qubit in range(N_QUBITS):
        circuit.rz(features[4 + qubit], qubit)
    _variational_block(circuit, weights, 8)
    return circuit, features, weights


def build_training_circuit() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_unitary()
    circuit.measure_all()
    return circuit, features, weights


def build_submission_circuit() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_unitary()
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


def export_submission_qasm(destination: Path) -> dict:
    circuit, _, _ = build_submission_circuit()
    report = constraint_report(circuit)
    if not report["passes"]:
        raise ValueError(f"Circuit constraint failure: {report}")
    qasm = qasm3.dumps(circuit)
    destination.write_text(
        "// Raw x1..x8 are mapped directly to x_0..x_7; no preprocessing.\n" + qasm,
        encoding="utf-8",
    )
    return report

