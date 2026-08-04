"""Competition-compliant two-qubit, two-upload deep variational circuit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit.circuit import Parameter

from .circuit import ALLOWED_GATES

D1_ARCHITECTURE = "D1_TWO_QUBIT_TWO_UPLOAD_DEEP"
D1_N_QUBITS = 2
D1_N_FEATURES = 8
D1_N_WEIGHTS = 80
D1_READOUT_QUBIT = 0
D1_REUPLOAD_BLOCKS = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
)

# Alternation lets information flow in both directions.  The last block uses
# q1 -> q0 so the final q1 feature remains inside q0's backward causal cone.
D1_ENTANGLERS = tuple(
    (0, 1) if block_index % 2 == 0 else (1, 0)
    for block_index in range(len(D1_REUPLOAD_BLOCKS))
)


def d1_input_parameters() -> list[Parameter]:
    return [Parameter(f"x_{index}") for index in range(D1_N_FEATURES)]


def d1_weight_parameters() -> list[Parameter]:
    return [Parameter(f"theta_{index}") for index in range(D1_N_WEIGHTS)]


def build_d1_unitary() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build the two-qubit circuit with exactly two complete feature uploads."""
    features = d1_input_parameters()
    weights = d1_weight_parameters()
    circuit = QuantumCircuit(D1_N_QUBITS, name="compliant_d1_two_upload_deep")

    for block_index, (block_features, entangler) in enumerate(
        zip(D1_REUPLOAD_BLOCKS, D1_ENTANGLERS, strict=True)
    ):
        offset = 10 * block_index
        for qubit, feature_index in enumerate(block_features):
            scale = weights[offset + qubit]
            bias = weights[offset + 2 + qubit]
            circuit.ry(scale * features[feature_index] + bias, qubit)
        for qubit in range(D1_N_QUBITS):
            circuit.rz(weights[offset + 4 + qubit], qubit)
        for qubit in range(D1_N_QUBITS):
            circuit.ry(weights[offset + 6 + qubit], qubit)
        circuit.cx(*entangler)
        for qubit in range(D1_N_QUBITS):
            circuit.ry(weights[offset + 8 + qubit], qubit)

    return circuit, features, weights


def build_d1_submission_circuit() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_d1_unitary()
    circuit.add_register(ClassicalRegister(1, "c"))
    circuit.measure(D1_READOUT_QUBIT, circuit.clbits[0])
    return circuit, features, weights


def d1_constraint_report(circuit: QuantumCircuit) -> dict:
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


def export_d1_submission_qasm(destination: Path) -> dict:
    circuit, _, _ = build_d1_submission_circuit()
    report = d1_constraint_report(circuit)
    if not report["passes"]:
        raise ValueError(f"D1 circuit constraint failure: {report}")
    destination.write_text(
        "// D1: two qubits, exactly two raw-feature uploads, single-feature affine RY encoding.\n"
        + qasm3.dumps(circuit),
        encoding="utf-8",
    )
    return report
