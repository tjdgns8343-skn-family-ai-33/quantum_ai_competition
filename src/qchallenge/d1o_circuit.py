"""Compliant shallow two-qubit circuit with exactly one feature upload."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit.circuit import Parameter

from .circuit import ALLOWED_GATES

D1O_ARCHITECTURE = "D1O_TWO_QUBIT_ONE_UPLOAD_SHALLOW"
D1O_N_QUBITS = 2
D1O_N_FEATURES = 8
D1O_N_WEIGHTS = 28
D1O_READOUT_QUBIT = 0
D1O_REUPLOAD_BLOCKS = (
    (0, 1), (2, 3), (4, 5), (6, 7),
)
D1O_ENTANGLERS = tuple(
    (0, 1) if block_index % 2 == 0 else (1, 0)
    for block_index in range(len(D1O_REUPLOAD_BLOCKS))
)


def d1o_input_parameters() -> list[Parameter]:
    return [Parameter(f"x_{index}") for index in range(D1O_N_FEATURES)]


def d1o_weight_parameters() -> list[Parameter]:
    return [Parameter(f"theta_{index}") for index in range(D1O_N_WEIGHTS)]


def build_d1o_unitary() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    features = d1o_input_parameters()
    weights = d1o_weight_parameters()
    circuit = QuantumCircuit(D1O_N_QUBITS, name="compliant_d1o_one_upload_affine")

    for block_index, (block_features, entangler) in enumerate(
        zip(D1O_REUPLOAD_BLOCKS, D1O_ENTANGLERS, strict=True)
    ):
        offset = 6 * block_index
        for qubit, feature_index in enumerate(block_features):
            circuit.ry(
                weights[offset + qubit] * features[feature_index]
                + weights[offset + 2 + qubit],
                qubit,
            )
        circuit.cx(*entangler)
        for qubit in range(D1O_N_QUBITS):
            circuit.ry(weights[offset + 4 + qubit], qubit)

        if block_index == 3:
            mixer_offset = 24
            for qubit in range(D1O_N_QUBITS):
                circuit.rz(weights[mixer_offset + qubit], qubit)
            for qubit in range(D1O_N_QUBITS):
                circuit.ry(weights[mixer_offset + 2 + qubit], qubit)

    return circuit, features, weights


def build_d1o_submission_circuit() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_d1o_unitary()
    circuit.add_register(ClassicalRegister(1, "c"))
    circuit.measure(D1O_READOUT_QUBIT, circuit.clbits[0])
    return circuit, features, weights


def d1o_constraint_report(circuit: QuantumCircuit) -> dict:
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


def export_d1o_submission_qasm(destination: Path) -> dict:
    circuit, _, _ = build_d1o_submission_circuit()
    report = d1o_constraint_report(circuit)
    if not report["passes"]:
        raise ValueError(f"D1-O circuit constraint failure: {report}")
    destination.write_text(
        "// D1-O: two qubits, exactly one raw-feature upload, shallow mixers.\n"
        + qasm3.dumps(circuit),
        encoding="utf-8",
    )
    return report


