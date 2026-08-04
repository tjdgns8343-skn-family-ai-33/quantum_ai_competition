"""Shared dual-axis affine extension of the compliant C1 circuit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit.circuit import Parameter

from .circuit import ALLOWED_GATES

E1S_ARCHITECTURE = "E1S_SHARED_DUAL_AXIS_AFFINE"
E1S_N_QUBITS = 4
E1S_N_FEATURES = 8
E1S_N_WEIGHTS = 49
E1S_READOUT_QUBIT = 0
E1S_REUPLOAD_BLOCKS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 2, 3),
    (4, 5, 6, 7),
)
E1S_CAUSAL_FUNNEL = ((3, 2), (2, 1), (1, 0))


def e1s_input_parameters() -> list[Parameter]:
    return [Parameter(f"x_{index}") for index in range(E1S_N_FEATURES)]


def e1s_weight_parameters() -> list[Parameter]:
    return [Parameter(f"theta_{index}") for index in range(E1S_N_WEIGHTS)]


def build_e1s_unitary() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build two uploads with shared per-feature affine RY and RZ maps."""
    features = e1s_input_parameters()
    weights = e1s_weight_parameters()
    circuit = QuantumCircuit(E1S_N_QUBITS, name="compliant_e1s_shared_dual_affine")

    for block_index, block_features in enumerate(E1S_REUPLOAD_BLOCKS):
        for qubit, feature_index in enumerate(block_features):
            offset = 4 * feature_index
            circuit.ry(
                weights[offset] * features[feature_index] + weights[offset + 1],
                qubit,
            )
        for qubit, feature_index in enumerate(block_features):
            offset = 4 * feature_index
            circuit.rz(
                weights[offset + 2] * features[feature_index] + weights[offset + 3],
                qubit,
            )
        mixer_offset = 32 + 4 * block_index
        for qubit in range(E1S_N_QUBITS):
            circuit.ry(weights[mixer_offset + qubit], qubit)
        for control, target in E1S_CAUSAL_FUNNEL:
            circuit.cx(control, target)

    circuit.ry(weights[48], E1S_READOUT_QUBIT)
    return circuit, features, weights


def build_e1s_submission_circuit() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_e1s_unitary()
    circuit.add_register(ClassicalRegister(1, "c"))
    circuit.measure(E1S_READOUT_QUBIT, circuit.clbits[0])
    return circuit, features, weights


def e1s_constraint_report(circuit: QuantumCircuit) -> dict:
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


def export_e1s_submission_qasm(destination: Path) -> dict:
    circuit, _, _ = build_e1s_submission_circuit()
    report = e1s_constraint_report(circuit)
    if not report["passes"]:
        raise ValueError(f"E1-S circuit constraint failure: {report}")
    destination.write_text(
        "// E1-S: shared single-feature affine RY/RZ maps; no preprocessing.\n"
        + qasm3.dumps(circuit),
        encoding="utf-8",
    )
    return report
