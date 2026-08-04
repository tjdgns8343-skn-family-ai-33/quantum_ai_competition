"""Competition-compliant causal data-reuploading variational circuit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit.circuit import Parameter

from .circuit import ALLOWED_GATES

C2_ARCHITECTURE = "C2_CAUSAL_REUPLOAD"
C2_N_QUBITS = 4
C2_N_FEATURES = 8
C2_N_WEIGHTS = 97
C2_READOUT_QUBIT = 0
C2_REUPLOAD_BLOCKS = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 2, 3),
    (4, 5, 6, 7),
)

# The farthest information is moved first.  Unlike the B1 order, this makes
# every block part of the q0 backward causal cone.
C2_CAUSAL_FUNNEL = ((3, 2), (2, 1), (1, 0))


def c2_input_parameters() -> list[Parameter]:
    return [Parameter(f"x_{index}") for index in range(C2_N_FEATURES)]


def c2_weight_parameters() -> list[Parameter]:
    return [Parameter(f"theta_{index}") for index in range(C2_N_WEIGHTS)]


def build_c2_unitary() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build C2 with three complete raw-feature re-uploading rounds.

    Each data gate has the explicitly permitted single-feature affine angle
    ``theta_scale * x_i + theta_bias``.  Feature interactions are produced
    only by the quantum circuit after encoding.
    """
    features = c2_input_parameters()
    weights = c2_weight_parameters()
    circuit = QuantumCircuit(C2_N_QUBITS, name="compliant_c2_causal_reupload")

    for block_index, block_features in enumerate(C2_REUPLOAD_BLOCKS):
        offset = 16 * block_index
        for qubit, feature_index in enumerate(block_features):
            scale = weights[offset + qubit]
            bias = weights[offset + 4 + qubit]
            circuit.ry(scale * features[feature_index] + bias, qubit)
        for qubit in range(C2_N_QUBITS):
            circuit.rz(weights[offset + 8 + qubit], qubit)
        for qubit in range(C2_N_QUBITS):
            circuit.ry(weights[offset + 12 + qubit], qubit)
        for control, target in C2_CAUSAL_FUNNEL:
            circuit.cx(control, target)

    # A final non-diagonal readout rotation converts the accumulated phase
    # information on q0 into its Z-basis measurement probability.
    circuit.ry(weights[96], C2_READOUT_QUBIT)
    return circuit, features, weights


def build_c2_training_circuit() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_c2_unitary()
    circuit.measure_all()
    return circuit, features, weights


def build_c2_submission_circuit() -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_c2_unitary()
    circuit.add_register(ClassicalRegister(1, "c"))
    circuit.measure(C2_READOUT_QUBIT, circuit.clbits[0])
    return circuit, features, weights


def c2_constraint_report(circuit: QuantumCircuit) -> dict:
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


def export_c2_submission_qasm(destination: Path) -> dict:
    circuit, _, _ = build_c2_submission_circuit()
    report = c2_constraint_report(circuit)
    if not report["passes"]:
        raise ValueError(f"C2 circuit constraint failure: {report}")
    destination.write_text(
        "// C2: raw x1..x8 enter only single-feature affine RY gates; "
        "no preprocessing or augmentation.\n"
        + qasm3.dumps(circuit),
        encoding="utf-8",
    )
    return report


