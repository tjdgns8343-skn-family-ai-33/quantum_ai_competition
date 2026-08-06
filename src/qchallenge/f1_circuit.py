"""Competition-compliant eight-qubit tree-funnel data-reuploading circuit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit.circuit import Parameter

from .circuit import ALLOWED_GATES

F1_ARCHITECTURE = "F1_EIGHT_QUBIT_TREE_FUNNEL_REUPLOAD"
F1_N_QUBITS = 8
F1_N_FEATURES = 8
F1_READOUT_QUBIT = 0
F1_DEFAULT_BLOCKS = 8
F1_PARAMETERS_PER_BLOCK = 4 * F1_N_QUBITS

# A binary-tree funnel instead of C1's linear chain.  Every qubit enters the q0
# backward causal cone inside a single block using depth 3 and seven CX gates,
# so no feature has to wait for later blocks to influence the readout.
F1_TREE_FUNNEL = ((1, 0), (3, 2), (5, 4), (7, 6), (2, 0), (6, 4), (4, 0))


def f1_validate_blocks(n_blocks: int) -> int:
    if n_blocks < 1:
        raise ValueError("F1 requires at least one re-uploading block.")
    return int(n_blocks)


def f1_weight_count(n_blocks: int = F1_DEFAULT_BLOCKS) -> int:
    """Per block: 8 affine scales, 8 affine biases, 8 RZ and 8 RY mixers."""
    return F1_PARAMETERS_PER_BLOCK * f1_validate_blocks(n_blocks) + 1


def f1_reupload_blocks(n_blocks: int = F1_DEFAULT_BLOCKS) -> tuple[tuple[int, ...], ...]:
    """Block ``b`` encodes raw feature ``(q + b) % 8`` on qubit ``q``.

    The rotation makes each raw feature visit a different position of the tree
    funnel on every upload, so no feature is permanently stuck in the shallowest
    or deepest branch.  The choice is fixed in advance and uses no label
    information or data statistic.
    """
    return tuple(
        tuple((qubit + block) % F1_N_FEATURES for qubit in range(F1_N_QUBITS))
        for block in range(f1_validate_blocks(n_blocks))
    )


def f1_input_parameters() -> list[Parameter]:
    return [Parameter(f"x_{index}") for index in range(F1_N_FEATURES)]


def f1_weight_parameters(n_blocks: int = F1_DEFAULT_BLOCKS) -> list[Parameter]:
    return [Parameter(f"theta_{index}") for index in range(f1_weight_count(n_blocks))]


def build_f1_unitary(
    n_blocks: int = F1_DEFAULT_BLOCKS,
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build F1 with ``n_blocks`` complete raw-feature re-uploading rounds.

    Every data gate keeps the explicitly permitted single-feature affine angle
    ``theta_scale * x_i + theta_bias``.  All feature interaction is produced by
    the CX tree funnel inside the circuit, never by classical arithmetic.
    """
    blocks = f1_reupload_blocks(n_blocks)
    features = f1_input_parameters()
    weights = f1_weight_parameters(n_blocks)
    circuit = QuantumCircuit(F1_N_QUBITS, name="compliant_f1_tree_funnel_reupload")

    for block_index, block_features in enumerate(blocks):
        offset = F1_PARAMETERS_PER_BLOCK * block_index
        for qubit, feature_index in enumerate(block_features):
            scale = weights[offset + qubit]
            bias = weights[offset + F1_N_QUBITS + qubit]
            circuit.ry(scale * features[feature_index] + bias, qubit)
        for qubit in range(F1_N_QUBITS):
            circuit.rz(weights[offset + 2 * F1_N_QUBITS + qubit], qubit)
        for qubit in range(F1_N_QUBITS):
            circuit.ry(weights[offset + 3 * F1_N_QUBITS + qubit], qubit)
        for control, target in F1_TREE_FUNNEL:
            circuit.cx(control, target)

    # A final non-diagonal readout rotation turns the accumulated phase on q0
    # into its Z-basis measurement probability.
    circuit.ry(weights[F1_PARAMETERS_PER_BLOCK * len(blocks)], F1_READOUT_QUBIT)
    return circuit, features, weights


def build_f1_training_circuit(
    n_blocks: int = F1_DEFAULT_BLOCKS,
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_f1_unitary(n_blocks)
    circuit.measure_all()
    return circuit, features, weights


def build_f1_submission_circuit(
    n_blocks: int = F1_DEFAULT_BLOCKS,
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    circuit, features, weights = build_f1_unitary(n_blocks)
    circuit.add_register(ClassicalRegister(1, "c"))
    circuit.measure(F1_READOUT_QUBIT, circuit.clbits[0])
    return circuit, features, weights


def f1_constraint_report(circuit: QuantumCircuit) -> dict:
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


def export_f1_submission_qasm(
    destination: Path, n_blocks: int = F1_DEFAULT_BLOCKS
) -> dict:
    circuit, _, _ = build_f1_submission_circuit(n_blocks)
    report = f1_constraint_report(circuit)
    if not report["passes"]:
        raise ValueError(f"F1 circuit constraint failure: {report}")
    destination.write_text(
        "// F1: raw x1..x8 enter only single-feature affine RY gates; "
        "no preprocessing or augmentation.\n"
        + qasm3.dumps(circuit),
        encoding="utf-8",
    )
    return report
