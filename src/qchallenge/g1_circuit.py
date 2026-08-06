"""G1: a circuit shaped by the group structure found in the raw features.

Section 13.4 of the compliance report measured, without touching labels, that
x3 tracks the second harmonic of x2, that x6-x8 track x5, and that x1 is nearly
constant (std 0.372 in a narrow arc, and the earlier quantum-only beam search
never selected it).  So the eight columns carry about three independent
directions presented through several nonlinear views.

G1 uses that directly.  x1 is dropped, which announcement #5 explicitly allows.
Each qubit receives every view of one group in sequence, so the circuit itself
forms a learned combination of those views -- something a single affine angle
cannot do, and which we are not permitted to precompute classically.

The views alternate RY and RZ on purpose.  Consecutive rotations about the same
axis would compose into one rotation whose angle mixes two features, which could
be read as violating the single-feature angle rule.  Alternating axes keeps
every gate unambiguously single-feature and is strictly more expressive.
"""

from __future__ import annotations

from pathlib import Path

from qiskit import qasm3

from .gatespec import (
    CircuitSpec,
    Gate,
    build_circuit,
    constraint_report,
    validate,
)

G1_ARCHITECTURE = "G1_LATENT_GROUP_REUPLOAD"
G1_N_QUBITS = 4
G1_READOUT_QUBIT = 0
G1_DEFAULT_BLOCKS = 4

# Zero-based raw feature indices.  x1 (index 0) is deliberately unused.
G1_GROUP_A = (1, 2, 3)          # x2 and its harmonics x3, x4
G1_GROUP_B = (4, 5, 6, 7)       # x5 and its harmonics x6, x7, x8

# Two independently parameterized copies of each group, so the readout sees two
# different learned projections of the same latent direction.
G1_QUBIT_GROUPS = (G1_GROUP_A, G1_GROUP_B, G1_GROUP_A, G1_GROUP_B)

# Tree entangler: depth 3, three CX, and q0 ends up depending on all four qubits.
G1_ENTANGLER = ((1, 0), (3, 2), (2, 0))


G1_X1 = 0  # zero-based index of raw column x1


def g1_qubit_groups(include_x1: bool = False) -> tuple[tuple[int, ...], ...]:
    """Feature sequence each qubit receives.

    x1 is dropped by default: it has std 0.372 against 0.61-1.61 for the other
    columns, only 147 distinct values in 6,000 rows, and the earlier
    quantum-only beam search never selected it.  ``include_x1`` appends it to
    every qubit instead, since it is a third independent direction rather than a
    view of either group, and both groups should be able to combine with it.
    Everything else stays identical, so the two specs differ in one factor.
    """
    if not include_x1:
        return G1_QUBIT_GROUPS
    return tuple((*group, G1_X1) for group in G1_QUBIT_GROUPS)


def g1_gates_per_block(include_x1: bool = False) -> int:
    return sum(len(group) for group in g1_qubit_groups(include_x1))


def g1_weights_per_block(include_x1: bool = False) -> int:
    """Two weights per data gate, plus an RZ and an RY mixer per qubit."""
    return 2 * g1_gates_per_block(include_x1) + 2 * G1_N_QUBITS


def g1_weight_count(
    n_blocks: int = G1_DEFAULT_BLOCKS, include_x1: bool = False
) -> int:
    if n_blocks < 1:
        raise ValueError("G1 requires at least one block.")
    return g1_weights_per_block(include_x1) * n_blocks + 1


def build_g1_spec(
    n_blocks: int = G1_DEFAULT_BLOCKS, include_x1: bool = False
) -> CircuitSpec:
    groups = g1_qubit_groups(include_x1)
    gates: list[Gate] = []
    cursor = 0
    for _ in range(n_blocks):
        for qubit, group in enumerate(groups):
            for position, feature in enumerate(group):
                gates.append(
                    Gate(
                        kind="ry" if position % 2 == 0 else "rz",
                        qubit=qubit,
                        feature=feature,
                        scale_index=cursor,
                        bias_index=cursor + 1,
                    )
                )
                cursor += 2
        for qubit in range(G1_N_QUBITS):
            gates.append(Gate(kind="rz", qubit=qubit, param_index=cursor))
            cursor += 1
        for qubit in range(G1_N_QUBITS):
            gates.append(Gate(kind="ry", qubit=qubit, param_index=cursor))
            cursor += 1
        for control, target in G1_ENTANGLER:
            gates.append(Gate(kind="cx", control=control, target=target))
    # Final readout rotation turns accumulated phase on q0 into a Z probability.
    gates.append(Gate(kind="ry", qubit=G1_READOUT_QUBIT, param_index=cursor))
    cursor += 1

    spec = CircuitSpec(
        name="compliant_g1_latent_group_reupload"
        + ("_with_x1" if include_x1 else ""),
        n_qubits=G1_N_QUBITS,
        n_weights=cursor,
        readout_qubit=G1_READOUT_QUBIT,
        gates=tuple(gates),
    )
    validate(spec)
    if spec.n_weights != g1_weight_count(n_blocks, include_x1):
        raise RuntimeError("G1 weight bookkeeping disagrees with the spec.")
    return spec


def g1_constraint_report(
    n_blocks: int = G1_DEFAULT_BLOCKS, include_x1: bool = False
) -> dict:
    circuit, _, _ = build_circuit(build_g1_spec(n_blocks, include_x1), measured=True)
    return constraint_report(circuit)


def export_g1_submission_qasm(
    destination: Path, n_blocks: int = G1_DEFAULT_BLOCKS, include_x1: bool = False
) -> dict:
    spec = build_g1_spec(n_blocks, include_x1)
    circuit, _, _ = build_circuit(spec, measured=True)
    report = constraint_report(circuit)
    if not report["passes"]:
        raise ValueError(f"G1 circuit constraint failure: {report}")
    columns = ", ".join(f"x{index + 1}" for index in spec.used_features())
    destination.write_text(
        f"// G1: raw {columns} enter only single-feature affine RY/RZ gates; "
        "no preprocessing or augmentation.\n" + qasm3.dumps(circuit),
        encoding="utf-8",
    )
    return report
