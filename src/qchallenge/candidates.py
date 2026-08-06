"""Candidate architectures aimed at the bottleneck the trained C1 revealed.

Feature-sensitivity of the submitted AUC-trained C1, weighted by each column's
spread, puts 70.7% of the output variation on x1, x5 and x2 -- exactly the three
independent directions found label-free (x3 and x4 track x2, x6 to x8 track x5).
The circuit rediscovered the data's effective dimension on its own.

C1's structural limit follows from its layout: blocks 1 and 3 encode x1-x4 on
q0-q3 and blocks 2 and 4 encode x5-x8, so every feature touches exactly one
qubit and q0's rotation depends only on x1 and x5.  All cross-feature structure
has to come from twelve CX gates, and each direction gets only two uploads.
Meanwhile the rules allow depth 50 and 80 two-qubit gates against C1's 23 and 12.

Two opposite bets on that:

``all_to_all`` gives every qubit every feature, so each qubit becomes a learned
projection of all eight columns and entanglement combines four different
projections.  Nothing is discarded.

``latent_focused`` keeps only x1, x2 and x5 and uploads each of them twenty
times.  With that many uploads at freely trained scales the circuit can generate
its own harmonics, which is what x3, x4 and x6-x8 already are -- they were worth
having when a direction got only two uploads.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .gatespec import CircuitSpec, Gate

N_QUBITS = 4
READOUT_QUBIT = 0
ALTERNATING_AXES = ("ry", "rz")

# Depth 3, three CX, and q0 ends up depending on all four qubits.
CX_TREE = (("cx", 1, 0), ("cx", 3, 2), ("cx", 2, 0))
# Same connectivity with the diagonal entangler.  CX permutes amplitudes while
# CZ multiplies phases, so the two generate different interaction terms; no
# architecture in this repository has used CZ.
CZ_TREE = (("cz", 1, 0), ("cz", 3, 2), ("cz", 2, 0))

ALL_FEATURES = tuple(range(8))
LATENT_DIRECTIONS = (0, 1, 4)  # x1, x2, x5


def build_spec(
    name: str,
    qubit_features: Sequence[Sequence[int]],
    n_blocks: int,
    *,
    entangler: Sequence[tuple[str, int, int]] = CX_TREE,
) -> CircuitSpec:
    """Assemble a block-structured spec.

    Each qubit receives its own feature sequence with alternating RY/RZ axes.
    Alternating matters: consecutive rotations about one axis would compose into
    a single rotation whose angle mixes two features, which could read as
    breaking the single-feature angle rule.
    """
    if n_blocks < 1:
        raise ValueError("A candidate needs at least one block.")
    gates: list[Gate] = []
    cursor = 0
    for _ in range(n_blocks):
        for qubit, features in enumerate(qubit_features):
            for position, feature in enumerate(features):
                gates.append(
                    Gate(
                        kind=ALTERNATING_AXES[position % 2],
                        qubit=qubit,
                        feature=feature,
                        scale_index=cursor,
                        bias_index=cursor + 1,
                    )
                )
                cursor += 2
        for kind in ("rz", "ry"):
            for qubit in range(len(qubit_features)):
                gates.append(Gate(kind=kind, qubit=qubit, param_index=cursor))
                cursor += 1
        for entangling_kind, control, target in entangler:
            gates.append(Gate(kind=entangling_kind, control=control, target=target))
    gates.append(Gate(kind="ry", qubit=READOUT_QUBIT, param_index=cursor))
    cursor += 1
    return CircuitSpec(
        name=name,
        n_qubits=len(qubit_features),
        n_weights=cursor,
        readout_qubit=READOUT_QUBIT,
        gates=tuple(gates),
    )


def build_spec_per_block(
    name: str,
    blocks: Sequence[Sequence[Sequence[int]]],
    *,
    entangler: Sequence[tuple[str, int, int]] = CX_TREE,
) -> CircuitSpec:
    """Like ``build_spec`` but each block has its own feature assignment."""
    gates: list[Gate] = []
    cursor = 0
    n_qubits = len(blocks[0])
    for qubit_features in blocks:
        for qubit, features in enumerate(qubit_features):
            for position, feature in enumerate(features):
                gates.append(
                    Gate(
                        kind=ALTERNATING_AXES[position % 2],
                        qubit=qubit,
                        feature=feature,
                        scale_index=cursor,
                        bias_index=cursor + 1,
                    )
                )
                cursor += 2
        for kind in ("rz", "ry"):
            for qubit in range(n_qubits):
                gates.append(Gate(kind=kind, qubit=qubit, param_index=cursor))
                cursor += 1
        for entangling_kind, control, target in entangler:
            gates.append(Gate(kind=entangling_kind, control=control, target=target))
    gates.append(Gate(kind="ry", qubit=READOUT_QUBIT, param_index=cursor))
    cursor += 1
    return CircuitSpec(
        name=name,
        n_qubits=n_qubits,
        n_weights=cursor,
        readout_qubit=READOUT_QUBIT,
        gates=tuple(gates),
    )


# The submitted circuit, expressed as a spec so it can be screened on exactly
# the same instrument as every challenger.  Blocks alternate x1-x4 and x5-x8
# across q0-q3, one feature per qubit per block, with the linear causal funnel.
C1_FUNNEL = (("cx", 3, 2), ("cx", 2, 1), ("cx", 1, 0))
C1_BLOCKS = (
    ((0,), (1,), (2,), (3,)),
    ((4,), (5,), (6,), (7,)),
    ((0,), (1,), (2,), (3,)),
    ((4,), (5,), (6,), (7,)),
)


C1_FUNNEL_CZ = (("cz", 3, 2), ("cz", 2, 1), ("cz", 1, 0))
# Mixed: CZ builds phase correlations, the trailing CX still moves population
# toward the readout. CZ is diagonal, so on its own it cannot change any
# computational-basis amplitude magnitude -- its effect on the measured q0
# probability appears only through rotations that follow it.
C1_FUNNEL_MIXED = (("cz", 3, 2), ("cz", 2, 1), ("cx", 1, 0))


def c1_reference(entangler=C1_FUNNEL, suffix: str = "") -> CircuitSpec:
    return build_spec_per_block(
        f"reference_c1_causal_reupload{suffix}", C1_BLOCKS, entangler=entangler
    )


def all_to_all(n_blocks: int, *, entangler=CX_TREE) -> CircuitSpec:
    """Every qubit sees every raw feature."""
    return build_spec(
        f"candidate_a1_all_to_all_b{n_blocks}",
        (ALL_FEATURES,) * N_QUBITS,
        n_blocks,
        entangler=entangler,
    )


def latent_focused(n_blocks: int, *, entangler=CX_TREE) -> CircuitSpec:
    """Only the three independent directions, uploaded many times each."""
    return build_spec(
        f"candidate_l1_latent_focused_b{n_blocks}",
        (LATENT_DIRECTIONS,) * N_QUBITS,
        n_blocks,
        entangler=entangler,
    )


def pair_focused(n_blocks: int, *, entangler=CX_TREE) -> CircuitSpec:
    """Qubits own pairs of directions, so two-way terms form inside a qubit."""
    return build_spec(
        f"candidate_p1_pair_focused_b{n_blocks}",
        ((0, 1), (0, 4), (1, 4), (0, 1)),
        n_blocks,
        entangler=entangler,
    )


CANDIDATES: dict[str, Callable[[], CircuitSpec]] = {
    "c1ref": c1_reference,
    # Z1: the winning encoding with the diagonal entangler.
    "c1ref_cz": lambda: c1_reference(C1_FUNNEL_CZ, "_cz"),
    "c1ref_cxcz": lambda: c1_reference(C1_FUNNEL_MIXED, "_cxcz"),
    "a1b2": lambda: all_to_all(2),
    "a1b3": lambda: all_to_all(3),
    # Upload sweep for the three directions: 4 qubits x n blocks uploads each.
    "l1b2": lambda: latent_focused(2),   # 8 uploads per direction
    "l1b3": lambda: latent_focused(3),   # 12
    "l1b5": lambda: latent_focused(5),   # 20
    "l1b6": lambda: latent_focused(6),  # 24
    "p1b4": lambda: pair_focused(4),
    "p1b6": lambda: pair_focused(6),
    # Z1: the winning encoding re-run with the diagonal entangler.
    "a1b2_cz": lambda: all_to_all(2, entangler=CZ_TREE),
    "a1b3_cz": lambda: all_to_all(3, entangler=CZ_TREE),
    "l1b5_cz": lambda: latent_focused(5, entangler=CZ_TREE),
}


def build_candidate(name: str) -> CircuitSpec:
    if name not in CANDIDATES:
        raise ValueError(
            f"Unknown candidate {name!r}; available: {', '.join(sorted(CANDIDATES))}"
        )
    return CANDIDATES[name]()
