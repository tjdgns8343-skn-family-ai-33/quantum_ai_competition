"""Declarative circuit specification shared by the QASM builder and simulator.

Every earlier architecture in this repository hand-wrote a Qiskit builder and a
matching statevector simulator, so the two could silently disagree.  Here both
are generated from one ``CircuitSpec``, which makes that class of bug
impossible and lets a new architecture be described as data.

Compliance is a property of the spec and is checked by ``validate``: a data gate
carries exactly one raw feature and one scale/bias pair, so its angle is always
the permitted ``theta_bias + theta_weight * x_i``.  Combining several features
is done by applying several such gates, which the circuit composes -- never by
arithmetic outside the circuit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter

from .circuit import ALLOWED_GATES

ROTATIONS = ("ry", "rz")


@dataclass(frozen=True)
class Gate:
    """One circuit operation.

    A data rotation sets ``feature``, ``scale_index`` and ``bias_index``.
    A mixer rotation sets ``param_index``.  An entangler sets ``control`` and
    ``target``.
    """

    kind: str
    qubit: int | None = None
    control: int | None = None
    target: int | None = None
    feature: int | None = None
    scale_index: int | None = None
    bias_index: int | None = None
    param_index: int | None = None

    @property
    def is_data(self) -> bool:
        return self.feature is not None


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    n_qubits: int
    n_weights: int
    readout_qubit: int
    gates: tuple[Gate, ...]

    def feature_uses(self) -> Counter:
        return Counter(
            f"x_{gate.feature}" for gate in self.gates if gate.is_data
        )

    def used_features(self) -> tuple[int, ...]:
        return tuple(
            sorted({gate.feature for gate in self.gates if gate.is_data})
        )

    def two_qubit_count(self) -> int:
        return sum(1 for gate in self.gates if gate.kind == "cx")


def validate(spec: CircuitSpec) -> None:
    if not 2 <= spec.n_qubits <= 8:
        raise ValueError(f"{spec.name}: qubit count must be 2..8.")
    if not 0 <= spec.readout_qubit < spec.n_qubits:
        raise ValueError(f"{spec.name}: readout qubit outside the register.")
    seen_weights: set[int] = set()
    for gate in spec.gates:
        if gate.kind == "cx":
            if gate.control is None or gate.target is None:
                raise ValueError(f"{spec.name}: CX needs a control and a target.")
            if gate.control == gate.target:
                raise ValueError(f"{spec.name}: CX control and target must differ.")
            if not all(
                0 <= q < spec.n_qubits for q in (gate.control, gate.target)
            ):
                raise ValueError(f"{spec.name}: CX qubit outside the register.")
            continue
        if gate.kind not in ROTATIONS:
            raise ValueError(f"{spec.name}: unsupported gate {gate.kind!r}.")
        if gate.qubit is None or not 0 <= gate.qubit < spec.n_qubits:
            raise ValueError(f"{spec.name}: rotation qubit outside the register.")
        if gate.is_data:
            if gate.scale_index is None or gate.bias_index is None:
                raise ValueError(
                    f"{spec.name}: a data gate needs both a scale and a bias."
                )
            if gate.param_index is not None:
                raise ValueError(
                    f"{spec.name}: a data gate cannot also be a mixer."
                )
            seen_weights.update((gate.scale_index, gate.bias_index))
        else:
            if gate.param_index is None:
                raise ValueError(f"{spec.name}: a mixer gate needs a parameter.")
            seen_weights.add(gate.param_index)
    if seen_weights and max(seen_weights) >= spec.n_weights:
        raise ValueError(f"{spec.name}: a gate references a weight out of range.")
    if spec.two_qubit_count() < 1:
        raise ValueError(f"{spec.name}: at least one entangling gate is required.")


def build_circuit(spec: CircuitSpec, *, measured: bool = False) -> tuple[
    QuantumCircuit, list[Parameter], list[Parameter]
]:
    """Build the Qiskit circuit the spec describes."""
    validate(spec)
    features = [Parameter(f"x_{index}") for index in range(8)]
    weights = [Parameter(f"theta_{index}") for index in range(spec.n_weights)]
    circuit = QuantumCircuit(spec.n_qubits, name=spec.name)
    for gate in spec.gates:
        if gate.kind == "cx":
            circuit.cx(gate.control, gate.target)
            continue
        if gate.is_data:
            angle = (
                weights[gate.scale_index] * features[gate.feature]
                + weights[gate.bias_index]
            )
        else:
            angle = weights[gate.param_index]
        getattr(circuit, gate.kind)(angle, gate.qubit)
    if measured:
        circuit.add_register(ClassicalRegister(1, "c"))
        circuit.measure(spec.readout_qubit, circuit.clbits[0])
    used = spec.used_features()
    return circuit, [features[index] for index in used], weights


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
