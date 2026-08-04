"""Mechanical checks supporting, but not replacing, training-process review."""

from __future__ import annotations

import json
from pathlib import Path

from qiskit import qasm3

from .circuit import N_FEATURES, N_WEIGHTS, constraint_report

BANNED_SOURCE_TOKENS = (
    "sklearn.linear_model",
    "sklearn.svm",
    "sklearn.ensemble",
    "xgboost",
    "lightgbm",
    "catboost",
    "LogisticRegression",
    "HistGradientBoosting",
    "teacher_model.fit",
)


def audit_source(source_root: Path) -> dict:
    findings: list[dict] = []
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in BANNED_SOURCE_TOKENS:
            if token in text and path.name != "compliance.py":
                findings.append({"path": str(path), "token": token})
    return {"passes": not findings, "findings": findings, "checked_root": str(source_root)}


def audit_artifact(directory: Path) -> dict:
    required = {
        "classifier.qasm", "weights.json", "parameter_provenance.json",
        "training_metrics.json", "encoding_note.txt",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        return {"passes": False, "missing": missing}
    circuit = qasm3.loads((directory / "classifier.qasm").read_text(encoding="utf-8"))
    constraints = constraint_report(circuit)
    parameter_names = {str(parameter) for parameter in circuit.parameters}
    expected_parameters = {f"x_{i}" for i in range(N_FEATURES)} | {f"theta_{i}" for i in range(N_WEIGHTS)}
    data_gate_violations: list[dict] = []
    observed_features: list[str] = []
    measured_qubits: list[int] = []
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            measured_qubits.extend(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        for expression in instruction.operation.params:
            data_parameters = sorted(
                str(parameter)
                for parameter in getattr(expression, "parameters", set())
                if str(parameter).startswith("x_")
            )
            if not data_parameters:
                continue
            observed_features.extend(data_parameters)
            if len(data_parameters) != 1 or str(expression) != data_parameters[0]:
                data_gate_violations.append(
                    {
                        "gate": instruction.operation.name,
                        "expression": str(expression),
                        "data_parameters": data_parameters,
                    }
                )
    weights = json.loads((directory / "weights.json").read_text(encoding="utf-8"))
    expected_weight_keys = {f"theta_{i}" for i in range(N_WEIGHTS)} | {"threshold"}
    provenance = json.loads((directory / "parameter_provenance.json").read_text(encoding="utf-8"))
    provenance_passes = bool(
        provenance.get("classical_predictive_model") is None
        and provenance.get("surrogate_model") is None
        and provenance.get("teacher_model") is None
        and provenance.get("warm_start") is False
        and provenance.get("transferred_coefficients") is False
        and provenance.get("feature_processing") == "none"
    )
    checks = {
        "constraints": constraints["passes"],
        "parameter_names": parameter_names == expected_parameters,
        "single_raw_feature_per_data_gate": not data_gate_violations,
        "all_raw_features_used_once": sorted(observed_features) == [f"x_{i}" for i in range(N_FEATURES)],
        "only_q0_measured": measured_qubits == [0],
        "weight_names": set(weights) == expected_weight_keys,
        "fixed_threshold": float(weights.get("threshold", -1)) == 0.5,
        "provenance": provenance_passes,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "constraint_report": constraints,
        "data_gate_violations": data_gate_violations,
        "measured_qubits": measured_qubits,
    }
