"""Mechanical checks supporting, but not replacing, training-process review."""

from __future__ import annotations

import json
from collections import Counter
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
    provenance = json.loads(
        (directory / "parameter_provenance.json").read_text(encoding="utf-8")
    )
    selected_features = tuple(
        int(index)
        for index in provenance.get(
            "selected_raw_features_zero_based", range(N_FEATURES)
        )
    )
    parameter_names = {str(parameter) for parameter in circuit.parameters}
    expected_feature_names = {f"x_{i}" for i in selected_features}
    weight_count = int(provenance.get("weight_count", N_WEIGHTS))
    expected_parameters = expected_feature_names | {
        f"theta_{i}" for i in range(weight_count)
    }
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
            is_affine = False
            if len(data_parameters) == 1:
                feature_parameter = next(
                    parameter
                    for parameter in getattr(expression, "parameters", set())
                    if str(parameter) == data_parameters[0]
                )
                first_derivative = expression.gradient(feature_parameter)
                second_derivative = (
                    first_derivative.gradient(feature_parameter)
                    if hasattr(first_derivative, "gradient")
                    else 0.0
                )
                is_affine = str(second_derivative) in {"0", "0.0"}
            if len(data_parameters) != 1 or not is_affine:
                data_gate_violations.append(
                    {
                        "gate": instruction.operation.name,
                        "expression": str(expression),
                        "data_parameters": data_parameters,
                    }
                )
    weights = json.loads((directory / "weights.json").read_text(encoding="utf-8"))
    expected_weight_keys = {f"theta_{i}" for i in range(weight_count)} | {"threshold"}
    default_feature_uses = {name: 1 for name in expected_feature_names}
    expected_feature_uses = {
        str(name): int(count)
        for name, count in provenance.get(
            "expected_feature_uses", default_feature_uses
        ).items()
    }
    observed_feature_uses = Counter(observed_features)
    provenance_threshold = float(
        provenance.get("decision_threshold", {}).get("value", 0.5)
    )
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
        "selected_raw_features_used_expected_number": observed_feature_uses
        == Counter(expected_feature_uses),
        "selected_feature_indices_valid": bool(
            selected_features
            and len(set(selected_features)) == len(selected_features)
            and all(0 <= index < N_FEATURES for index in selected_features)
        ),
        "only_q0_measured": measured_qubits == [0],
        "weight_names": set(weights) == expected_weight_keys,
        "threshold_matches_provenance": bool(
            0.0 < provenance_threshold < 1.0
            and abs(float(weights.get("threshold", -1)) - provenance_threshold) <= 1e-12
        ),
        "provenance": provenance_passes,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "constraint_report": constraints,
        "data_gate_violations": data_gate_violations,
        "observed_feature_uses": dict(observed_feature_uses),
        "expected_feature_uses": expected_feature_uses,
        "measured_qubits": measured_qubits,
    }
