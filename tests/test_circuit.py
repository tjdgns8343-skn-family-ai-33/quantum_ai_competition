from qchallenge.circuit import (
    N_FEATURES,
    N_WEIGHTS,
    build_submission_circuit,
    build_unitary,
    constraint_report,
)


def test_b1_constraints_and_parameters():
    circuit, features, weights = build_submission_circuit()
    report = constraint_report(circuit)
    assert report["passes"]
    assert len(features) == N_FEATURES
    assert len(weights) == N_WEIGHTS
    assert {str(p) for p in features} == {f"x_{i}" for i in range(8)}
    assert {str(p) for p in weights} == {f"theta_{i}" for i in range(16)}


def test_every_data_gate_has_one_raw_feature_only():
    circuit, _, _ = build_unitary()
    seen = []
    for instruction in circuit.data:
        for expression in instruction.operation.params:
            data_parameters = {str(p) for p in getattr(expression, "parameters", set()) if str(p).startswith("x_")}
            assert len(data_parameters) <= 1
            if data_parameters:
                assert str(expression) in data_parameters
                seen.extend(data_parameters)
    assert sorted(seen) == [f"x_{i}" for i in range(8)]

