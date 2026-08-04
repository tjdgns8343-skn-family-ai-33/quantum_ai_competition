from qchallenge.circuit import (
    N_FEATURES,
    N_WEIGHTS,
    FEATURE_LAYOUTS,
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


def test_all_pre_registered_layouts_remain_direct_and_constrained():
    for layout in FEATURE_LAYOUTS:
        circuit, _, _ = build_submission_circuit(layout)
        assert constraint_report(circuit)["passes"]
        seen = []
        for instruction in circuit.data:
            for expression in instruction.operation.params:
                names = [
                    str(parameter)
                    for parameter in getattr(expression, "parameters", set())
                    if str(parameter).startswith("x_")
                ]
                assert len(names) <= 1
                if names:
                    assert str(expression) == names[0]
                    seen.extend(names)
        assert sorted(seen) == [f"x_{i}" for i in range(8)]


def test_feature_subset_omits_unselected_parameters_without_transforming_selected():
    selected = (0, 2, 3, 6)
    circuit, features, _ = build_submission_circuit(
        "proposed_1347_2568", selected
    )
    assert {str(parameter) for parameter in features} == {
        "x_0", "x_2", "x_3", "x_6"
    }
    assert constraint_report(circuit)["passes"]
    seen = []
    for instruction in circuit.data:
        for expression in instruction.operation.params:
            names = [
                str(parameter)
                for parameter in getattr(expression, "parameters", set())
                if str(parameter).startswith("x_")
            ]
            if names:
                assert len(names) == 1
                assert str(expression) == names[0]
                seen.extend(names)
    assert sorted(seen) == ["x_0", "x_2", "x_3", "x_6"]


def test_packed_feature_order_uses_first_active_b1_slots():
    selected = (7, 1, 5)
    circuit, features, _ = build_unitary(
        "sequential", selected, pack_selected_features=True
    )
    assert [str(parameter) for parameter in features] == ["x_7", "x_1", "x_5"]
    data_expressions = [
        str(instruction.operation.params[0])
        for instruction in circuit.data
        if instruction.operation.params
        and any(
            str(parameter).startswith("x_")
            for parameter in getattr(
                instruction.operation.params[0], "parameters", set()
            )
        )
    ]
    assert data_expressions == ["x_7", "x_1", "x_5"]
