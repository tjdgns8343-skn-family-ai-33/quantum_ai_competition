"""The encoding note is a required submission document, so it must not drift.

It previously hard-coded "balanced cross-entropy" and kept saying so after the
objective became selectable, and it named the raw columns from zero-based
parameter indices, shifting every column name by one.
"""

from qchallenge.encoding_note import build_encoding_note

FACTS = dict(
    architecture="C1_CAUSAL_REUPLOAD",
    encoding_description="Raw features enter as RY rotations.",
    entanglement_description="Causal CX funnel.",
    readout_qubit=0,
    used_features=range(8),
    uploads_per_feature={f"x_{index}": 2 for index in range(8)},
    qubits=4,
    depth=23,
    two_qubit_gates=12,
    weight_count=65,
    threshold=0.4575,
    threshold_source="train-only 5-fold out-of-fold probabilities",
    initialization="Label-independent uniform random draw.",
)


def test_note_names_csv_columns_not_zero_based_parameters():
    note = build_encoding_note(objective="balanced_bce", **FACTS)
    assert "x1: 2" in note and "x8: 2" in note
    assert "x0:" not in note


def test_note_states_the_objective_actually_used():
    bce = build_encoding_note(objective="balanced_bce", **FACTS)
    auc = build_encoding_note(objective="smooth_auc", **FACTS)
    assert "cross-entropy" in bce and "AUC" not in bce
    assert "AUC" in auc and "positive/negative pairs" in auc


def test_note_records_the_annealing_stage_when_one_was_used():
    note = build_encoding_note(
        objective="smooth_auc",
        annealing={
            "temperature_start": 0.30,
            "temperature_stop": 0.015,
            "stages": 10,
            "submitted_stage": 6,
            "stage_and_temperature_selection": "train-only out-of-fold curve",
        },
        **FACTS,
    )
    assert "stage 6" in note
    assert "train-only out-of-fold curve" in note


def test_note_lists_unused_columns_when_a_subset_is_encoded():
    facts = {**FACTS, "used_features": range(1, 8),
             "uploads_per_feature": {f"x_{i}": 8 for i in range(1, 8)}}
    note = build_encoding_note(objective="smooth_auc", **facts)
    assert "not used: x1" in note
    assert "Announcement #5" in note


def test_note_always_denies_preprocessing_and_classical_pretraining():
    for objective in ("balanced_bce", "soft_balanced_accuracy", "smooth_auc"):
        # The note is wrapped prose, so match on normalized whitespace.
        note = " ".join(build_encoding_note(objective=objective, **FACTS).split())
        assert "There is no scaling, normalization, PCA" in note
        assert "No separately trained classical predictive, surrogate or teacher" in note
        assert "No part of public_test.csv" in note
