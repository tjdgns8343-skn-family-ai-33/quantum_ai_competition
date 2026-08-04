from pathlib import Path

from qchallenge.compliance import audit_source
from qchallenge.training import random_initial_point


def test_source_contains_no_banned_predictive_model_code():
    source_root = Path(__file__).resolve().parents[1] / "src"
    assert audit_source(source_root)["passes"]


def test_random_initialization_is_reproducible_and_label_independent():
    first = random_initial_point(2026, 0.05)
    second = random_initial_point(2026, 0.05)
    third = random_initial_point(2027, 0.05)
    assert (first == second).all()
    assert not (first == third).all()
