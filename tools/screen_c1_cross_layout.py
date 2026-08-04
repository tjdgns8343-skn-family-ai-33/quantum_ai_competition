"""Screen competition-compliant C1 second-upload feature layouts.

This experiment changes only the assignment of raw features to the four
single-feature affine RY gates in upload blocks 3 and 4.  Circuit topology,
initialization, optimizer, loss, sample selection, and validation are inherited
unchanged from C1R.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

import qchallenge.c1r_circuit as circuit_module
import qchallenge.c1r_simulator as simulator_module
import qchallenge.c1r_training as training_module


LAYOUTS: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "rotate": (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (3, 0, 1, 2),
        (7, 4, 5, 6),
    ),
    "zip": (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 4, 1, 5),
        (2, 6, 3, 7),
    ),
    "swap": (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (4, 5, 6, 7),
        (0, 1, 2, 3),
    ),
}


def activate_layout(name: str) -> tuple[tuple[int, int, int, int], ...]:
    blocks = LAYOUTS[name]
    architecture = f"C1_{name.upper()}_SECOND_UPLOAD"
    # These modules intentionally share the same immutable layout object so
    # circuit construction, the exact simulator, and training metadata agree.
    circuit_module.C1R_REUPLOAD_BLOCKS = blocks
    circuit_module.C1R_ARCHITECTURE = architecture
    simulator_module.C1R_REUPLOAD_BLOCKS = blocks
    training_module.C1R_REUPLOAD_BLOCKS = blocks
    training_module.C1R_ARCHITECTURE = architecture
    return blocks


def validate_layout(name: str) -> dict:
    blocks = activate_layout(name)
    submission, _, _ = circuit_module.build_c1r_submission_circuit()
    constraint = circuit_module.c1r_constraint_report(submission)
    if not constraint["passes"]:
        raise RuntimeError(f"{name} constraint failure: {constraint}")

    seen: list[str] = []
    unitary, features, weights = circuit_module.build_c1r_unitary()
    for instruction in unitary.data:
        for expression in instruction.operation.params:
            raw_parameters = [
                parameter
                for parameter in getattr(expression, "parameters", set())
                if str(parameter).startswith("x_")
            ]
            if len(raw_parameters) > 1:
                raise RuntimeError(f"{name} contains a multi-feature angle")
            if raw_parameters:
                raw = raw_parameters[0]
                seen.append(str(raw))
                derivative = expression.gradient(raw)
                second = derivative.gradient(raw) if hasattr(derivative, "gradient") else 0.0
                if str(second) not in {"0", "0.0"}:
                    raise RuntimeError(f"{name} contains a non-affine data angle")
    expected = sorted([f"x_{index}" for index in range(8)] * 2)
    if sorted(seen) != expected:
        raise RuntimeError(f"{name} does not use every raw feature exactly twice")

    rng = np.random.default_rng(1408)
    x = rng.normal(size=(8, 8))
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    initial = training_module.c1r_random_initial_point(
        2026, local_scale=0.05, affine_scale_jitter=0.1
    )
    exact = simulator_module.c1r_exact_probabilities(x, initial)
    qiskit_probabilities: list[float] = []
    for row in x:
        bindings = {
            **dict(zip(features, row, strict=True)),
            **dict(zip(weights, initial, strict=True)),
        }
        state = Statevector.from_instruction(unitary.assign_parameters(bindings))
        qiskit_probabilities.append(float(state.probabilities([0])[1]))
    equivalence_error = float(
        np.max(np.abs(exact - np.asarray(qiskit_probabilities)))
    )
    if equivalence_error >= 1e-10:
        raise RuntimeError(f"{name} simulator/Qiskit mismatch: {equivalence_error}")

    _, gradient = simulator_module.c1r_balanced_bce_value_and_gradient(x, y, initial)
    epsilon = 1e-6
    gradient_errors: list[float] = []
    for index in (0, 16, 32, 48, 64):
        direction = np.zeros(65)
        direction[index] = epsilon
        plus, _ = simulator_module.c1r_balanced_bce_value_and_gradient(
            x, y, initial + direction
        )
        minus, _ = simulator_module.c1r_balanced_bce_value_and_gradient(
            x, y, initial - direction
        )
        gradient_errors.append(abs(gradient[index] - (plus - minus) / (2 * epsilon)))
    maximum_gradient_error = float(max(gradient_errors))
    if maximum_gradient_error >= 1e-6:
        raise RuntimeError(f"{name} adjoint gradient mismatch: {maximum_gradient_error}")

    causal = training_module.c1r_feature_causal_report(x, initial)
    if not causal["all_features_nonzero_at_1e_10"]:
        raise RuntimeError(f"{name} has a feature outside q0's causal cone")
    return {
        "layout": name,
        "blocks_zero_based": blocks,
        "constraint_report": constraint,
        "simulator_qiskit_max_abs_error": equivalence_error,
        "adjoint_gradient_max_abs_error": maximum_gradient_error,
        "all_features_in_q0_causal_cone": True,
        "raw_feature_uses": {f"x{index + 1}": seen.count(f"x_{index}") for index in range(8)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--init-seed", type=int, default=2026)
    parser.add_argument("--max-rows", type=int, default=1200)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    validation = validate_layout(args.layout)
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    if args.train_csv is None:
        parser.error("--train-csv is required unless --validate-only is used")

    output_root = args.artifacts_dir / f"c1_{args.layout}"
    config = training_module.C1RTrainConfig(
        train_csv=args.train_csv,
        artifacts_dir=output_root,
        seed=args.init_seed,
        split_seed=args.split_seed,
        max_rows=args.max_rows,
        validation_fraction=args.validation_fraction,
        maxiter=args.maxiter,
        shots=args.shots,
    )
    run_dir = training_module.run_c1r_screen(config)
    (run_dir / "layout_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(run_dir)


if __name__ == "__main__":
    main()
