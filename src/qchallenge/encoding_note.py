"""Builds the submission's required encoding note from the actual run config.

The competition asks every submission to describe how the training data was
encoded into the circuit.  The note used to be a hand-written string per
architecture, which meant it silently kept claiming cross-entropy training after
the objective became selectable.  Rendering it from the same values the run
actually used removes that failure mode.
"""

from __future__ import annotations

from collections.abc import Sequence

OBJECTIVE_SENTENCES = {
    "balanced_bce": (
        "Training minimized the class-balanced binary cross-entropy between that "
        "measured probability and the raw public_train.csv label."
    ),
    "soft_balanced_accuracy": (
        "Training first minimized the class-balanced binary cross-entropy of that "
        "measured probability against the raw public_train.csv label, then annealed "
        "a smooth surrogate of balanced accuracy, sigmoid((p - threshold) / T), "
        "lowering T so the objective approaches the scored metric."
    ),
    "smooth_auc": (
        "Training first minimized the class-balanced binary cross-entropy of that "
        "measured probability against the raw public_train.csv label, then annealed "
        "a smooth surrogate of ROC AUC, the mean of sigmoid((p_pos - p_neg) / T) "
        "over all positive/negative pairs of training rows, lowering T. Both stages "
        "read nothing but the circuit's measured probability and the raw training "
        "label."
    ),
}


def build_encoding_note(
    *,
    architecture: str,
    encoding_description: str,
    entanglement_description: str,
    readout_qubit: int,
    used_features: Sequence[int],
    uploads_per_feature: dict[str, int],
    qubits: int,
    depth: int,
    two_qubit_gates: int,
    weight_count: int,
    objective: str,
    threshold: float,
    threshold_source: str,
    initialization: str,
    annealing: dict | None = None,
) -> str:
    feature_names = ", ".join(f"x{index + 1}" for index in used_features)
    unused = [
        f"x{index + 1}" for index in range(8) if index not in set(used_features)
    ]
    # Parameter names are zero-based (x_0..x_7) but the CSV columns are x1..x8,
    # so shift when naming columns; an off-by-one here would misdescribe the
    # submission to the reviewers.
    uploads = ", ".join(
        f"x{int(name.split('_')[1]) + 1}: {count}"
        for name, count in sorted(
            uploads_per_feature.items(), key=lambda item: int(item[0].split("_")[1])
        )
    )

    lines = [
        f"ENCODING NOTE - {architecture}",
        "",
        "1. How the training data enters the circuit",
        "",
        f"   Raw columns used: {feature_names}.",
    ]
    if unused:
        lines.append(
            f"   Raw columns deliberately not used: {', '.join(unused)}. "
            "Announcement #5 permits using a subset of the raw features."
        )
    lines += [
        "   The CSV values are passed to the circuit exactly as stored. There is no",
        "   scaling, normalization, PCA or other dimensionality reduction, imputation,",
        "   augmentation, binning, polynomial or product feature, kernel evaluation, or",
        "   any other transformation applied before or outside the circuit.",
        "",
        f"   {encoding_description}",
        "",
        "   Every angle of every single-qubit rotation therefore has the permitted",
        "   single-feature affine form theta_bias + theta_weight * x_i. No rotation",
        "   angle contains a product of features, a power of a feature, an arithmetic",
        "   combination of several features, or any constant derived from data",
        "   statistics. Interactions between features are produced only by the",
        "   entangling gates inside the circuit.",
        "",
        f"   Uploads per raw feature: {uploads}.",
        "",
        "2. Circuit structure",
        "",
        f"   {entanglement_description}",
        f"   Qubits: {qubits}. Depth: {depth}. Two-qubit gates: {two_qubit_gates}.",
        f"   Trainable parameters: {weight_count}.",
        f"   Exactly one qubit is measured: q{readout_qubit}. The classifier compares",
        f"   its measured probability against the threshold below.",
        "",
        "3. How the parameters were produced",
        "",
        f"   {initialization}",
        f"   {OBJECTIVE_SENTENCES[objective]}",
        "   The optimizer is L-BFGS-B driven by the exact adjoint gradient of the",
        "   circuit itself. The training emulator's probabilities were checked against",
        "   qiskit.quantum_info.Statevector and agree to within 1e-10.",
    ]
    if annealing is not None:
        lines += [
            "",
            f"   Annealing schedule: T from {annealing['temperature_start']} to "
            f"{annealing['temperature_stop']} in {annealing['stages']} stages;"
            f" the submitted parameters are from stage {annealing['submitted_stage']}.",
            f"   That stage was chosen by {annealing['stage_and_temperature_selection']}.",
        ]
    lines += [
        "",
        "   No separately trained classical predictive, surrogate or teacher model was",
        "   used at any point. No coefficient, weight, prediction, residual or other",
        "   output of such a model was transferred into any theta, as an initial value",
        "   or otherwise. There is no warm start from any external file.",
        "",
        "4. Decision threshold",
        "",
        f"   threshold = {threshold:g}.",
        f"   Source: {threshold_source}.",
        "",
        "5. Data usage",
        "",
        "   Only public_train.csv was used for training, for threshold selection and",
        "   for every model-selection decision. No part of public_test.csv and no",
        "   leaderboard feedback entered training, initialization, the objective, the",
        "   threshold, or the choice between candidates.",
        "",
    ]
    return "\n".join(lines)
