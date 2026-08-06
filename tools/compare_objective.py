"""A/B the training objective on the full train set with a shared warm start.

Both arms use the identical balanced-BCE multi-restart warm start, so any
difference comes only from the annealed soft balanced-accuracy stage that
follows it.  Balanced accuracy is what the competition scores, but cross-entropy
keeps paying to push already-confident rows further from the threshold; on the
trained C1 circuit 82% of rows sit more than 0.1 away from it, so most of the
cross-entropy gradient lands where the score cannot change.

Train-only: the objective reads circuit probabilities and raw train labels, the
holdout comes from public_train.csv, and public_test.csv is never touched.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qchallenge.c1_simulator import c1_exact_probabilities
from qchallenge.c1_training import (
    optimize_c1_quantum_loss,
    optimize_c1_annealed,
)
from qchallenge.data import load_raw_train
from qchallenge.metrics import balanced_accuracy, balanced_binary_cross_entropy
from qchallenge.training import stratified_holdout


def _report(x: np.ndarray, y: np.ndarray, weights: np.ndarray, threshold: float) -> dict:
    probability = c1_exact_probabilities(x, weights)
    return {
        "balanced_accuracy": balanced_accuracy(
            y, (probability >= threshold).astype(int)
        ),
        "balanced_binary_cross_entropy": balanced_binary_cross_entropy(y, probability),
        "rows_within_0_05_of_threshold": int(
            np.sum(np.abs(probability - threshold) < 0.05)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--init-scale", type=float, default=1.5)
    parser.add_argument("--affine-scale-jitter", type=float, default=0.6)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature-start", type=float, default=0.20)
    parser.add_argument("--temperature-stop", type=float, default=0.02)
    parser.add_argument("--anneal-stages", type=int, default=5)
    parser.add_argument("--stage-maxiter", type=int, default=60)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    x, y, data_info = load_raw_train(Path(args.data_dir) / "public_train.csv")
    fit, valid = stratified_holdout(y, args.validation_fraction, args.split_seed)
    shared = dict(
        seed=args.seed,
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        n_restarts=args.restarts,
    )

    bce_weights, bce_summary, _ = optimize_c1_quantum_loss(x[fit], y[fit], **shared)
    soft_weights, soft_summary, _ = optimize_c1_annealed(
        x[fit],
        y[fit],
        threshold=args.threshold,
        temperature_start=args.temperature_start,
        temperature_stop=args.temperature_stop,
        anneal_stages=args.anneal_stages,
        stage_maxiter=args.stage_maxiter,
        **shared,
    )

    payload = {
        "mode": "objective_ab_shared_warm_start",
        "architecture": "C1_CAUSAL_REUPLOAD",
        "submission_created": False,
        "public_test_used": False,
        "classical_predictive_model_used": False,
        "data": data_info,
        "config": vars(args),
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(valid)),
        "balanced_bce": {
            "fit": _report(x[fit], y[fit], bce_weights, args.threshold),
            "validation": _report(x[valid], y[valid], bce_weights, args.threshold),
            "optimization": bce_summary,
        },
        "annealed_soft_balanced_accuracy": {
            "fit": _report(x[fit], y[fit], soft_weights, args.threshold),
            "validation": _report(x[valid], y[valid], soft_weights, args.threshold),
            "optimization": soft_summary,
        },
    }
    run_dir = Path(args.artifacts_dir) / (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_objective_ab_"
        f"split{args.split_seed}_init{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "objective_ab.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    for name in ("balanced_bce", "annealed_soft_balanced_accuracy"):
        arm = payload[name]
        print(
            f"{name:38s} fit BA {arm['fit']['balanced_accuracy']:.4f}   "
            f"val BA {arm['validation']['balanced_accuracy']:.4f}"
        )
    for stage in soft_summary["anneal_stages"]:
        print(
            f"  T={stage['temperature']:.4f}  train BA "
            f"{stage['train_balanced_accuracy_before']:.4f} -> "
            f"{stage['train_balanced_accuracy_after']:.4f}"
        )
    print(run_dir)


if __name__ == "__main__":
    main()
