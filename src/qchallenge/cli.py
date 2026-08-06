"""Command-line interface for screening, final training, auditing and packaging."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .c1_training import (
    C1CrossValidationConfig,
    C1TrainConfig,
    run_c1_cross_validation,
    run_c1_final,
    run_c1_screen,
)
from .c1o_training import C1OTrainConfig, run_c1o_screen
from .c1x_training import C1XTrainConfig, run_c1x_screen
from .c2_training import (
    C2CrossValidationConfig,
    C2TrainConfig,
    run_c2_cross_validation,
    run_c2_final,
    run_c2_screen,
)
from .c3_training import C3TrainConfig, run_c3_screen
from .d1_training import D1TrainConfig, run_d1_screen
from .d1s_training import D1STrainConfig, run_d1s_screen
from .d1o_training import (
    D1OCrossValidationConfig,
    D1OTrainConfig,
    run_d1o_cross_validation,
    run_d1o_final,
    run_d1o_screen,
)
from .e1s_training import (
    E1SCrossValidationConfig,
    E1STrainConfig,
    run_e1s_cross_validation,
    run_e1s_screen,
)
from .f1_training import (
    F1CrossValidationConfig,
    F1TrainConfig,
    run_f1_cross_validation,
    run_f1_final,
    run_f1_screen,
)
from .compliance import audit_artifact, audit_source
from .evaluate import evaluate_artifact_on_csv
from .candidates import CANDIDATES, build_candidate
from .g1_circuit import G1_DEFAULT_BLOCKS, build_g1_spec
from .spec_training import SpecTrainConfig, run_spec_cross_validation, run_spec_screen
from .circuit import FEATURE_LAYOUTS
from .feature_search import FeatureSearchConfig, run_feature_search
from .training import TrainConfig, run_final, run_screen


def _train_config(args: argparse.Namespace) -> TrainConfig:
    selected_features = tuple(value - 1 for value in args.features)
    return TrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed,
        init_scale=args.init_scale,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        feature_layout=args.feature_layout,
        selected_features=selected_features,
        pack_selected_features=args.pack_features,
    )


def _add_training_args(parser: argparse.ArgumentParser, *, validation: bool) -> None:
    parser.add_argument("--data-dir", default="raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument(
        "--feature-layout", choices=tuple(FEATURE_LAYOUTS), default="sequential"
    )
    parser.add_argument(
        "--pack-features",
        action="store_true",
        help="Map listed features in order to RY q0-q3 then RZ q0.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=int,
        default=list(range(1, 9)),
        help="One-based raw feature numbers; for example --features 1 3 4 7",
    )
    if validation:
        parser.add_argument("--validation-fraction", type=float, default=0.2)


def _add_c1_training_args(parser: argparse.ArgumentParser, *, validation: bool) -> None:
    parser.add_argument("--data-dir", default="raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--affine-scale-jitter", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="Label-independent random restarts; the lowest training loss wins.",
    )
    if validation:
        parser.add_argument("--validation-fraction", type=float, default=0.2)
        parser.add_argument(
            "--max-rows",
            type=int,
            default=None,
            help="Train-only stratified screen size; omit to use all 6000 rows.",
        )


def _c1_config(args: argparse.Namespace, *, validation: bool) -> C1TrainConfig:
    init_seed = getattr(args, "init_seed", None)
    return C1TrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if init_seed is None else init_seed,
        split_seed=getattr(args, "split_seed", None),
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
        n_restarts=getattr(args, "restarts", 1),
        objective=getattr(args, "objective", "balanced_bce"),
        temperature_start=getattr(args, "temperature_start", 0.30),
        temperature_stop=getattr(args, "temperature_stop", 0.015),
        anneal_stages=getattr(args, "anneal_stages", 10),
        stage_maxiter=getattr(args, "stage_maxiter", 40),
        select_stage=getattr(args, "select_stage", None),
    )


def _c2_config(args: argparse.Namespace, *, validation: bool) -> C2TrainConfig:
    return C2TrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed,
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
    )


def _c3_config(args: argparse.Namespace, *, validation: bool) -> C3TrainConfig:
    return C3TrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if args.init_seed is None else args.init_seed,
        split_seed=args.split_seed,
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
    )


def _c1o_config(args: argparse.Namespace, *, validation: bool) -> C1OTrainConfig:
    init_seed = getattr(args, "init_seed", None)
    return C1OTrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if init_seed is None else init_seed,
        split_seed=getattr(args, "split_seed", None),
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
    )


def _c1x_config(args: argparse.Namespace, *, validation: bool) -> C1XTrainConfig:
    init_seed = getattr(args, "init_seed", None)
    return C1XTrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if init_seed is None else init_seed,
        split_seed=getattr(args, "split_seed", None),
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
    )


def _d1_config(args: argparse.Namespace, *, validation: bool) -> D1TrainConfig:
    return D1TrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if args.init_seed is None else args.init_seed,
        split_seed=args.split_seed,
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
    )


def _d1s_config(args: argparse.Namespace, *, validation: bool) -> D1STrainConfig:
    return D1STrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if args.init_seed is None else args.init_seed,
        split_seed=args.split_seed,
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
    )


def _d1o_config(args: argparse.Namespace, *, validation: bool) -> D1OTrainConfig:
    init_seed = getattr(args, "init_seed", None)
    return D1OTrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if init_seed is None else init_seed,
        split_seed=getattr(args, "split_seed", None),
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
    )


def _e1s_config(args: argparse.Namespace, *, validation: bool) -> E1STrainConfig:
    init_seed = getattr(args, "init_seed", None)
    return E1STrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if init_seed is None else init_seed,
        split_seed=getattr(args, "split_seed", None),
        init_scale=args.init_scale,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
    )


def _f1_config(args: argparse.Namespace, *, validation: bool) -> F1TrainConfig:
    init_seed = getattr(args, "init_seed", None)
    return F1TrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed if init_seed is None else init_seed,
        split_seed=getattr(args, "split_seed", None),
        n_blocks=args.blocks,
        init_scale=args.init_scale,
        affine_scale_center=args.affine_scale_center,
        affine_scale_jitter=args.affine_scale_jitter,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
        max_rows=getattr(args, "max_rows", None) if validation else None,
        decision_threshold=args.threshold,
        n_restarts=getattr(args, "restarts", 1),
    )


def _add_f1_training_args(parser: argparse.ArgumentParser, *, validation: bool) -> None:
    _add_c1_training_args(parser, validation=validation)
    parser.set_defaults(maxiter=200)
    parser.add_argument(
        "--blocks",
        type=int,
        default=8,
        help="Re-uploading blocks; 8 fills the depth-50 and 80 two-qubit-gate budget.",
    )
    parser.add_argument("--affine-scale-center", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--init-seed", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("screen", help="Train/evaluate on a train-only holdout; creates no submission")
    _add_training_args(screen, validation=True)
    final = sub.add_parser("train-final", help="Train directly on the full public train set")
    _add_training_args(final, validation=False)
    c1_screen = sub.add_parser(
        "screen-c1",
        help="Train/evaluate the causal affine data-reuploading C1 VQC",
    )
    _add_c1_training_args(c1_screen, validation=True)
    c1_screen.add_argument("--split-seed", type=int, default=None)
    c1_screen.add_argument("--init-seed", type=int, default=None)
    c1_final = sub.add_parser(
        "train-final-c1",
        help="Train C1 directly on the complete public train set",
    )
    _add_c1_training_args(c1_final, validation=False)
    c1_final.add_argument(
        "--objective",
        choices=("balanced_bce", "soft_balanced_accuracy", "smooth_auc", "smooth_ks"),
        default="balanced_bce",
    )
    c1_final.add_argument("--temperature-start", type=float, default=0.30)
    c1_final.add_argument("--temperature-stop", type=float, default=0.015)
    c1_final.add_argument("--anneal-stages", type=int, default=10)
    c1_final.add_argument("--stage-maxiter", type=int, default=40)
    c1_final.add_argument(
        "--select-stage",
        type=int,
        default=None,
        help="Annealing stage to submit, chosen from the train-only OOF curve.",
    )
    c1_cv = sub.add_parser(
        "cross-validate-c1",
        help="Generate C1 quantum-only OOF probabilities and threshold",
    )
    c1_cv.add_argument("--data-dir", default="raw")
    c1_cv.add_argument("--artifacts-dir", default="artifacts")
    c1_cv.add_argument("--split-seed", type=int, default=2026)
    c1_cv.add_argument("--init-seed", type=int, default=2026)
    c1_cv.add_argument("--init-scale", type=float, default=0.05)
    c1_cv.add_argument("--affine-scale-jitter", type=float, default=0.1)
    c1_cv.add_argument("--maxiter", type=int, default=80)
    c1_cv.add_argument("--folds", type=int, default=5)
    c1_cv.add_argument("--restarts", type=int, default=1)
    c1_cv.add_argument(
        "--objective",
        choices=("balanced_bce", "soft_balanced_accuracy", "smooth_auc", "smooth_ks"),
        default="balanced_bce",
        help="Annealed surrogates warm up on BCE; smooth_auc scores all pairs instead of the threshold band.",
    )
    c1_cv.add_argument("--temperature-start", type=float, default=0.20)
    c1_cv.add_argument("--temperature-stop", type=float, default=0.02)
    c1_cv.add_argument("--anneal-stages", type=int, default=6)
    c1_cv.add_argument("--stage-maxiter", type=int, default=50)
    c1_cv.add_argument("--shots", type=int, default=1024)
    c1o_screen = sub.add_parser(
        "screen-c1o",
        help="Train/evaluate the four-qubit, exactly-one-upload affine C1-O VQC",
    )
    _add_c1_training_args(c1o_screen, validation=True)
    c1o_screen.set_defaults(maxiter=60)
    c1o_screen.add_argument("--split-seed", type=int, default=None)
    c1o_screen.add_argument("--init-seed", type=int, default=None)
    c1x_screen = sub.add_parser(
        "screen-c1x",
        help="Train/evaluate C1 with an odd/even cross-paired second upload",
    )
    _add_c1_training_args(c1x_screen, validation=True)
    c1x_screen.set_defaults(maxiter=60)
    c1x_screen.add_argument("--split-seed", type=int, default=None)
    c1x_screen.add_argument("--init-seed", type=int, default=None)
    c2_screen = sub.add_parser(
        "screen-c2",
        help="Train/evaluate the three-round causal affine C2 VQC",
    )
    _add_c1_training_args(c2_screen, validation=True)
    c2_screen.set_defaults(maxiter=120)
    c2_final = sub.add_parser(
        "train-final-c2",
        help="Train C2 directly on the complete public train set",
    )
    _add_c1_training_args(c2_final, validation=False)
    c2_final.set_defaults(maxiter=120)
    c2_cv = sub.add_parser(
        "cross-validate-c2",
        help="Generate C2 quantum-only OOF probabilities and threshold",
    )
    c2_cv.add_argument("--data-dir", default="raw")
    c2_cv.add_argument("--artifacts-dir", default="artifacts")
    c2_cv.add_argument("--split-seed", type=int, default=2026)
    c2_cv.add_argument("--init-seed", type=int, default=2027)
    c2_cv.add_argument("--init-scale", type=float, default=0.05)
    c2_cv.add_argument("--affine-scale-jitter", type=float, default=0.1)
    c2_cv.add_argument("--maxiter", type=int, default=120)
    c2_cv.add_argument("--folds", type=int, default=5)
    c2_cv.add_argument("--shots", type=int, default=1024)
    c3_screen = sub.add_parser(
        "screen-c3",
        help="Train/evaluate the four-round causal affine C3 VQC",
    )
    _add_c1_training_args(c3_screen, validation=True)
    c3_screen.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for train-only subsampling and holdout splitting.",
    )
    c3_screen.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Independent label-free parameter initialization seed.",
    )
    d1_screen = sub.add_parser(
        "screen-d1",
        help="Train/evaluate the two-qubit, exactly-two-upload deep D1 VQC",
    )
    _add_c1_training_args(d1_screen, validation=True)
    d1_screen.set_defaults(maxiter=60)
    d1_screen.add_argument("--split-seed", type=int, default=None)
    d1_screen.add_argument("--init-seed", type=int, default=None)
    d1s_screen = sub.add_parser(
        "screen-d1s",
        help="Train/evaluate the depth-29 two-qubit, two-upload D1-S VQC",
    )
    _add_c1_training_args(d1s_screen, validation=True)
    d1s_screen.set_defaults(maxiter=60)
    d1s_screen.add_argument("--split-seed", type=int, default=None)
    d1s_screen.add_argument("--init-seed", type=int, default=None)
    d1o_screen = sub.add_parser(
        "screen-d1o",
        help="Train/evaluate the two-qubit, exactly-one-upload affine D1-O VQC",
    )
    _add_c1_training_args(d1o_screen, validation=True)
    d1o_screen.set_defaults(maxiter=60)
    d1o_screen.add_argument("--split-seed", type=int, default=None)
    d1o_screen.add_argument("--init-seed", type=int, default=None)
    d1o_final = sub.add_parser(
        "train-final-d1o",
        help="Train D1-O directly on the complete public train set",
    )
    _add_c1_training_args(d1o_final, validation=False)
    d1o_final.set_defaults(maxiter=120, seed=2028)
    d1o_cv = sub.add_parser(
        "cross-validate-d1o",
        help="Generate D1-O quantum-only OOF probabilities and threshold",
    )
    d1o_cv.add_argument("--data-dir", default="raw")
    d1o_cv.add_argument("--artifacts-dir", default="artifacts")
    d1o_cv.add_argument("--split-seed", type=int, default=2026)
    d1o_cv.add_argument("--init-seed", type=int, default=2028)
    d1o_cv.add_argument("--init-scale", type=float, default=0.05)
    d1o_cv.add_argument("--affine-scale-jitter", type=float, default=0.1)
    d1o_cv.add_argument("--maxiter", type=int, default=120)
    d1o_cv.add_argument("--folds", type=int, default=5)
    d1o_cv.add_argument("--shots", type=int, default=1024)
    e1s_screen = sub.add_parser(
        "screen-e1s",
        help="Train/evaluate the C1-based shared dual-axis affine E1-S VQC",
    )
    _add_c1_training_args(e1s_screen, validation=True)
    e1s_screen.set_defaults(maxiter=60)
    e1s_screen.add_argument("--split-seed", type=int, default=None)
    e1s_screen.add_argument("--init-seed", type=int, default=None)
    e1s_cv = sub.add_parser(
        "cross-validate-e1s",
        help="Generate E1-Shared quantum-only OOF probabilities and threshold",
    )
    e1s_cv.add_argument("--data-dir", default="raw")
    e1s_cv.add_argument("--artifacts-dir", default="artifacts")
    e1s_cv.add_argument("--split-seed", type=int, default=2026)
    e1s_cv.add_argument("--init-seed", type=int, default=2026)
    e1s_cv.add_argument("--init-scale", type=float, default=0.05)
    e1s_cv.add_argument("--affine-scale-jitter", type=float, default=0.1)
    e1s_cv.add_argument("--maxiter", type=int, default=120)
    e1s_cv.add_argument("--folds", type=int, default=5)
    e1s_cv.add_argument("--shots", type=int, default=1024)
    c3_screen.set_defaults(maxiter=60)
    f1_screen = sub.add_parser(
        "screen-f1",
        help="Train/evaluate the eight-qubit tree-funnel F1 VQC",
    )
    _add_f1_training_args(f1_screen, validation=True)
    f1_final = sub.add_parser(
        "train-final-f1",
        help="Train F1 directly on the complete public train set",
    )
    _add_f1_training_args(f1_final, validation=False)
    f1_cv = sub.add_parser(
        "cross-validate-f1",
        help="Generate F1 quantum-only OOF probabilities and threshold",
    )
    f1_cv.add_argument("--data-dir", default="raw")
    f1_cv.add_argument("--artifacts-dir", default="artifacts")
    f1_cv.add_argument("--split-seed", type=int, default=2026)
    f1_cv.add_argument("--init-seed", type=int, default=2026)
    f1_cv.add_argument("--blocks", type=int, default=8)
    f1_cv.add_argument("--init-scale", type=float, default=0.05)
    f1_cv.add_argument("--affine-scale-center", type=float, default=1.0)
    f1_cv.add_argument("--affine-scale-jitter", type=float, default=0.1)
    f1_cv.add_argument("--maxiter", type=int, default=200)
    f1_cv.add_argument("--folds", type=int, default=5)
    f1_cv.add_argument("--restarts", type=int, default=1)
    f1_cv.add_argument("--shots", type=int, default=1024)
    search = sub.add_parser(
        "search-features",
        help="Quantum-only forward beam search over packed raw features",
    )
    search.add_argument("--data-dir", default="raw")
    search.add_argument("--artifacts-dir", default="artifacts")
    search.add_argument("--seed", type=int, default=2026)
    search.add_argument("--init-scale", type=float, default=0.05)
    search.add_argument("--maxiter", type=int, default=24)
    search.add_argument("--screen-rows", type=int, default=1200)
    search.add_argument("--validation-fraction", type=float, default=0.25)
    search.add_argument("--beam-width", type=int, default=3)
    search.add_argument("--max-features", type=int, default=5)
    search.add_argument(
        "--feature-layout", choices=tuple(FEATURE_LAYOUTS), default="sequential"
    )
    for command, helptext in (
        ("screen-g1", "Train/evaluate the latent-group G1 VQC"),
        ("cross-validate-g1", "Generate G1 quantum-only OOF probabilities and threshold"),
    ):
        parser_g1 = sub.add_parser(command, help=helptext)
        parser_g1.add_argument("--data-dir", default="raw")
        parser_g1.add_argument("--artifacts-dir", default="artifacts")
        parser_g1.add_argument("--blocks", type=int, default=G1_DEFAULT_BLOCKS)
        parser_g1.add_argument(
            "--include-x1",
            action="store_true",
            help="Give every qubit x1 as well; G1 drops it by default.",
        )
        parser_g1.add_argument("--seed", type=int, default=2026)
        parser_g1.add_argument("--split-seed", type=int, default=None)
        parser_g1.add_argument("--init-scale", type=float, default=1.0)
        parser_g1.add_argument("--affine-scale-center", type=float, default=1.0)
        parser_g1.add_argument("--affine-scale-jitter", type=float, default=0.3)
        parser_g1.add_argument("--maxiter", type=int, default=200)
        parser_g1.add_argument("--restarts", type=int, default=4)
        parser_g1.add_argument("--folds", type=int, default=5)
        parser_g1.add_argument("--shots", type=int, default=1024)
        parser_g1.add_argument("--threshold", type=float, default=0.5)
        parser_g1.add_argument("--validation-fraction", type=float, default=0.2)
        parser_g1.add_argument(
            "--objective",
            choices=("balanced_bce", "soft_balanced_accuracy", "smooth_auc", "smooth_ks"),
            default="balanced_bce",
        )
        parser_g1.add_argument("--temperature-start", type=float, default=0.30)
        parser_g1.add_argument("--temperature-stop", type=float, default=0.015)
        parser_g1.add_argument("--anneal-stages", type=int, default=10)
        parser_g1.add_argument("--stage-maxiter", type=int, default=40)
        parser_g1.add_argument("--max-rows", type=int, default=None)
    candidate = sub.add_parser(
        "cross-validate-candidate",
        help="Quantum-only OOF for a named candidate architecture",
    )
    candidate.add_argument("--candidate", required=True, choices=sorted(CANDIDATES))
    candidate.add_argument(
        "--screen",
        action="store_true",
        help="Single holdout instead of five folds; a first-stage filter.",
    )
    candidate.add_argument("--validation-fraction", type=float, default=0.2)
    candidate.add_argument("--data-dir", default="raw")
    candidate.add_argument("--artifacts-dir", default="artifacts")
    candidate.add_argument("--seed", type=int, default=2026)
    candidate.add_argument("--split-seed", type=int, default=2026)
    candidate.add_argument("--init-scale", type=float, default=1.0)
    candidate.add_argument("--affine-scale-center", type=float, default=1.0)
    candidate.add_argument("--affine-scale-jitter", type=float, default=0.5)
    candidate.add_argument("--maxiter", type=int, default=150)
    candidate.add_argument("--restarts", type=int, default=1)
    candidate.add_argument("--folds", type=int, default=5)
    candidate.add_argument("--shots", type=int, default=1024)
    candidate.add_argument("--threshold", type=float, default=0.5)
    candidate.add_argument(
        "--objective",
        choices=("balanced_bce", "soft_balanced_accuracy", "smooth_auc", "smooth_ks"),
        default="smooth_auc",
    )
    candidate.add_argument("--temperature-start", type=float, default=0.30)
    candidate.add_argument("--temperature-stop", type=float, default=0.015)
    candidate.add_argument("--anneal-stages", type=int, default=10)
    candidate.add_argument("--stage-maxiter", type=int, default=30)
    report = sub.add_parser(
        "report-test",
        help="Reporting-only evaluation of an artifact on a labelled CSV",
    )
    report.add_argument("--artifact-dir", required=True)
    report.add_argument("--data-dir", default="raw")
    report.add_argument("--csv-name", default="public_test.csv")
    report.add_argument("--shots", type=int, default=1024)
    report.add_argument("--threshold", type=float, default=None)
    source = sub.add_parser("audit-source", help="Reject known classical predictive-model code")
    source.add_argument("--source-root", default="src")
    artifact = sub.add_parser("audit-artifact", help="Audit a completed final artifact")
    artifact.add_argument("--artifact-dir", required=True)
    package = sub.add_parser("package", help="Copy an audited artifact into submit/<name>")
    package.add_argument("--artifact-dir", required=True)
    package.add_argument("--name", required=True)
    package.add_argument("--submit-dir", default="submit")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "screen":
        print(run_screen(_train_config(args)))
        return
    if args.command == "train-final":
        print(run_final(_train_config(args)))
        return
    if args.command == "screen-c1":
        print(run_c1_screen(_c1_config(args, validation=True)))
        return
    if args.command == "train-final-c1":
        print(run_c1_final(_c1_config(args, validation=False)))
        return
    if args.command == "cross-validate-c1":
        print(
            run_c1_cross_validation(
                C1CrossValidationConfig(
                    train_csv=Path(args.data_dir) / "public_train.csv",
                    artifacts_dir=Path(args.artifacts_dir),
                    split_seed=args.split_seed,
                    init_seed=args.init_seed,
                    init_scale=args.init_scale,
                    affine_scale_jitter=args.affine_scale_jitter,
                    maxiter=args.maxiter,
                    folds=args.folds,
                    n_restarts=args.restarts,
                    shots=args.shots,
                    objective=args.objective,
                    temperature_start=args.temperature_start,
                    temperature_stop=args.temperature_stop,
                    anneal_stages=args.anneal_stages,
                    stage_maxiter=args.stage_maxiter,
                )
            )
        )
        return
    if args.command == "screen-c1o":
        print(run_c1o_screen(_c1o_config(args, validation=True)))
        return
    if args.command == "screen-c1x":
        print(run_c1x_screen(_c1x_config(args, validation=True)))
        return
    if args.command == "screen-c2":
        print(run_c2_screen(_c2_config(args, validation=True)))
        return
    if args.command == "train-final-c2":
        print(run_c2_final(_c2_config(args, validation=False)))
        return
    if args.command == "cross-validate-c2":
        print(
            run_c2_cross_validation(
                C2CrossValidationConfig(
                    train_csv=Path(args.data_dir) / "public_train.csv",
                    artifacts_dir=Path(args.artifacts_dir),
                    split_seed=args.split_seed,
                    init_seed=args.init_seed,
                    init_scale=args.init_scale,
                    affine_scale_jitter=args.affine_scale_jitter,
                    maxiter=args.maxiter,
                    folds=args.folds,
                    shots=args.shots,
                )
            )
        )
        return
    if args.command == "screen-c3":
        print(run_c3_screen(_c3_config(args, validation=True)))
        return
    if args.command == "screen-d1":
        print(run_d1_screen(_d1_config(args, validation=True)))
        return
    if args.command == "screen-d1s":
        print(run_d1s_screen(_d1s_config(args, validation=True)))
        return
    if args.command == "screen-d1o":
        print(run_d1o_screen(_d1o_config(args, validation=True)))
        return
    if args.command == "cross-validate-d1o":
        print(
            run_d1o_cross_validation(
                D1OCrossValidationConfig(
                    train_csv=Path(args.data_dir) / "public_train.csv",
                    artifacts_dir=Path(args.artifacts_dir),
                    split_seed=args.split_seed,
                    init_seed=args.init_seed,
                    init_scale=args.init_scale,
                    affine_scale_jitter=args.affine_scale_jitter,
                    maxiter=args.maxiter,
                    folds=args.folds,
                    shots=args.shots,
                )
            )
        )
        return
    if args.command == "train-final-d1o":
        print(run_d1o_final(_d1o_config(args, validation=False)))
        return
    if args.command == "screen-e1s":
        print(run_e1s_screen(_e1s_config(args, validation=True)))
        return
    if args.command == "cross-validate-e1s":
        print(
            run_e1s_cross_validation(
                E1SCrossValidationConfig(
                    train_csv=Path(args.data_dir) / "public_train.csv",
                    artifacts_dir=Path(args.artifacts_dir),
                    split_seed=args.split_seed,
                    init_seed=args.init_seed,
                    init_scale=args.init_scale,
                    affine_scale_jitter=args.affine_scale_jitter,
                    maxiter=args.maxiter,
                    folds=args.folds,
                    shots=args.shots,
                )
            )
        )
        return
    if args.command == "screen-f1":
        print(run_f1_screen(_f1_config(args, validation=True)))
        return
    if args.command == "train-final-f1":
        print(run_f1_final(_f1_config(args, validation=False)))
        return
    if args.command == "cross-validate-f1":
        print(
            run_f1_cross_validation(
                F1CrossValidationConfig(
                    train_csv=Path(args.data_dir) / "public_train.csv",
                    artifacts_dir=Path(args.artifacts_dir),
                    split_seed=args.split_seed,
                    init_seed=args.init_seed,
                    n_blocks=args.blocks,
                    init_scale=args.init_scale,
                    affine_scale_center=args.affine_scale_center,
                    affine_scale_jitter=args.affine_scale_jitter,
                    maxiter=args.maxiter,
                    folds=args.folds,
                    n_restarts=args.restarts,
                    shots=args.shots,
                )
            )
        )
        return
    if args.command == "search-features":
        print(
            run_feature_search(
                FeatureSearchConfig(
                    train_csv=Path(args.data_dir) / "public_train.csv",
                    artifacts_dir=Path(args.artifacts_dir),
                    seed=args.seed,
                    init_scale=args.init_scale,
                    maxiter=args.maxiter,
                    screen_rows=args.screen_rows,
                    validation_fraction=args.validation_fraction,
                    beam_width=args.beam_width,
                    max_features=args.max_features,
                    feature_layout=args.feature_layout,
                )
            )
        )
        return
    if args.command in ("screen-g1", "cross-validate-g1"):
        spec = build_g1_spec(args.blocks, args.include_x1)
        config = SpecTrainConfig(
            train_csv=Path(args.data_dir) / "public_train.csv",
            artifacts_dir=Path(args.artifacts_dir),
            label=f"g1b{args.blocks}" + ("x1" if args.include_x1 else ""),
            seed=args.seed,
            split_seed=args.split_seed,
            init_scale=args.init_scale,
            affine_scale_center=args.affine_scale_center,
            affine_scale_jitter=args.affine_scale_jitter,
            maxiter=args.maxiter,
            n_restarts=args.restarts,
            folds=args.folds,
            shots=args.shots,
            validation_fraction=args.validation_fraction,
            max_rows=args.max_rows,
            decision_threshold=args.threshold,
            objective=args.objective,
            temperature_start=args.temperature_start,
            temperature_stop=args.temperature_stop,
            anneal_stages=args.anneal_stages,
            stage_maxiter=args.stage_maxiter,
        )
        runner = run_spec_screen if args.command == "screen-g1" else run_spec_cross_validation
        print(runner(spec, config))
        return
    if args.command == "cross-validate-candidate":
        runner = run_spec_screen if args.screen else run_spec_cross_validation
        print(
            runner(
                build_candidate(args.candidate),
                SpecTrainConfig(
                    train_csv=Path(args.data_dir) / "public_train.csv",
                    artifacts_dir=Path(args.artifacts_dir),
                    label=args.candidate,
                    seed=args.seed,
                    split_seed=args.split_seed,
                    init_scale=args.init_scale,
                    affine_scale_center=args.affine_scale_center,
                    affine_scale_jitter=args.affine_scale_jitter,
                    maxiter=args.maxiter,
                    n_restarts=args.restarts,
                    folds=args.folds,
                    shots=args.shots,
                    decision_threshold=args.threshold,
                    objective=args.objective,
                    temperature_start=args.temperature_start,
                    temperature_stop=args.temperature_stop,
                    anneal_stages=args.anneal_stages,
                    stage_maxiter=args.stage_maxiter,
                    validation_fraction=args.validation_fraction,
                ),
            )
        )
        return
    if args.command == "report-test":
        print(
            json.dumps(
                evaluate_artifact_on_csv(
                    Path(args.artifact_dir),
                    Path(args.data_dir) / args.csv_name,
                    shots=args.shots,
                    threshold=args.threshold,
                ),
                indent=2,
            )
        )
        return
    if args.command == "audit-source":
        report = audit_source(Path(args.source_root))
    else:
        source = Path(args.artifact_dir)
        report = audit_artifact(source)
        if args.command == "package" and report["passes"]:
            destination = Path(args.submit_dir) / args.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing package: {destination}")
            destination.mkdir(parents=True)
            for name in ("classifier.qasm", "weights.json", "encoding_note.txt", "parameter_provenance.json"):
                shutil.copy2(source / name, destination / name)
            report["packaged_to"] = str(destination)
    print(json.dumps(report, indent=2))
    if not report["passes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
