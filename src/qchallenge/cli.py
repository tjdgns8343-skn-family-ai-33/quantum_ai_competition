"""Command-line interface for screening, final training, auditing and packaging."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .compliance import audit_artifact, audit_source
from .training import TrainConfig, run_final, run_screen


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        train_csv=Path(args.data_dir) / "public_train.csv",
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed,
        init_scale=args.init_scale,
        maxiter=args.maxiter,
        shots=args.shots,
        validation_fraction=getattr(args, "validation_fraction", 0.2),
    )


def _add_training_args(parser: argparse.ArgumentParser, *, validation: bool) -> None:
    parser.add_argument("--data-dir", default="raw")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init-scale", type=float, default=0.05)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--shots", type=int, default=1024)
    if validation:
        parser.add_argument("--validation-fraction", type=float, default=0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("screen", help="Train/evaluate on a train-only holdout; creates no submission")
    _add_training_args(screen, validation=True)
    final = sub.add_parser("train-final", help="Train directly on the full public train set")
    _add_training_args(final, validation=False)
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

