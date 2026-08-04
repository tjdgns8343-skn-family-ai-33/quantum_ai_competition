"""Strict raw CSV loading without feature transformation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = tuple(f"x{i}" for i in range(1, 9))
LABEL_COLUMN = "label"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_train(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the eight raw columns exactly as stored; reject missing/non-finite data."""
    frame = pd.read_csv(path)
    expected = [*FEATURE_COLUMNS, LABEL_COLUMN]
    if list(frame.columns) != expected:
        raise ValueError(f"Expected columns {expected}, got {list(frame.columns)}")
    x = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float, copy=True)
    y = frame.loc[:, LABEL_COLUMN].to_numpy(dtype=int, copy=True)
    if x.ndim != 2 or x.shape[1] != 8 or len(x) == 0:
        raise ValueError(f"Expected a non-empty (n, 8) feature matrix, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Missing or non-finite raw features are not accepted.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("label must contain both binary classes 0 and 1.")
    return x, y, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": int(len(x)),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_processing": "none",
    }

