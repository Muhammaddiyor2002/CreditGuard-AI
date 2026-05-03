"""I/O helpers — joblib persistence, parquet round-tripping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .logging import get_logger

log = get_logger(__name__)


def save_joblib(obj: Any, path: str | Path) -> Path:
    """Persist a Python object with joblib (compressed)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)
    log.info("Saved joblib artifact to %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def load_joblib(path: str | Path) -> Any:
    """Load a joblib-pickled object."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist a DataFrame as parquet, falling back to CSV if engines missing."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        csv_path = path.with_suffix(".csv")
        log.warning("Parquet write failed (%s); falling back to %s", exc, csv_path)
        df.to_csv(csv_path, index=False)
        return csv_path
    return path


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a parquet/csv into a DataFrame."""

    path = Path(path)
    if not path.exists():
        # Try sibling extensions.
        for alt in (path.with_suffix(".parquet"), path.with_suffix(".csv")):
            if alt.exists():
                path = alt
                break
        else:  # noqa: PLW0120
            raise FileNotFoundError(f"DataFrame not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


__all__ = ["save_joblib", "load_joblib", "save_dataframe", "load_dataframe"]
