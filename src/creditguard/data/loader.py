"""Dataset loading — German Credit (OpenML) + synthetic augmentation.

The German Credit (statlog) dataset has only 1000 rows, which is too small to
train robust gradient-boosted models. We therefore:

  1. Pull the raw OpenML data (cached on disk).
  2. Translate the German-Credit columns into the CreditGuard 13-feature schema.
  3. Augment with synthetic rows generated from the same latent-risk model so
     the final training set is ~10k rows with realistic defaults distribution.

If the OpenML download fails (no internet), we transparently fall back to a
fully-synthetic dataset so the pipeline still runs end-to-end offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    ALL_FEATURES,
    CATEGORICAL_LEVELS,
    DATASET_CONFIG,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)
from ..utils.io import load_dataframe, save_dataframe
from ..utils.logging import get_logger
from .synthetic import generate_synthetic_dataset

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# OpenML → CreditGuard schema mapping
# ---------------------------------------------------------------------------

_PURPOSE_MAP = {
    "radio/tv": "vacation",
    "education": "education",
    "furniture/equipment": "personal",
    "new car": "car",
    "used car": "car",
    "business": "business",
    "domestic appliance": "personal",
    "repairs": "personal",
    "other": "personal",
    "retraining": "education",
}

_EDUCATION_FROM_JOB = {
    "unemp/unskilled non res": "none",
    "unskilled resident": "high_school",
    "skilled": "bachelor",
    "high qualif/self emp/mgmt": "master",
}

_MARITAL_MAP = {
    # Note: original 'personal_status' encodes both gender + marital status.
    "male single": "single",
    "male div/sep": "divorced",
    "male mar/wid": "married",
    "female div/dep/mar": "married",
}


def _to_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _german_credit_to_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Project the OpenML German Credit dataset onto the CreditGuard schema."""

    df = raw.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Target — class is "good"/"bad"; map "bad" → 1 (defaulted).
    if "class" in df.columns:
        df[TARGET_COLUMN] = (df["class"].astype(str).str.lower() == "bad").astype(int)
    else:
        raise ValueError("Expected `class` column in German Credit dataset.")

    out = pd.DataFrame()

    # Numeric features.
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    out["income"] = pd.to_numeric(df.get("credit_amount"), errors="coerce") * 12 / 4
    out["employment_years"] = (
        df.get("employment", pd.Series(dtype=str))
        .astype(str)
        .map(
            {
                "unemployed": 0.0,
                "<1": 0.5,
                "1<=X<4": 2.5,
                "4<=X<7": 5.5,
                ">=7": 9.0,
            }
        )
    )
    out["loan_amount"] = pd.to_numeric(df.get("credit_amount"), errors="coerce")
    out["loan_term"] = pd.to_numeric(df.get("duration"), errors="coerce")
    # Construct a 300-850 credit score from existing_credits + history quality.
    history_score = (
        df.get("credit_history", pd.Series(dtype=str))
        .astype(str)
        .map(
            {
                "no credits/all paid": 800,
                "all paid": 750,
                "existing paid": 700,
                "delayed previously": 600,
                "critical/other existing credit": 500,
            }
        )
    )
    out["credit_score"] = history_score.fillna(650).astype(int)
    out["debt_to_income_ratio"] = (
        out["loan_amount"] / (out["income"].replace(0, np.nan))
    ).clip(0, 1.5)
    out["previous_defaults"] = (
        df.get("credit_history", pd.Series(dtype=str))
        .astype(str)
        .map(
            {
                "no credits/all paid": 0,
                "all paid": 0,
                "existing paid": 0,
                "delayed previously": 1,
                "critical/other existing credit": 2,
            }
        )
        .fillna(0)
        .astype(int)
    )
    out["dependents"] = pd.to_numeric(
        df.get("num_dependents", df.get("num_people_being_liable")), errors="coerce"
    ).fillna(1).astype(int)
    out["savings_balance"] = (
        df.get("savings_status", pd.Series(dtype=str))
        .astype(str)
        .map(
            {
                "no known savings": 0.0,
                "<100": 50.0,
                "100<=X<500": 250.0,
                "500<=X<1000": 750.0,
                ">=1000": 2500.0,
            }
        )
        .fillna(0.0)
    )

    # Categorical features.
    out["marital_status"] = (
        df.get("personal_status", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .map(_MARITAL_MAP)
        .fillna("single")
    )
    out["education"] = (
        df.get("job", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .map(_EDUCATION_FROM_JOB)
        .fillna("high_school")
    )
    out["loan_purpose"] = (
        df.get("purpose", pd.Series(dtype=str))
        .apply(_to_str)
        .map(_PURPOSE_MAP)
        .fillna("personal")
    )

    out[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    out = out[[*ALL_FEATURES, TARGET_COLUMN]]
    log.info("German Credit projected to schema: %s, default_rate=%.3f",
             out.shape, out[TARGET_COLUMN].mean())
    return out


def _enforce_categorical_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Replace any out-of-vocabulary categorical with the most common level."""

    df = df.copy()
    for col, levels in CATEGORICAL_LEVELS.items():
        if col not in df.columns:
            continue
        mask = ~df[col].isin(levels)
        if mask.any():
            df.loc[mask, col] = levels[0]
    return df


def _load_openml_german_credit(cache_path: Path) -> pd.DataFrame | None:
    """Download the OpenML German Credit dataset, with on-disk caching."""

    if cache_path.exists():
        log.info("Loading cached German Credit data from %s", cache_path)
        try:
            return load_dataframe(cache_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Cache read failed (%s); re-downloading.", exc)

    try:
        import openml  # local import — heavy dependency
    except Exception as exc:  # noqa: BLE001
        log.warning("openml not importable (%s); skipping OpenML fetch.", exc)
        return None

    try:
        log.info("Downloading German Credit dataset from OpenML (id=%d)…",
                 DATASET_CONFIG.openml_dataset_id)
        ds = openml.datasets.get_dataset(
            DATASET_CONFIG.openml_dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        x, y, _, attribute_names = ds.get_data(
            dataset_format="dataframe",
            target=ds.default_target_attribute,
        )
        df = x.copy()
        df.columns = attribute_names
        df["class"] = y.astype(str)
        df = _german_credit_to_schema(df)
        save_dataframe(df, cache_path)
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("OpenML download failed (%s); falling back to synthetic only.", exc)
        return None


def load_credit_dataset(
    *,
    use_openml: bool = True,
    synthetic_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load the German Credit dataset and augment it with synthetic rows.

    Args:
        use_openml: If True, attempt to fetch the German Credit dataset from
            OpenML. If the download fails (offline), the loader transparently
            falls back to a fully-synthetic dataset.
        synthetic_rows: Number of synthetic rows to mix in. Defaults to
            :data:`DATASET_CONFIG.synthetic_rows`.
        seed: Random seed for the synthetic component.

    Returns:
        Combined DataFrame with the 13-feature schema and ``defaulted`` target.
    """

    synthetic_rows = synthetic_rows if synthetic_rows is not None else DATASET_CONFIG.synthetic_rows
    cache_path = RAW_DATA_DIR / "german_credit.parquet"

    frames: list[pd.DataFrame] = []
    if use_openml:
        german = _load_openml_german_credit(cache_path)
        if german is not None and not german.empty:
            frames.append(german)

    if synthetic_rows > 0:
        frames.append(generate_synthetic_dataset(n_rows=synthetic_rows, seed=seed))

    if not frames:
        raise RuntimeError("No data sources available — both OpenML and synthetic disabled.")

    df = pd.concat(frames, ignore_index=True)
    df = _enforce_categorical_domain(df)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    raw_path = RAW_DATA_DIR / DATASET_CONFIG.raw_filename
    save_dataframe(df, raw_path)
    log.info(
        "Final dataset assembled: shape=%s, default_rate=%.3f, saved to %s",
        df.shape,
        df[TARGET_COLUMN].mean(),
        raw_path,
    )
    return df


__all__ = ["load_credit_dataset"]
