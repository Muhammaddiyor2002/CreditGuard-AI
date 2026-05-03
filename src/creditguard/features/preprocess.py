"""Preprocessing pipeline.

Builds a single ``sklearn`` ColumnTransformer that:
  * imputes missing values (median for numerics, mode for categoricals);
  * winsorizes numeric outliers via the 1st/99th percentile;
  * one-hot encodes categoricals with a stable category set;
  * (optionally) scales numerics for linear models.

The pipeline is deterministic, picklable, and used identically for training,
evaluation, batch prediction, and single-applicant inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_LEVELS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from ..utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------

class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric values to learned ``[lower, upper]`` percentile bounds."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray | pd.DataFrame, y=None) -> Winsorizer:  # noqa: N803
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower_quantile, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper_quantile, axis=0)
        # Safety: any all-NaN column gets a fallback range.
        self.lower_ = np.where(np.isnan(self.lower_), 0.0, self.lower_)
        self.upper_ = np.where(np.isnan(self.upper_), 1.0, self.upper_)
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:  # noqa: N803
        arr = np.asarray(X, dtype=float).copy()
        return np.clip(arr, self.lower_, self.upper_)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingArtifacts:
    """Bundle returned by :func:`build_preprocessor`."""

    pipeline: ColumnTransformer
    feature_names: list[str]


def build_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Construct the ColumnTransformer used by every model."""

    numeric_steps: list[tuple[str, BaseEstimator]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("winsorize", Winsorizer(0.01, 0.99)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categories = [CATEGORICAL_LEVELS[c] for c in CATEGORICAL_FEATURES]
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=categories,
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def expand_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the post-encoding feature names produced by the preprocessor."""

    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:  # noqa: BLE001
        names: list[str] = list(NUMERIC_FEATURES)
        for col, levels in CATEGORICAL_LEVELS.items():
            names.extend(f"{col}_{lvl}" for lvl in levels)
        return names


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate schema and return ``(X, y)`` from a labeled DataFrame."""

    missing = [c for c in (*ALL_FEATURES, TARGET_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    return df[ALL_FEATURES].copy(), df[TARGET_COLUMN].astype(int).copy()


__all__ = [
    "Winsorizer",
    "PreprocessingArtifacts",
    "build_preprocessor",
    "expand_feature_names",
    "split_features_target",
]
