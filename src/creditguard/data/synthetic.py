"""Reproducible synthetic credit-risk dataset generator.

The synthetic data is generated from a transparent latent-risk model:

    risk = β·standardized(features) + ε
    p_default = sigmoid(risk)
    defaulted ~ Bernoulli(p_default)

Coefficients are chosen so that the resulting feature/target relationships
match domain intuition (e.g. higher credit_score reduces default probability,
more previous_defaults increases it). This gives realistic SHAP plots and
sensible decision boundaries while remaining 100% reproducible offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    ALL_FEATURES,
    CATEGORICAL_LEVELS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from ..utils.logging import get_logger

log = get_logger(__name__)


# Latent-risk linear coefficients (sign indicates direction of effect on default).
# These are intentionally interpretable and align with common credit scorecards.
_NUMERIC_COEFFS: dict[str, float] = {
    "age": -0.20,
    "income": -0.55,
    "employment_years": -0.35,
    "loan_amount": 0.45,
    "loan_term": 0.20,
    "credit_score": -0.95,
    "debt_to_income_ratio": 0.85,
    "previous_defaults": 0.95,
    "dependents": 0.18,
    "savings_balance": -0.45,
}

_CATEGORICAL_COEFFS: dict[tuple[str, str], float] = {
    ("marital_status", "single"): 0.10,
    ("marital_status", "married"): -0.10,
    ("marital_status", "divorced"): 0.20,
    ("marital_status", "widowed"): 0.05,
    ("education", "none"): 0.30,
    ("education", "high_school"): 0.10,
    ("education", "bachelor"): -0.10,
    ("education", "master"): -0.20,
    ("education", "phd"): -0.30,
    ("loan_purpose", "education"): 0.05,
    ("loan_purpose", "home"): -0.10,
    ("loan_purpose", "car"): 0.00,
    ("loan_purpose", "business"): 0.20,
    ("loan_purpose", "medical"): 0.15,
    ("loan_purpose", "personal"): 0.10,
    ("loan_purpose", "vacation"): 0.30,
    ("loan_purpose", "debt_consolidation"): 0.40,
}

_INTERCEPT: float = -0.4


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_numeric(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Draw realistic numeric features with sensible joint structure."""

    age = np.clip(rng.normal(38, 11, size=n), 18, 80).round().astype(int)
    employment_years = np.clip(
        rng.normal(np.maximum(age - 22, 0) * 0.55, 4.5, size=n),
        0,
        np.maximum(age - 18, 0),
    ).round().astype(int)
    income = np.clip(
        rng.lognormal(mean=10.6, sigma=0.55, size=n) + employment_years * 500,
        9_000,
        500_000,
    ).round(2)
    loan_amount = np.clip(
        rng.lognormal(mean=9.4, sigma=0.6, size=n),
        500,
        300_000,
    ).round(2)
    loan_term = rng.choice([12, 24, 36, 48, 60, 72, 84, 96, 120], size=n)
    credit_score = np.clip(rng.normal(680, 70, size=n), 300, 850).round().astype(int)
    debt_to_income_ratio = np.clip(rng.beta(2.2, 5.5, size=n) * 1.1, 0.0, 1.5).round(3)
    previous_defaults = rng.poisson(lam=0.35, size=n)
    dependents = rng.poisson(lam=1.1, size=n)
    savings_balance = np.clip(
        rng.lognormal(mean=8.2, sigma=1.1, size=n),
        0,
        500_000,
    ).round(2)

    return {
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "previous_defaults": previous_defaults,
        "dependents": dependents,
        "savings_balance": savings_balance,
    }


def _sample_categorical(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Draw categorical features with realistic mass distributions."""

    return {
        "marital_status": rng.choice(
            CATEGORICAL_LEVELS["marital_status"],
            size=n,
            p=[0.35, 0.50, 0.10, 0.05],
        ),
        "education": rng.choice(
            CATEGORICAL_LEVELS["education"],
            size=n,
            p=[0.05, 0.35, 0.40, 0.15, 0.05],
        ),
        "loan_purpose": rng.choice(
            CATEGORICAL_LEVELS["loan_purpose"],
            size=n,
            p=[0.10, 0.15, 0.20, 0.10, 0.10, 0.15, 0.05, 0.15],
        ),
    }


def _compute_default_probability(df: pd.DataFrame) -> np.ndarray:
    """Apply the latent-risk linear model to compute per-row default probability."""

    numeric_matrix = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    means = numeric_matrix.mean(axis=0)
    stds = numeric_matrix.std(axis=0) + 1e-9
    standardized = (numeric_matrix - means) / stds

    coeffs = np.array([_NUMERIC_COEFFS[c] for c in NUMERIC_FEATURES])
    risk = standardized @ coeffs

    for col, levels in CATEGORICAL_LEVELS.items():
        for lvl in levels:
            mask = (df[col] == lvl).to_numpy()
            risk = risk + mask * _CATEGORICAL_COEFFS[(col, lvl)]

    return _sigmoid(risk + _INTERCEPT)


def generate_synthetic_dataset(
    n_rows: int = 9000,
    *,
    missing_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic credit-risk dataset with realistic risk structure.

    Args:
        n_rows: Number of synthetic applicants to generate.
        missing_rate: Fraction of cells (per non-target column) to set to NaN
            so the preprocessing pipeline is exercised on realistic data.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with all 13 features + ``defaulted`` target column.
    """

    rng = np.random.default_rng(seed)
    log.info("Generating %d synthetic credit-risk rows (seed=%d)", n_rows, seed)

    data = {**_sample_numeric(rng, n_rows), **_sample_categorical(rng, n_rows)}
    df = pd.DataFrame(data, columns=ALL_FEATURES)

    probabilities = _compute_default_probability(df)
    df[TARGET_COLUMN] = (rng.random(size=n_rows) < probabilities).astype(int)

    if missing_rate > 0:
        feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
        n_cells = int(missing_rate * len(df) * len(feature_cols))
        rows = rng.integers(0, len(df), size=n_cells)
        cols = rng.integers(0, len(feature_cols), size=n_cells)
        for r, c in zip(rows, cols, strict=False):
            df.iat[r, df.columns.get_loc(feature_cols[c])] = np.nan

    default_rate = df[TARGET_COLUMN].mean()
    log.info("Synthetic dataset built: shape=%s, default_rate=%.3f", df.shape, default_rate)
    return df


__all__ = ["generate_synthetic_dataset"]
