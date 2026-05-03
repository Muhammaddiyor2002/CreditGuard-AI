"""Tests for the synthetic dataset generator."""

from __future__ import annotations

import pandas as pd

from creditguard.config import ALL_FEATURES, TARGET_COLUMN
from creditguard.data.synthetic import generate_synthetic_dataset


def test_synthetic_schema():
    df = generate_synthetic_dataset(n_rows=500, missing_rate=0.0, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    for col in ALL_FEATURES + [TARGET_COLUMN]:
        assert col in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_synthetic_default_rate_within_bounds():
    df = generate_synthetic_dataset(n_rows=2000, missing_rate=0.0, seed=42)
    rate = df[TARGET_COLUMN].mean()
    assert 0.10 < rate < 0.55, f"Default rate {rate:.3f} is unrealistic."


def test_synthetic_reproducible():
    df1 = generate_synthetic_dataset(n_rows=300, missing_rate=0.0, seed=7)
    df2 = generate_synthetic_dataset(n_rows=300, missing_rate=0.0, seed=7)
    pd.testing.assert_frame_equal(df1, df2)
