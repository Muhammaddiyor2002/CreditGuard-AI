"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from creditguard.config import ALL_FEATURES, NUMERIC_FEATURES
from creditguard.data.synthetic import generate_synthetic_dataset
from creditguard.features.preprocess import (
    Winsorizer,
    build_preprocessor,
    expand_feature_names,
)


def test_winsorizer_clips_outliers():
    arr = np.array([[10.0], [20.0], [30.0], [40.0], [1000.0]])
    w = Winsorizer(lower_quantile=0.0, upper_quantile=0.8).fit(arr)
    out = w.transform(arr)
    assert out.max() < 1000.0
    assert out.min() >= 10.0


def test_preprocessor_fit_transform_shape():
    df = generate_synthetic_dataset(n_rows=500, missing_rate=0.05, seed=0)
    pre = build_preprocessor(scale_numeric=True)
    x_t = pre.fit_transform(df[ALL_FEATURES])
    assert x_t.shape[0] == 500
    assert x_t.shape[1] >= len(NUMERIC_FEATURES)
    assert not np.isnan(x_t).any()


def test_expand_feature_names_includes_categories():
    df = generate_synthetic_dataset(n_rows=300, missing_rate=0.0, seed=1)
    pre = build_preprocessor()
    pre.fit(df[ALL_FEATURES])
    names = expand_feature_names(pre)
    assert any("loan_purpose" in n for n in names)
    assert any("education" in n for n in names)
