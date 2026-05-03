"""Tests for the evaluation module."""

from __future__ import annotations

import numpy as np

from creditguard.evaluation import evaluate_model


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.05, 0.10, 0.92, 0.97])
    report = evaluate_model(y, proba, threshold=0.5)
    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["precision"] == 1.0
    assert report.metrics["recall"] == 1.0
    assert report.metrics["roc_auc"] == 1.0
    assert report.confusion == [[2, 0], [0, 2]]


def test_metrics_random_classifier_better_than_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    proba = rng.random(size=200)
    report = evaluate_model(y, proba)
    assert 0.0 <= report.metrics["roc_auc"] <= 1.0
    assert len(report.threshold_table) > 0
