"""Evaluation utilities — metrics + Plotly visualizations.

The evaluation step takes a fitted :class:`TrainedModel` and a held-out test
set, then populates ``test_metrics``, ``test_curve``, and ``confusion_matrix``
on the artifact so the Streamlit UI can render charts without re-computing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..config import APP_CONFIG
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Bundle of metrics + curves produced by :func:`evaluate_model`."""

    metrics: dict[str, float]
    confusion: list[list[int]]
    roc_curve: dict[str, list[float]]
    pr_curve: dict[str, list[float]]
    threshold_table: pd.DataFrame
    classification_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "confusion_matrix": self.confusion,
            "roc_curve": self.roc_curve,
            "pr_curve": self.pr_curve,
            "threshold_table": self.threshold_table.to_dict(orient="records"),
            "classification_threshold": self.classification_threshold,
        }


def _threshold_table(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """Return a small grid of metrics across decision thresholds."""

    rows: list[dict[str, float]] = []
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (y_proba >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
) -> EvaluationReport:
    """Compute the full evaluation report for held-out predictions."""

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    return EvaluationReport(
        metrics=metrics,
        confusion=[list(map(int, row)) for row in cm],
        roc_curve={"fpr": list(map(float, fpr)), "tpr": list(map(float, tpr))},
        pr_curve={
            "precision": list(map(float, precision)),
            "recall": list(map(float, recall)),
        },
        threshold_table=_threshold_table(y_true, y_proba),
        classification_threshold=float(threshold),
    )


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def plot_roc_curve(report: EvaluationReport) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=report.roc_curve["fpr"],
            y=report.roc_curve["tpr"],
            mode="lines",
            name=f"ROC (AUC={report.metrics['roc_auc']:.3f})",
            line=dict(color=APP_CONFIG.primary_color, width=3),
            fill="tozeroy",
            fillcolor="rgba(30, 136, 229, 0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#999", dash="dash"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_pr_curve(report: EvaluationReport) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=report.pr_curve["recall"],
            y=report.pr_curve["precision"],
            mode="lines",
            name=f"PR (AUC={report.metrics['pr_auc']:.3f})",
            line=dict(color=APP_CONFIG.accent_color, width=3),
            fill="tozeroy",
            fillcolor="rgba(67, 160, 71, 0.12)",
        )
    )
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_confusion_matrix(report: EvaluationReport) -> go.Figure:
    cm = np.array(report.confusion)
    labels = ["Non-default", "Default"]
    text = [[f"{cm[i, j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"},
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_threshold_curves(report: EvaluationReport) -> go.Figure:
    df = report.threshold_table
    fig = go.Figure()
    for col, color in [
        ("precision", APP_CONFIG.primary_color),
        ("recall", APP_CONFIG.accent_color),
        ("f1", APP_CONFIG.warning_color),
        ("accuracy", "#6A1B9A"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["threshold"],
                y=df[col],
                mode="lines+markers",
                name=col.title(),
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title="Metrics vs. Decision Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


__all__ = [
    "EvaluationReport",
    "evaluate_model",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_confusion_matrix",
    "plot_threshold_curves",
]
