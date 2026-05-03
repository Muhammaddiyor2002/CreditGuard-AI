"""SHAP-based explainability — global and per-applicant.

We pick the right :class:`shap.Explainer` based on the underlying classifier:

  * ``XGBClassifier`` / ``LGBMClassifier`` / ``RandomForestClassifier`` →
    :class:`shap.TreeExplainer` (fast, exact for tree models).
  * ``LogisticRegression`` →
    :class:`shap.LinearExplainer`.
  * Anything else falls back to :class:`shap.Explainer` with a small KMeans
    background sample (slower but always works).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..config import ALL_FEATURES, APP_CONFIG
from ..features.preprocess import expand_feature_names
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ShapPayload:
    """Container of pre-computed SHAP values + the matrix they apply to."""

    values: np.ndarray
    base_value: float
    feature_names: list[str]
    X_transformed: np.ndarray  # noqa: N815
    X_raw: pd.DataFrame  # noqa: N815


def _classifier_kind(estimator: Any) -> str:
    name = type(estimator).__name__
    if name == "XGBClassifier":
        return "xgboost"
    if name == "LGBMClassifier":
        return "lightgbm"
    if isinstance(estimator, RandomForestClassifier):
        return "random_forest"
    if isinstance(estimator, LogisticRegression):
        return "logistic_regression"
    return "generic"


def _pick_explainer(
    classifier: Any, x_background: np.ndarray, feature_names: list[str]
) -> shap.Explainer:
    kind = _classifier_kind(classifier)
    if kind in {"xgboost", "lightgbm", "random_forest"}:
        return shap.TreeExplainer(
            classifier,
            data=x_background,
            feature_perturbation="interventional",
            model_output="raw",
            feature_names=feature_names,
        )
    if kind == "logistic_regression":
        return shap.LinearExplainer(
            classifier,
            x_background,
            feature_names=feature_names,
        )
    return shap.Explainer(classifier.predict_proba, x_background, feature_names=feature_names)


def _condense_background(x_transformed: np.ndarray, k: int = 100) -> np.ndarray:
    """Return a small background distribution for explainer initialization."""

    if len(x_transformed) <= k:
        return x_transformed
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(x_transformed)
    return km.cluster_centers_


def compute_shap_values(
    pipeline: ImbPipeline,
    x_raw: pd.DataFrame,
    *,
    sample_size: int = 300,
    background_size: int = 100,
    seed: int = 42,
) -> ShapPayload:
    """Compute SHAP values for a sample of the input rows."""

    rng = np.random.default_rng(seed)
    if len(x_raw) > sample_size:
        idx = rng.choice(len(x_raw), size=sample_size, replace=False)
        x_sample = x_raw.iloc[idx].copy().reset_index(drop=True)
    else:
        x_sample = x_raw.reset_index(drop=True)

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = expand_feature_names(preprocessor)

    x_transformed = preprocessor.transform(x_sample[ALL_FEATURES])
    background = _condense_background(x_transformed, k=background_size)

    explainer = _pick_explainer(classifier, background, feature_names)
    log.info(
        "Running SHAP explainer (%s) on %d rows", type(explainer).__name__, len(x_transformed)
    )
    raw = explainer(x_transformed)

    values = np.asarray(raw.values)
    base = raw.base_values
    if values.ndim == 3:
        # Multi-output (e.g. RandomForest probability) → take positive class.
        values = values[..., 1]
        base = base[..., 1] if np.ndim(base) > 1 else base
    base_value = float(np.mean(base)) if hasattr(base, "__iter__") else float(base)

    return ShapPayload(
        values=values,
        base_value=base_value,
        feature_names=feature_names,
        X_transformed=x_transformed,
        X_raw=x_sample,
    )


# ---------------------------------------------------------------------------
# Plotly visualizations
# ---------------------------------------------------------------------------

def plot_global_importance(payload: ShapPayload, *, top_n: int = 15) -> go.Figure:
    """Mean absolute SHAP — the model's global feature ranking."""

    mean_abs = np.abs(payload.values).mean(axis=0)
    df = (
        pd.DataFrame({"feature": payload.feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Global SHAP Feature Importance",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=140, r=20, t=50, b=40),
        coloraxis_showscale=False,
    )
    return fig


def plot_summary_beeswarm(payload: ShapPayload, *, top_n: int = 12) -> go.Figure:
    """Plotly approximation of a SHAP beeswarm.

    Each point = (shap value, feature value) for one applicant.
    Colors encode the feature value (blue = low, red = high).
    """

    mean_abs = np.abs(payload.values).mean(axis=0)
    order = np.argsort(mean_abs)[-top_n:]
    fig = go.Figure()
    rng = np.random.default_rng(0)

    for rank, idx in enumerate(order):
        shap_col = payload.values[:, idx]
        feature_col = payload.X_transformed[:, idx]
        feature_norm = feature_col
        if feature_col.std() > 1e-9:
            feature_norm = (feature_col - feature_col.mean()) / feature_col.std()
        jitter = rng.normal(0.0, 0.08, size=len(shap_col))
        fig.add_trace(
            go.Scatter(
                x=shap_col,
                y=np.full_like(shap_col, rank, dtype=float) + jitter,
                mode="markers",
                marker=dict(
                    size=6,
                    color=feature_norm,
                    colorscale="RdBu_r",
                    showscale=rank == len(order) - 1,
                    colorbar=dict(title="Feature value (z-score)") if rank == len(order) - 1 else None,
                    opacity=0.78,
                ),
                name=payload.feature_names[idx],
                hovertemplate=(
                    f"<b>{payload.feature_names[idx]}</b><br>"
                    "SHAP=%{x:.3f}<br>"
                    "value (z)=%{marker.color:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=f"Top {top_n} SHAP Beeswarm",
        xaxis_title="SHAP value (impact on default log-odds)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(order))),
            ticktext=[payload.feature_names[i] for i in order],
        ),
        template="plotly_white",
        height=560,
        margin=dict(l=160, r=20, t=50, b=40),
    )
    return fig


def plot_local_waterfall(
    payload: ShapPayload, row_index: int, *, top_n: int = 12
) -> go.Figure:
    """Per-applicant SHAP waterfall."""

    if not 0 <= row_index < len(payload.values):
        raise IndexError(f"row_index {row_index} out of range [0, {len(payload.values) - 1}]")

    contribs = payload.values[row_index]
    order = np.argsort(np.abs(contribs))[-top_n:]
    feature_names = [payload.feature_names[i] for i in order][::-1]
    values = [float(contribs[i]) for i in order][::-1]

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(values) + ["total"],
            y=[*feature_names, "Predicted log-odds"],
            x=[*values, payload.base_value + sum(values)],
            connector={"line": {"color": "#bbb"}},
            increasing={"marker": {"color": APP_CONFIG.danger_color}},
            decreasing={"marker": {"color": APP_CONFIG.accent_color}},
            totals={"marker": {"color": APP_CONFIG.primary_color}},
            text=[f"{v:+.3f}" for v in values] + [""],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Local SHAP Waterfall (applicant #{row_index})",
        template="plotly_white",
        height=520,
        margin=dict(l=180, r=40, t=50, b=40),
        xaxis_title="Contribution to default log-odds",
    )
    return fig


def shap_reasons(
    payload: ShapPayload,
    row_index: int,
    *,
    top_n: int = 4,
) -> dict[str, list[tuple[str, float]]]:
    """Return human-readable approve/reject drivers for one applicant."""

    contribs = payload.values[row_index]
    order = np.argsort(contribs)
    bottom = order[:top_n]  # most negative — push toward "approve"
    top = order[-top_n:][::-1]  # most positive — push toward "reject"
    return {
        "raises_risk": [(payload.feature_names[i], float(contribs[i])) for i in top],
        "lowers_risk": [(payload.feature_names[i], float(contribs[i])) for i in bottom],
    }


__all__ = [
    "ShapPayload",
    "compute_shap_values",
    "plot_global_importance",
    "plot_summary_beeswarm",
    "plot_local_waterfall",
    "shap_reasons",
]
