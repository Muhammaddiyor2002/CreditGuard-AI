"""Evaluate Model — render the held-out evaluation report with live threshold sweep."""

from __future__ import annotations

import streamlit as st

from ...evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    plot_threshold_curves,
)
from ..state import auto_load_model
from ..styling import hero


def render() -> None:
    hero("Evaluate Model", "Hold-out test metrics, ROC/PR curves, and threshold sweep")

    model = auto_load_model()
    if model is None:
        st.warning("No trained model loaded. Train one on the **Train Model** page first.")
        return

    if model.y_test_true is None or model.y_test_proba is None:
        st.info("No cached test predictions. Re-run training to populate metrics.")
        return

    threshold = st.slider(
        "Decision threshold (default probability)",
        0.05,
        0.95,
        0.50,
        0.01,
        help="Move the threshold to see how precision, recall, and the confusion matrix change.",
    )
    report = evaluate_model(model.y_test_true, model.y_test_proba, threshold=threshold)

    cols = st.columns(len(report.metrics))
    for col, (k, v) in zip(cols, report.metrics.items(), strict=False):
        col.metric(k.upper(), f"{v:.4f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_roc_curve(report), width="stretch")
    with c2:
        st.plotly_chart(plot_pr_curve(report), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_confusion_matrix(report), width="stretch")
    with c4:
        st.plotly_chart(plot_threshold_curves(report), width="stretch")

    with st.expander("Per-model cross-validation history", expanded=False):
        st.dataframe(
            [
                {
                    "model": r.name,
                    "cv_roc_auc_mean": round(r.cv_score_mean, 4),
                    "cv_roc_auc_std": round(r.cv_score_std, 4),
                    "fit_seconds": round(r.fit_seconds, 1),
                    "optuna_trials": r.n_optuna_trials,
                    "best_params": r.best_params,
                }
                for r in model.cv_results
            ],
            width="stretch",
        )

    st.caption(f"Best model in production: **{model.best_model_name}**")
