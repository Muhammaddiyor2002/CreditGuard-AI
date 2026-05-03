"""SHAP Explainability — global + per-row drill-down."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...explain import (
    ShapPayload,
    compute_shap_values,
    plot_global_importance,
    plot_local_waterfall,
    plot_summary_beeswarm,
)
from ..state import auto_load_model, get_state
from ..styling import hero


def _payload_from_artifact(model) -> ShapPayload | None:
    if model.shap_values is None or model.shap_X_sample is None:
        return None
    pre = model.pipeline.named_steps["preprocessor"]
    x_transformed = pre.transform(model.shap_X_sample)
    return ShapPayload(
        values=model.shap_values,
        base_value=model.shap_base_value or 0.0,
        feature_names=model.shap_feature_names or [],
        X_transformed=x_transformed,
        X_raw=model.shap_X_sample,
    )


def render() -> None:
    hero("SHAP Explainability", "Why the model behaves the way it does — globally and locally")

    model = auto_load_model()
    if model is None:
        st.warning("No trained model. Train one on the **Train Model** page first.")
        return

    state = get_state()
    payload: ShapPayload | None = state.get("shap_payload") or _payload_from_artifact(model)

    if payload is None:
        if model.x_test_sample is None:
            st.info("No test sample cached on this artifact. Retrain to enable SHAP.")
            return
        with st.spinner("Computing SHAP values…"):
            payload = compute_shap_values(
                model.pipeline,
                model.x_test_sample,
                sample_size=min(300, len(model.x_test_sample)),
            )
        state["shap_payload"] = payload

    tab_global, tab_local = st.tabs(["Global importance", "Local waterfall"])

    with tab_global:
        c1, c2 = st.columns([1.0, 1.4])
        with c1:
            st.plotly_chart(plot_global_importance(payload), width="stretch")
        with c2:
            st.plotly_chart(plot_summary_beeswarm(payload), width="stretch")

    with tab_local:
        n = len(payload.values)
        idx = st.slider("Applicant row", 0, max(0, n - 1), 0)
        st.plotly_chart(plot_local_waterfall(payload, idx, top_n=12), width="stretch")
        st.markdown("**Applicant features**")
        st.dataframe(
            pd.DataFrame([payload.X_raw.iloc[idx].to_dict()]),
            width="stretch",
        )
