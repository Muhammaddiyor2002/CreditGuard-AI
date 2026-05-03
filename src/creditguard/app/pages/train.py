"""Train Model — kick off the full training pipeline from the UI."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from ...config import TRAINING_CONFIG
from ...models import list_models
from ...pipeline import train_full_pipeline
from ..state import auto_load_dataset, get_state
from ..styling import hero


def render() -> None:
    hero("Train Model", "Configure and run the CreditGuard training pipeline")

    state = get_state()
    df = state.get("uploaded_dataset") or auto_load_dataset()

    with st.expander("Training configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            use_optuna = st.checkbox("Optuna hyperparameter tuning", value=True)
        with c2:
            trials = st.number_input(
                "Optuna trials per model",
                min_value=5,
                max_value=200,
                value=15,
                step=5,
                disabled=not use_optuna,
            )
        with c3:
            timeout = st.number_input(
                "Per-model timeout (seconds)",
                min_value=60,
                max_value=3600,
                value=int(TRAINING_CONFIG.optuna_timeout_seconds),
                step=30,
                disabled=not use_optuna,
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            models = st.multiselect(
                "Models to train",
                options=list_models(),
                default=list(TRAINING_CONFIG.candidate_models),
            )
        with c5:
            use_openml = st.checkbox(
                "Include OpenML German Credit",
                value=True,
                help="Disable to train on synthetic data only (offline mode).",
            )
        with c6:
            synthetic_rows = st.number_input(
                "Synthetic rows to blend in",
                min_value=0,
                max_value=50_000,
                value=9_000,
                step=500,
            )

        compute_shap = st.checkbox("Compute & cache SHAP values after training", value=True)

    st.markdown("---")

    btn = st.button(
        "Train all models",
        type="primary",
        disabled=not models,
        width="stretch",
    )

    if not btn:
        if df is not None:
            st.caption(f"Currently loaded dataset: **{len(df):,}** rows.")
        else:
            st.info("No dataset cached on disk; one will be downloaded/generated when you train.")
        return

    if df is None:
        df_arg: pd.DataFrame | None = None
    else:
        df_arg = df

    progress = st.progress(0, text="Starting training…")
    status = st.empty()

    t0 = time.time()
    try:
        artifact = train_full_pipeline(
            df=df_arg,
            use_optuna=use_optuna,
            optuna_trials=int(trials) if use_optuna else None,
            optuna_timeout=int(timeout) if use_optuna else None,
            models=models or None,
            use_openml=use_openml,
            synthetic_rows=int(synthetic_rows) if df_arg is None else None,
            compute_shap=compute_shap,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Training failed: {exc}")
        return
    progress.progress(100, text="Training complete.")
    status.success(f"Training finished in {time.time() - t0:.1f}s.")

    state["model"] = artifact

    st.markdown("### Cross-validated leaderboard")
    leaderboard = pd.DataFrame(
        [
            {
                "model": r.name,
                "cv_roc_auc_mean": r.cv_score_mean,
                "cv_roc_auc_std": r.cv_score_std,
                "fit_seconds": r.fit_seconds,
                "optuna_trials": r.n_optuna_trials,
            }
            for r in artifact.cv_results
        ]
    ).sort_values("cv_roc_auc_mean", ascending=False)
    st.dataframe(leaderboard.style.format(precision=4), width="stretch")

    st.markdown("### Held-out test metrics")
    metrics_cols = st.columns(len(artifact.test_metrics))
    for col, (name, value) in zip(metrics_cols, artifact.test_metrics.items(), strict=False):
        col.metric(name.upper(), f"{value:.4f}")

    st.success(f"Best model: **{artifact.best_model_name}** — saved to `models/`.")
