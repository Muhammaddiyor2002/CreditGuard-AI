"""Reports — JSON / CSV / PDF exports + drift-detection placeholder."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import wasserstein_distance

from ...config import ALL_FEATURES, NUMERIC_FEATURES
from ...utils.reporting import generate_applicant_pdf
from ..state import auto_load_dataset, auto_load_model, get_state
from ..styling import hero


def _drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Compute Wasserstein drift per numeric feature."""

    rows = []
    for col in NUMERIC_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy()
        cur = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy()
        if len(ref) == 0 or len(cur) == 0:
            continue
        rows.append(
            {
                "feature": col,
                "ref_mean": float(np.mean(ref)),
                "current_mean": float(np.mean(cur)),
                "wasserstein_distance": float(wasserstein_distance(ref, cur)),
            }
        )
    return pd.DataFrame(rows).sort_values("wasserstein_distance", ascending=False)


def render() -> None:
    hero("Reports", "Export training summaries, scored portfolios, and drift snapshots")

    state = get_state()
    model = auto_load_model()
    df = auto_load_dataset()

    tab_summary, tab_export, tab_drift = st.tabs(
        ["Training summary", "Portfolio export", "Drift detection"]
    )

    # ------------------------------------------------------- training summary
    with tab_summary:
        if model is None:
            st.info("Train a model first.")
        else:
            payload = {
                "best_model": model.best_model_name,
                "metric": model.metric_name,
                "train_default_rate": model.train_default_rate,
                "n_train_rows": model.n_train_rows,
                "n_test_rows": model.n_test_rows,
                "test_metrics": model.test_metrics,
                "cv_results": [r.to_dict() for r in model.cv_results],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            st.json(payload, expanded=False)
            st.download_button(
                "Download training_summary.json",
                json.dumps(payload, indent=2, default=float),
                file_name="training_summary.json",
                mime="application/json",
                type="primary",
            )

    # ------------------------------------------------------- portfolio export
    with tab_export:
        if model is None or df is None:
            st.info("Need both a trained model and a dataset to export a scored portfolio.")
        else:
            sample = df.head(2000)[ALL_FEATURES].copy()
            proba = model.predict_proba(sample)
            sample["default_probability"] = proba
            buf = io.StringIO()
            sample.to_csv(buf, index=False)
            st.dataframe(sample.head(50), width="stretch")
            st.download_button(
                "Download scored_portfolio.csv",
                buf.getvalue(),
                file_name="scored_portfolio.csv",
                mime="text/csv",
                type="primary",
            )

            last = state.get("last_prediction")
            if last is not None:
                pdf = generate_applicant_pdf(
                    last["applicant"],
                    last["decision"],
                    model_name=model.best_model_name,
                )
                st.download_button(
                    "Download last applicant report (PDF)",
                    pdf,
                    file_name="last_applicant_report.pdf",
                    mime="application/pdf",
                )

    # ------------------------------------------------------- drift
    with tab_drift:
        if df is None:
            st.info("Load a dataset to enable drift checks.")
            return
        st.markdown(
            "Upload a **new** applicants CSV to compare against the current "
            "training distribution using Wasserstein distance per numeric feature."
        )
        f = st.file_uploader("Current-period applicants", type=["csv"], key="drift_upload")
        if f is None:
            return
        current = pd.read_csv(f)
        try:
            drift = _drift_table(df, current)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Drift comparison failed: {exc}")
            return

        if drift.empty:
            st.warning("No common numeric features between the two datasets.")
            return
        st.dataframe(drift.style.format(precision=3), width="stretch")
        flagged = drift[drift["wasserstein_distance"] > drift["wasserstein_distance"].mean() + 1e-6]
        if len(flagged):
            st.warning(
                f"{len(flagged)} feature(s) show above-average drift. "
                "Consider re-training the model with fresh data."
            )
