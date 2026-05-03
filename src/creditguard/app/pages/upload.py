"""Upload CSV — bring your own dataset (training) or applicants (batch scoring)."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from ...config import ALL_FEATURES, BUSINESS_RULES, TARGET_COLUMN
from ..state import auto_load_model, get_state
from ..styling import hero


def _validate_schema(df: pd.DataFrame, *, require_target: bool = False) -> tuple[bool, list[str]]:
    expected = list(ALL_FEATURES)
    if require_target:
        expected.append(TARGET_COLUMN)
    missing = [c for c in expected if c not in df.columns]
    return len(missing) == 0, missing


def render() -> None:
    hero("Upload CSV", "Bring your own dataset for training or batch-score applicants")

    state = get_state()
    tab_train, tab_score = st.tabs(["Training dataset", "Batch scoring"])

    # --------------------------------------------------------------- training
    with tab_train:
        st.markdown(
            "Upload a CSV with all 13 CreditGuard features **plus** a `defaulted` "
            "column (0/1). The data will be used by the **Train Model** page."
        )
        f = st.file_uploader("Training CSV", type=["csv"], key="train_upload")
        if f is not None:
            df = pd.read_csv(f)
            ok, missing = _validate_schema(df, require_target=True)
            if not ok:
                st.error(f"Missing required columns: {missing}")
            else:
                state["uploaded_dataset"] = df
                state["dataset"] = df
                st.success(f"Loaded {len(df):,} rows. Default rate: "
                           f"{df[TARGET_COLUMN].mean() * 100:.2f}%")
                st.dataframe(df.head(20), width="stretch")

    # --------------------------------------------------------------- scoring
    with tab_score:
        model = auto_load_model()
        if model is None:
            st.warning("No trained model in memory. Train one on the **Train Model** page first.")
            return

        st.markdown(
            "Upload a CSV of applicants (no `defaulted` column needed) and the "
            "current model will score every row."
        )
        f = st.file_uploader("Applicant CSV", type=["csv"], key="score_upload")
        if f is None:
            return

        df = pd.read_csv(f)
        ok, missing = _validate_schema(df, require_target=False)
        if not ok:
            st.error(f"Missing required columns: {missing}")
            return

        proba = model.predict_proba(df[ALL_FEATURES])
        scored = df.copy()
        scored["default_probability"] = proba
        scored["risk_band"] = [BUSINESS_RULES.risk_band(p) for p in proba]
        scored["decision"] = [BUSINESS_RULES.decision(p) for p in proba]

        st.success(f"Scored {len(scored):,} applicants.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Approve", int((scored["decision"] == "Approve").sum()))
        c2.metric("Manual Review", int((scored["decision"] == "Manual Review").sum()))
        c3.metric("Reject", int((scored["decision"] == "Reject").sum()))

        st.dataframe(
            scored[["default_probability", "risk_band", "decision", *ALL_FEATURES[:6]]].head(50),
            width="stretch",
        )

        buf = io.StringIO()
        scored.to_csv(buf, index=False)
        st.download_button(
            label="Download scored CSV",
            data=buf.getvalue(),
            file_name="creditguard_scored.csv",
            mime="text/csv",
            type="primary",
        )
