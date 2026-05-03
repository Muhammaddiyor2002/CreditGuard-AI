"""Dashboard — portfolio-level KPI overview."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ...config import APP_CONFIG, BUSINESS_RULES, NUMERIC_FEATURES, TARGET_COLUMN
from ..state import auto_load_dataset, auto_load_model
from ..styling import hero


def _portfolio_distribution(model, df: pd.DataFrame) -> pd.DataFrame:
    """Score every row, bucket by risk band, return summary DataFrame."""

    proba = model.predict_proba(df)
    df_scored = df.copy()
    df_scored["default_probability"] = proba
    df_scored["risk_band"] = [BUSINESS_RULES.risk_band(p) for p in proba]
    return df_scored


def render() -> None:
    hero(APP_CONFIG.title, "Portfolio risk overview & monthly trends")

    model = auto_load_model()
    df = auto_load_dataset()

    if df is None:
        st.info(
            "No dataset loaded yet. Upload a CSV on the **Upload CSV** page or train "
            "a model on the **Train Model** page to generate the default dataset."
        )
        return

    # ---------------- portfolio KPIs ----------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Applicants", f"{len(df):,}")
    if TARGET_COLUMN in df.columns:
        col2.metric("Historical default rate", f"{df[TARGET_COLUMN].mean() * 100:.2f}%")
    if "loan_amount" in df.columns:
        col3.metric("Avg loan amount", f"${df['loan_amount'].mean():,.0f}")
    if "credit_score" in df.columns:
        col4.metric("Avg credit score", f"{df['credit_score'].mean():.0f}")

    st.markdown("---")

    if model is None:
        st.warning(
            "No trained model available — risk-band breakdown will be skipped. "
            "Go to **Train Model** to fit one."
        )
    else:
        scored = _portfolio_distribution(model, df.head(5000))
        breakdown = (
            scored["risk_band"]
            .value_counts()
            .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
            .reset_index()
        )
        breakdown.columns = ["risk_band", "count"]

        c1, c2 = st.columns([1.1, 1.4])
        with c1:
            st.subheader("Risk band distribution")
            fig = px.pie(
                breakdown,
                names="risk_band",
                values="count",
                color="risk_band",
                color_discrete_map=APP_CONFIG.risk_palette,
                hole=0.55,
            )
            fig.update_layout(template="plotly_white", height=380, margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch")

        with c2:
            st.subheader("Default-probability histogram")
            fig = px.histogram(
                scored,
                x="default_probability",
                nbins=40,
                color="risk_band",
                color_discrete_map=APP_CONFIG.risk_palette,
            )
            fig.add_vline(
                x=BUSINESS_RULES.approve_below,
                line_dash="dash",
                line_color="#43A047",
                annotation_text="Approve <",
                annotation_position="top",
            )
            fig.add_vline(
                x=BUSINESS_RULES.reject_above,
                line_dash="dash",
                line_color="#E53935",
                annotation_text="Reject >",
                annotation_position="top",
            )
            fig.update_layout(template="plotly_white", height=380, margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Monthly portfolio risk (simulated)")
        rng = np.random.default_rng(42)
        months = pd.date_range("2024-12-01", periods=12, freq="MS")
        baseline = scored["default_probability"].mean()
        seasonal = baseline + 0.04 * np.sin(np.linspace(0, 2 * np.pi, len(months)))
        noise = rng.normal(0, 0.012, size=len(months))
        monthly = pd.DataFrame(
            {
                "month": months,
                "expected_default_rate": np.clip(seasonal + noise, 0.05, 0.6),
                "loan_volume": rng.integers(800, 1500, size=len(months)),
            }
        )
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthly["month"],
                y=monthly["loan_volume"],
                name="Loan volume",
                marker_color="rgba(30, 136, 229, 0.45)",
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=monthly["month"],
                y=monthly["expected_default_rate"],
                name="Expected default rate",
                mode="lines+markers",
                line=dict(color=APP_CONFIG.danger_color, width=3),
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=380,
            yaxis=dict(title="Default rate", tickformat=".0%"),
            yaxis2=dict(title="Loan volume", overlaying="y", side="right", showgrid=False),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Numeric feature snapshot")
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    if available:
        st.dataframe(df[available].describe().T.round(2), width="stretch")
