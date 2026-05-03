"""Predict Applicant — single-applicant risk scoring + PDF download."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...config import (
    APP_CONFIG,
    BUSINESS_RULES,
    CATEGORICAL_LEVELS,
)
from ...explain import compute_shap_values, plot_local_waterfall, shap_reasons
from ...utils.reporting import generate_applicant_pdf
from ..state import auto_load_model, get_state
from ..styling import decision_banner, hero


def _gauge(prob: float) -> go.Figure:
    color = APP_CONFIG.risk_palette[BUSINESS_RULES.risk_band(prob)]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Default probability"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.32},
                "steps": [
                    {"range": [0, BUSINESS_RULES.approve_below * 100], "color": "#E8F5E9"},
                    {
                        "range": [
                            BUSINESS_RULES.approve_below * 100,
                            BUSINESS_RULES.reject_above * 100,
                        ],
                        "color": "#FFF3E0",
                    },
                    {"range": [BUSINESS_RULES.reject_above * 100, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "#212121", "width": 3},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(t=30, b=10, l=10, r=10))
    return fig


def _applicant_form() -> dict[str, object]:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 90, 35)
        income = st.number_input("Annual income ($)", 0, 1_000_000, 60_000, step=1000)
        employment_years = st.number_input("Employment years", 0, 50, 6)
        loan_amount = st.number_input("Loan amount ($)", 100, 1_000_000, 12_000, step=500)
    with c2:
        loan_term = st.number_input("Loan term (months)", 6, 240, 36, step=6)
        credit_score = st.number_input("Credit score", 300, 850, 700)
        debt_to_income_ratio = st.slider("Debt-to-income ratio", 0.0, 1.5, 0.30, 0.01)
        previous_defaults = st.number_input("Previous defaults", 0, 10, 0)
    with c3:
        dependents = st.number_input("Dependents", 0, 10, 1)
        savings_balance = st.number_input("Savings balance ($)", 0, 1_000_000, 5_000, step=500)
        marital_status = st.selectbox("Marital status", CATEGORICAL_LEVELS["marital_status"])
        education = st.selectbox("Education", CATEGORICAL_LEVELS["education"], index=2)

    loan_purpose = st.selectbox("Loan purpose", CATEGORICAL_LEVELS["loan_purpose"])

    return {
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "previous_defaults": previous_defaults,
        "dependents": dependents,
        "savings_balance": savings_balance,
        "marital_status": marital_status,
        "education": education,
        "loan_purpose": loan_purpose,
    }


def render() -> None:
    hero("Predict Applicant", "Score a single loan applicant and explain the decision")

    model = auto_load_model()
    if model is None:
        st.warning("No trained model. Train one on the **Train Model** page first.")
        return

    state = get_state()
    with st.form("applicant_form"):
        applicant = _applicant_form()
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if not submitted and state.get("last_prediction") is None:
        st.info("Fill in the form and click **Predict** to score this applicant.")
        return

    if submitted:
        decision = model.score_applicant(applicant)
        state["last_prediction"] = {"applicant": applicant, "decision": decision}
    else:
        decision = state["last_prediction"]["decision"]
        applicant = state["last_prediction"]["applicant"]

    st.markdown("---")
    decision_banner(decision["decision"])

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        st.plotly_chart(_gauge(decision["default_probability"]), width="stretch")
    with c2:
        st.metric("Estimated credit score", decision["credit_score_estimate"])
        st.metric("Risk band", decision["risk_band"])
        st.metric(
            "Default probability",
            f"{decision['default_probability'] * 100:.2f}%",
        )
        st.caption(
            f"Approve if probability < {BUSINESS_RULES.approve_below:.0%} • "
            f"Reject if > {BUSINESS_RULES.reject_above:.0%}"
        )

    # SHAP explanation for this applicant -----------------------------------
    with st.expander("Why this decision? (SHAP local explanation)", expanded=True):
        applicant_df = pd.DataFrame([applicant])
        try:
            payload = compute_shap_values(
                model.pipeline, applicant_df, sample_size=1, background_size=64
            )
            st.plotly_chart(plot_local_waterfall(payload, 0, top_n=10), width="stretch")
            reasons = shap_reasons(payload, 0, top_n=4)
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Top factors raising risk**")
                for feat, val in reasons["raises_risk"]:
                    st.markdown(f"- `{feat}` → +{val:.3f}")
            with r2:
                st.markdown("**Top factors lowering risk**")
                for feat, val in reasons["lowers_risk"]:
                    st.markdown(f"- `{feat}` → {val:.3f}")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"SHAP local explanation unavailable: {exc}")
            reasons = None

    # PDF download ----------------------------------------------------------
    pdf_bytes = generate_applicant_pdf(
        applicant,
        decision,
        reasons=reasons,
        model_name=model.best_model_name,
    )
    st.download_button(
        "Download PDF risk report",
        data=pdf_bytes,
        file_name="creditguard_report.pdf",
        mime="application/pdf",
        type="primary",
    )
