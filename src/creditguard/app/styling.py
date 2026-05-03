"""Reusable CSS / styling helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st

from ..config import APP_CONFIG

_GLOBAL_CSS = f"""
<style>
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0D47A1 0%, #1565C0 50%, {APP_CONFIG.primary_color} 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #ECEFF1 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
        color: #FFFFFF !important;
    }}
    div[data-testid="metric-container"] {{
        background: #FFFFFF;
        padding: 14px 16px;
        border-radius: 12px;
        border-left: 4px solid {APP_CONFIG.primary_color};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }}
    .cg-hero {{
        background: linear-gradient(135deg, #0D47A1 0%, {APP_CONFIG.primary_color} 100%);
        padding: 26px 32px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 22px;
    }}
    .cg-hero h1 {{ margin: 0; font-size: 2.0rem; color: #FFFFFF; }}
    .cg-hero p  {{ margin: 6px 0 0; opacity: 0.92; font-size: 1.05rem; }}
    .cg-decision {{
        padding: 18px 22px;
        border-radius: 14px;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.5rem;
        text-align: center;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
    }}
    .cg-decision.approve {{ background: {APP_CONFIG.accent_color}; }}
    .cg-decision.review  {{ background: {APP_CONFIG.warning_color}; }}
    .cg-decision.reject  {{ background: {APP_CONFIG.danger_color}; }}
    .cg-card {{
        background: #FFFFFF;
        padding: 20px 22px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background: #F5F7FA;
        border-radius: 10px 10px 0 0;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {APP_CONFIG.primary_color} !important;
        color: #FFFFFF !important;
    }}
</style>
"""


def inject_global_css() -> None:
    """Inject CreditGuard's global CSS once per session."""

    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='cg-hero'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def decision_banner(decision: str) -> None:
    cls = {"Approve": "approve", "Manual Review": "review", "Reject": "reject"}.get(
        decision, "review"
    )
    st.markdown(f"<div class='cg-decision {cls}'>{decision}</div>", unsafe_allow_html=True)


__all__ = ["inject_global_css", "hero", "decision_banner"]
