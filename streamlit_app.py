"""CreditGuard AI — Streamlit entrypoint.

Run locally with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure `src/` is on the path when the app is launched directly with
# `streamlit run streamlit_app.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from creditguard import __version__  # noqa: E402
from creditguard.app.pages import PAGE_REGISTRY  # noqa: E402
from creditguard.app.state import get_state  # noqa: E402
from creditguard.app.styling import inject_global_css  # noqa: E402
from creditguard.config import APP_CONFIG  # noqa: E402

st.set_page_config(
    page_title=APP_CONFIG.title,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()
get_state()

with st.sidebar:
    st.markdown(f"# 🛡️ {APP_CONFIG.title}")
    st.markdown(f"##### {APP_CONFIG.tagline}")
    st.markdown("---")
    page = st.radio("Navigate", list(PAGE_REGISTRY.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"v{__version__} • Python ML stack")
    st.caption("XGBoost • LightGBM • SHAP • Optuna")

PAGE_REGISTRY[page]()
