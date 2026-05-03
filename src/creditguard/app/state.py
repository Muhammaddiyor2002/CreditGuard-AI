"""Streamlit session-state helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ..config import DATASET_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR
from ..models import DEFAULT_MODEL_PATH, TrainedModel
from ..utils.io import load_dataframe, load_joblib


def get_state() -> dict[str, Any]:
    """Initialize and return the session-state dict."""

    defaults = {
        "model": None,
        "dataset": None,
        "uploaded_dataset": None,
        "training_log": [],
        "shap_payload": None,
        "last_prediction": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    return st.session_state  # type: ignore[return-value]


def auto_load_model() -> TrainedModel | None:
    """Try to lazily restore a trained model from disk."""

    state = get_state()
    if state["model"] is not None:
        return state["model"]
    if Path(DEFAULT_MODEL_PATH).exists():
        try:
            state["model"] = load_joblib(DEFAULT_MODEL_PATH)
            return state["model"]
        except Exception:  # noqa: BLE001
            return None
    return None


def auto_load_dataset() -> pd.DataFrame | None:
    """Try to lazily restore a previously processed dataset from disk."""

    state = get_state()
    if state["dataset"] is not None:
        return state["dataset"]
    for candidate in (
        PROCESSED_DATA_DIR / DATASET_CONFIG.processed_filename,
        RAW_DATA_DIR / DATASET_CONFIG.raw_filename,
    ):
        if Path(candidate).exists():
            try:
                state["dataset"] = load_dataframe(candidate)
                return state["dataset"]
            except Exception:  # noqa: BLE001
                continue
    return None


__all__ = ["get_state", "auto_load_model", "auto_load_dataset"]
