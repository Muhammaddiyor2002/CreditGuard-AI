"""Streamlit page modules."""

from . import dashboard, evaluate, explainability, predict, reports, train, upload

PAGE_REGISTRY = {
    "Dashboard": dashboard.render,
    "Upload CSV": upload.render,
    "Train Model": train.render,
    "Evaluate Model": evaluate.render,
    "Predict Applicant": predict.render,
    "SHAP Explainability": explainability.render,
    "Reports": reports.render,
}

__all__ = ["PAGE_REGISTRY"]
