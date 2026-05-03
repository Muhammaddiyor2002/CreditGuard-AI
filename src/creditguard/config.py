"""Centralized configuration for CreditGuard AI.

All paths, feature lists, model hyperparameter search spaces, and business
thresholds live here so that the rest of the codebase has a single source of
truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Schema — canonical 13-feature CreditGuard schema
# ---------------------------------------------------------------------------

TARGET_COLUMN = "defaulted"

NUMERIC_FEATURES: list[str] = [
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "loan_term",
    "credit_score",
    "debt_to_income_ratio",
    "previous_defaults",
    "dependents",
    "savings_balance",
]

CATEGORICAL_FEATURES: list[str] = [
    "marital_status",
    "education",
    "loan_purpose",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORICAL_LEVELS: dict[str, list[str]] = {
    "marital_status": ["single", "married", "divorced", "widowed"],
    "education": ["none", "high_school", "bachelor", "master", "phd"],
    "loan_purpose": [
        "education",
        "home",
        "car",
        "business",
        "medical",
        "personal",
        "vacation",
        "debt_consolidation",
    ],
}


# ---------------------------------------------------------------------------
# Business rules — the bank's policy thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BusinessRules:
    """Probability thresholds that map a default-probability to a decision."""

    approve_below: float = 0.40
    reject_above: float = 0.70
    low_risk_below: float = 0.30
    medium_risk_below: float = 0.60

    def risk_band(self, probability: float) -> str:
        if probability < self.low_risk_below:
            return "Low Risk"
        if probability < self.medium_risk_below:
            return "Medium Risk"
        return "High Risk"

    def decision(self, probability: float) -> str:
        if probability < self.approve_below:
            return "Approve"
        if probability < self.reject_above:
            return "Manual Review"
        return "Reject"


BUSINESS_RULES = BusinessRules()


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    """Top-level knobs for the training pipeline."""

    test_size: float = 0.20
    validation_size: float = 0.10
    random_state: int = 42
    cv_folds: int = 5
    optuna_trials: int = 30
    optuna_timeout_seconds: int = 600
    use_smote: bool = True
    smote_strategy: str = "auto"
    n_jobs: int = -1
    primary_metric: str = "roc_auc"
    candidate_models: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    )


TRAINING_CONFIG = TrainingConfig()


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Where to source the training data from."""

    openml_dataset_id: int = 31
    """OpenML id 31 = `credit-g` (German Credit, 1000 rows)."""

    synthetic_rows: int = 9000
    """Synthetic rows blended in to bring dataset to ~10k for richer training."""

    raw_filename: str = "credit_data.parquet"
    processed_filename: str = "credit_data_processed.parquet"


DATASET_CONFIG = DatasetConfig()


# ---------------------------------------------------------------------------
# UI / display
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Streamlit app metadata."""

    title: str = "CreditGuard AI"
    tagline: str = "AI-powered credit risk prediction for modern banks"
    pages: tuple[str, ...] = (
        "Dashboard",
        "Upload CSV",
        "Train Model",
        "Evaluate Model",
        "Predict Applicant",
        "SHAP Explainability",
        "Reports",
    )
    primary_color: str = "#1E88E5"
    accent_color: str = "#43A047"
    danger_color: str = "#E53935"
    warning_color: str = "#FB8C00"
    risk_palette: dict[str, str] = field(
        default_factory=lambda: {
            "Low Risk": "#43A047",
            "Medium Risk": "#FB8C00",
            "High Risk": "#E53935",
        }
    )


APP_CONFIG = AppConfig()


__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "TARGET_COLUMN",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "ALL_FEATURES",
    "CATEGORICAL_LEVELS",
    "BusinessRules",
    "BUSINESS_RULES",
    "TrainingConfig",
    "TRAINING_CONFIG",
    "DatasetConfig",
    "DATASET_CONFIG",
    "AppConfig",
    "APP_CONFIG",
]
