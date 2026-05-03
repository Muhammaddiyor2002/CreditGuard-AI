"""Smoke test for the end-to-end training pipeline (no Optuna, no OpenML)."""

from __future__ import annotations

from creditguard.config import ALL_FEATURES, BUSINESS_RULES
from creditguard.data.synthetic import generate_synthetic_dataset
from creditguard.pipeline import train_full_pipeline


def test_pipeline_smoke(tmp_path):
    df = generate_synthetic_dataset(n_rows=600, missing_rate=0.05, seed=0)
    artifact = train_full_pipeline(
        df=df,
        use_optuna=False,
        models=["logistic_regression", "random_forest"],
        save_path=tmp_path / "model.joblib",
        compute_shap=False,
    )
    assert artifact.best_model_name in {"logistic_regression", "random_forest"}
    assert "roc_auc" in artifact.test_metrics
    assert artifact.test_metrics["roc_auc"] > 0.55  # better than random
    assert (tmp_path / "model.joblib").exists()


def test_business_rules_decisions():
    rules = BUSINESS_RULES
    assert rules.decision(0.10) == "Approve"
    assert rules.decision(0.55) == "Manual Review"
    assert rules.decision(0.90) == "Reject"
    assert rules.risk_band(0.10) == "Low Risk"
    assert rules.risk_band(0.45) == "Medium Risk"
    assert rules.risk_band(0.85) == "High Risk"


def test_score_applicant_payload(tmp_path):
    df = generate_synthetic_dataset(n_rows=400, missing_rate=0.0, seed=0)
    artifact = train_full_pipeline(
        df=df,
        use_optuna=False,
        models=["logistic_regression"],
        save_path=tmp_path / "model.joblib",
        compute_shap=False,
    )
    sample = df.iloc[0][ALL_FEATURES].to_dict()
    payload = artifact.score_applicant(sample)
    assert 0.0 <= payload["default_probability"] <= 1.0
    assert payload["risk_band"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert payload["decision"] in {"Approve", "Manual Review", "Reject"}
