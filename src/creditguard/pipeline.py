"""End-to-end training orchestration.

Single-call helper used by the CLI and the Streamlit "Train Model" page:

    from creditguard.pipeline import train_full_pipeline
    artifact = train_full_pipeline()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATASET_CONFIG, PROCESSED_DATA_DIR, TRAINING_CONFIG
from .data import load_credit_dataset
from .evaluation import evaluate_model
from .explain import compute_shap_values
from .models import (
    DEFAULT_MODEL_PATH,
    TrainedModel,
    Trainer,
    save_artifact,
    save_training_summary,
)
from .utils.io import save_dataframe
from .utils.logging import get_logger

log = get_logger(__name__)


def train_full_pipeline(
    df: pd.DataFrame | None = None,
    *,
    use_optuna: bool = True,
    optuna_trials: int | None = None,
    optuna_timeout: int | None = None,
    models: list[str] | None = None,
    save_path: Path | str = DEFAULT_MODEL_PATH,
    use_openml: bool = True,
    synthetic_rows: int | None = None,
    compute_shap: bool = True,
) -> TrainedModel:
    """Run data load → train → evaluate → SHAP → persist.

    Returns the populated :class:`TrainedModel` artifact.
    """

    if df is None:
        df = load_credit_dataset(use_openml=use_openml, synthetic_rows=synthetic_rows)
    save_dataframe(df, PROCESSED_DATA_DIR / DATASET_CONFIG.processed_filename)

    trainer = Trainer(
        config=TRAINING_CONFIG,
        models=models,
        use_optuna=use_optuna,
        optuna_trials=optuna_trials,
        optuna_timeout=optuna_timeout,
    )
    artifact, x_test, y_test = trainer.train(df)

    proba = artifact.predict_proba(x_test)
    report = evaluate_model(y_test, proba)
    artifact.test_metrics = report.metrics
    artifact.test_curve = {
        "fpr": report.roc_curve["fpr"],
        "tpr": report.roc_curve["tpr"],
        "precision": report.pr_curve["precision"],
        "recall": report.pr_curve["recall"],
    }
    artifact.confusion_matrix = report.confusion
    artifact.y_test_true = y_test.to_numpy()
    artifact.y_test_proba = proba
    artifact.x_test_sample = x_test.reset_index(drop=True)

    log.info(
        "Held-out metrics: ROC-AUC=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f",
        report.metrics["roc_auc"],
        report.metrics["f1"],
        report.metrics["precision"],
        report.metrics["recall"],
    )

    if compute_shap:
        try:
            payload = compute_shap_values(
                artifact.pipeline,
                x_test,
                sample_size=min(300, len(x_test)),
            )
            artifact.shap_values = payload.values
            artifact.shap_base_value = payload.base_value
            artifact.shap_feature_names = payload.feature_names
            artifact.shap_X_sample = payload.X_raw
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP computation failed (%s); continuing without cached SHAP.", exc)

    save_artifact(artifact, save_path)
    save_training_summary(artifact)
    return artifact


__all__ = ["train_full_pipeline"]
