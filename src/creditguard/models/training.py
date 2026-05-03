"""Training orchestrator.

Trains every candidate model with Optuna (or default params) using a unified
preprocessing + SMOTE pipeline, then picks the best one by mean
cross-validated ROC-AUC.

The artifact saved to disk is :class:`TrainedModel` — a single object that
holds the fitted preprocessor + classifier + metadata, and can score either
a raw applicant DataFrame or a batch CSV without any further wiring.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ..config import (
    ALL_FEATURES,
    BUSINESS_RULES,
    CATEGORICAL_LEVELS,
    MODELS_DIR,
    TARGET_COLUMN,
    TRAINING_CONFIG,
    TrainingConfig,
)
from ..features.preprocess import build_preprocessor, expand_feature_names, split_features_target
from ..utils.io import save_joblib
from ..utils.logging import get_logger
from .registry import compute_scale_pos_weight, get_model_spec, list_models

log = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ModelTrainingResult:
    """Per-model record produced by :class:`Trainer`."""

    name: str
    best_params: dict[str, Any]
    cv_score_mean: float
    cv_score_std: float
    cv_scores: list[float]
    fit_seconds: float
    n_optuna_trials: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cv_scores"] = [float(s) for s in self.cv_scores]
        return d


@dataclass
class TrainedModel:
    """Top-level saved artifact."""

    pipeline: ImbPipeline
    best_model_name: str
    feature_names: list[str]
    metric_name: str
    cv_results: list[ModelTrainingResult] = field(default_factory=list)
    train_default_rate: float = 0.0
    n_train_rows: int = 0
    n_test_rows: int = 0

    # Test-set metrics filled in by the evaluation step.
    test_metrics: dict[str, Any] = field(default_factory=dict)
    test_curve: dict[str, list[float]] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    # Held-out predictions, stored so the UI can sweep thresholds without retraining.
    y_test_true: np.ndarray | None = None
    y_test_proba: np.ndarray | None = None
    x_test_sample: pd.DataFrame | None = None

    # Optional cached SHAP values for the test set.
    shap_values: np.ndarray | None = None
    shap_base_value: float | None = None
    shap_feature_names: list[str] = field(default_factory=list)
    shap_X_sample: pd.DataFrame | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Return the per-row default probability."""

        proba = self.pipeline.predict_proba(X[ALL_FEATURES])
        return proba[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:  # noqa: N803
        return (self.predict_proba(X) >= threshold).astype(int)

    def score_applicant(self, applicant: dict | pd.DataFrame) -> dict[str, Any]:
        """Score a single applicant and return the full decision payload."""

        if isinstance(applicant, dict):
            df = pd.DataFrame([applicant])
        else:
            df = applicant.copy()

        # Coerce categorical levels to a known vocabulary.
        for col, levels in CATEGORICAL_LEVELS.items():
            if col in df.columns:
                df.loc[~df[col].isin(levels), col] = levels[0]

        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        df = df[ALL_FEATURES]

        prob = float(self.predict_proba(df)[0])
        return {
            "default_probability": prob,
            "credit_score_estimate": int(round((1.0 - prob) * 850)),
            "risk_band": BUSINESS_RULES.risk_band(prob),
            "decision": BUSINESS_RULES.decision(prob),
            "thresholds": {
                "approve_below": BUSINESS_RULES.approve_below,
                "reject_above": BUSINESS_RULES.reject_above,
            },
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """End-to-end CreditGuard training pipeline."""

    def __init__(
        self,
        config: TrainingConfig = TRAINING_CONFIG,
        *,
        models: list[str] | None = None,
        use_optuna: bool = True,
        optuna_trials: int | None = None,
        optuna_timeout: int | None = None,
    ) -> None:
        self.config = config
        self.models = models or list(config.candidate_models)
        for m in self.models:
            if m not in list_models():
                raise ValueError(f"Unknown model {m!r}; choose from {list_models()}.")
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials if optuna_trials is not None else config.optuna_trials
        self.optuna_timeout = (
            optuna_timeout if optuna_timeout is not None else config.optuna_timeout_seconds
        )

    # ------------------------------------------------------------------ pipeline
    def _build_pipeline(self, model_name: str, params: dict[str, Any]) -> ImbPipeline:
        spec = get_model_spec(model_name)
        preprocessor = build_preprocessor(scale_numeric=spec.requires_scaled_features)
        steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
        if self.config.use_smote:
            steps.append(
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=self.config.smote_strategy,
                        random_state=self.config.random_state,
                        k_neighbors=5,
                    ),
                )
            )
        steps.append(("classifier", spec.estimator_factory(params)))
        return ImbPipeline(steps=steps)

    # ------------------------------------------------------------------ tuning
    def _objective(
        self,
        trial: optuna.Trial,
        model_name: str,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        scale_pos_weight: float,
    ) -> float:
        spec = get_model_spec(model_name)
        params = spec.search_space(trial, scale_pos_weight)
        pipeline = self._build_pipeline(model_name, params)
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=self.config.primary_metric,
            n_jobs=1,
            error_score="raise",
        )
        return float(scores.mean())

    def _tune_model(
        self,
        model_name: str,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        scale_pos_weight: float,
    ) -> tuple[dict[str, Any], int]:
        log.info("→ Tuning %s with up to %d Optuna trials", model_name, self.optuna_trials)
        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(
            lambda t: self._objective(t, model_name, x_train, y_train, scale_pos_weight),
            n_trials=self.optuna_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=False,
            gc_after_trial=True,
        )
        return study.best_params, len(study.trials)

    # ------------------------------------------------------------------ scoring
    def _score_model(
        self,
        model_name: str,
        params: dict[str, Any],
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[float, float, list[float]]:
        pipeline = self._build_pipeline(model_name, params)
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=self.config.primary_metric,
            n_jobs=1,
        )
        return float(scores.mean()), float(scores.std()), [float(s) for s in scores]

    # ------------------------------------------------------------------ public
    def train(self, df: pd.DataFrame) -> tuple[TrainedModel, pd.DataFrame, pd.Series]:
        """Run the full training pipeline and return the best :class:`TrainedModel`."""

        X, y = split_features_target(df)
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        scale_pos_weight = compute_scale_pos_weight(y_train.to_numpy())
        log.info(
            "Training set: %d rows, default_rate=%.3f, scale_pos_weight=%.3f",
            len(x_train),
            float(y_train.mean()),
            scale_pos_weight,
        )

        results: list[ModelTrainingResult] = []
        best_pipeline: ImbPipeline | None = None
        best_score = -np.inf
        best_name = ""

        for model_name in self.models:
            t0 = time.time()
            n_trials = 0
            if self.use_optuna:
                best_params, n_trials = self._tune_model(
                    model_name, x_train, y_train, scale_pos_weight
                )
            else:
                spec = get_model_spec(model_name)
                best_params = spec.default_params_factory(scale_pos_weight)

            mean_score, std_score, scores = self._score_model(
                model_name, best_params, x_train, y_train
            )
            elapsed = time.time() - t0
            log.info(
                "%-20s ROC-AUC=%.4f (±%.4f) | trials=%d | %.1fs",
                model_name,
                mean_score,
                std_score,
                n_trials,
                elapsed,
            )
            results.append(
                ModelTrainingResult(
                    name=model_name,
                    best_params=best_params,
                    cv_score_mean=mean_score,
                    cv_score_std=std_score,
                    cv_scores=scores,
                    fit_seconds=elapsed,
                    n_optuna_trials=n_trials,
                )
            )

            if mean_score > best_score:
                best_score = mean_score
                best_name = model_name
                best_pipeline = self._build_pipeline(model_name, best_params)

        assert best_pipeline is not None
        log.info("Best model: %s with CV ROC-AUC=%.4f", best_name, best_score)

        best_pipeline.fit(x_train, y_train)
        preprocessor: ColumnTransformer = best_pipeline.named_steps["preprocessor"]
        feature_names = expand_feature_names(preprocessor)

        artifact = TrainedModel(
            pipeline=best_pipeline,
            best_model_name=best_name,
            feature_names=feature_names,
            metric_name=self.config.primary_metric,
            cv_results=results,
            train_default_rate=float(y_train.mean()),
            n_train_rows=len(x_train),
            n_test_rows=len(x_test),
        )
        return artifact, x_test, y_test


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = MODELS_DIR / "creditguard_model.joblib"
DEFAULT_METRICS_PATH = MODELS_DIR / "training_summary.json"


def save_artifact(model: TrainedModel, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    return save_joblib(model, path)


def save_training_summary(
    model: TrainedModel,
    path: Path | str = DEFAULT_METRICS_PATH,
) -> Path:
    summary = {
        "best_model": model.best_model_name,
        "metric": model.metric_name,
        "n_train_rows": model.n_train_rows,
        "n_test_rows": model.n_test_rows,
        "train_default_rate": model.train_default_rate,
        "cv_results": [r.to_dict() for r in model.cv_results],
        "test_metrics": model.test_metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=float))
    return path


def get_classifier(pipeline: ImbPipeline) -> ClassifierMixin:
    return pipeline.named_steps["classifier"]


__all__ = [
    "ModelTrainingResult",
    "TrainedModel",
    "Trainer",
    "save_artifact",
    "save_training_summary",
    "get_classifier",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_METRICS_PATH",
]
