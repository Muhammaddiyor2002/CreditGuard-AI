"""Per-model factory + Optuna search-space registry.

Each candidate model exposes:

  * ``estimator(params)`` — instantiate the sklearn-compatible estimator;
  * ``search_space(trial)`` — Optuna sampling of hyperparameters;
  * ``default_params()`` — a sane fallback if Optuna is skipped.

Adding a new model means writing one new ``ModelSpec`` here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ..config import TRAINING_CONFIG


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one candidate classifier."""

    name: str
    requires_scaled_features: bool
    estimator_factory: Callable[[dict], ClassifierMixin]
    default_params_factory: Callable[[float], dict]
    search_space: Callable[[optuna.Trial, float], dict]
    supports_native_class_weight: bool = True
    supports_class_weight_dict: bool = False


def _logistic_regression(params: dict) -> ClassifierMixin:
    return LogisticRegression(**params)


def _logreg_defaults(scale_pos_weight: float) -> dict:
    del scale_pos_weight  # we handle imbalance via SMOTE / class_weight="balanced"
    return {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": TRAINING_CONFIG.random_state,
    }


def _logreg_search(trial: optuna.Trial, scale_pos_weight: float) -> dict:
    del scale_pos_weight
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "random_state": TRAINING_CONFIG.random_state,
    }


def _random_forest(params: dict) -> ClassifierMixin:
    return RandomForestClassifier(**params)


def _rf_defaults(scale_pos_weight: float) -> dict:
    del scale_pos_weight
    return {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": TRAINING_CONFIG.random_state,
        "n_jobs": -1,
    }


def _rf_search(trial: optuna.Trial, scale_pos_weight: float) -> dict:
    del scale_pos_weight
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", None]
        ),
        "random_state": TRAINING_CONFIG.random_state,
        "n_jobs": -1,
    }


def _xgboost(params: dict) -> ClassifierMixin:
    base = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "eval_metric": "auc",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": TRAINING_CONFIG.random_state,
    }
    return XGBClassifier(**{**base, **params})


def _xgb_defaults(scale_pos_weight: float) -> dict:
    return {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    }


def _xgb_search(trial: optuna.Trial, scale_pos_weight: float) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 0.5 * scale_pos_weight, 2.0 * scale_pos_weight
        ),
    }


def _lightgbm(params: dict) -> ClassifierMixin:
    base = {
        "objective": "binary",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": TRAINING_CONFIG.random_state,
    }
    return LGBMClassifier(**{**base, **params})


def _lgbm_defaults(scale_pos_weight: float) -> dict:
    return {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": scale_pos_weight,
    }


def _lgbm_search(trial: optuna.Trial, scale_pos_weight: float) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 5.0, log=True),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 0.5 * scale_pos_weight, 2.0 * scale_pos_weight
        ),
    }


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Compute the gradient-boosting `scale_pos_weight` for class imbalance."""

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return max(neg / pos, 1e-3)


_REGISTRY: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        requires_scaled_features=True,
        estimator_factory=_logistic_regression,
        default_params_factory=_logreg_defaults,
        search_space=_logreg_search,
        supports_class_weight_dict=True,
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        requires_scaled_features=False,
        estimator_factory=_random_forest,
        default_params_factory=_rf_defaults,
        search_space=_rf_search,
        supports_class_weight_dict=True,
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        requires_scaled_features=False,
        estimator_factory=_xgboost,
        default_params_factory=_xgb_defaults,
        search_space=_xgb_search,
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        requires_scaled_features=False,
        estimator_factory=_lightgbm,
        default_params_factory=_lgbm_defaults,
        search_space=_lgbm_search,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = ["ModelSpec", "get_model_spec", "list_models", "compute_scale_pos_weight"]
