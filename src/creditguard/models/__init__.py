"""Model training, registry, and persistence."""

from .registry import compute_scale_pos_weight, get_model_spec, list_models
from .training import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    ModelTrainingResult,
    TrainedModel,
    Trainer,
    get_classifier,
    save_artifact,
    save_training_summary,
)

__all__ = [
    "compute_scale_pos_weight",
    "get_model_spec",
    "list_models",
    "ModelTrainingResult",
    "TrainedModel",
    "Trainer",
    "get_classifier",
    "save_artifact",
    "save_training_summary",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_METRICS_PATH",
]
