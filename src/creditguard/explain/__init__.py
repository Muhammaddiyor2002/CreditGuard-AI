"""SHAP explainability."""

from .shap_explain import (
    ShapPayload,
    compute_shap_values,
    plot_global_importance,
    plot_local_waterfall,
    plot_summary_beeswarm,
    shap_reasons,
)

__all__ = [
    "ShapPayload",
    "compute_shap_values",
    "plot_global_importance",
    "plot_local_waterfall",
    "plot_summary_beeswarm",
    "shap_reasons",
]
