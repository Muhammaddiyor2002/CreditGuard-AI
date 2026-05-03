"""Data ingestion and synthetic generation."""

from .loader import load_credit_dataset
from .synthetic import generate_synthetic_dataset

__all__ = ["load_credit_dataset", "generate_synthetic_dataset"]
