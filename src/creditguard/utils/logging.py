"""Structured logging helpers used across the project."""

from __future__ import annotations

import logging
import sys
from typing import Final

_FMT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"

_INITIALIZED = False


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger exactly once."""

    global _INITIALIZED
    if _INITIALIZED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Tame noisy libraries.
    for noisy in ("matplotlib", "PIL", "openml", "urllib3", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, initializing logging on first call."""

    setup_logging()
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
