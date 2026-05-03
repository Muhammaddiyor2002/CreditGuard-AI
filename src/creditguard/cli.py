"""``creditguard`` console-script entrypoint.

Examples:
    $ creditguard train --no-optuna           # quick smoke run
    $ creditguard train --trials 30           # full Optuna tuning
    $ creditguard predict --csv applicants.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import ALL_FEATURES, BUSINESS_RULES, TRAINING_CONFIG
from .models import DEFAULT_MODEL_PATH, TrainedModel
from .pipeline import train_full_pipeline
from .utils.io import load_joblib
from .utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def _cmd_train(args: argparse.Namespace) -> int:
    artifact = train_full_pipeline(
        use_optuna=not args.no_optuna,
        optuna_trials=args.trials,
        optuna_timeout=args.timeout,
        models=args.models,
        use_openml=not args.no_openml,
        synthetic_rows=args.synthetic_rows,
        compute_shap=not args.no_shap,
    )
    print(json.dumps(artifact.test_metrics, indent=2, default=float))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    artifact: TrainedModel = load_joblib(args.model)
    if args.csv:
        df = pd.read_csv(args.csv)
        for col in ALL_FEATURES:
            if col not in df.columns:
                raise SystemExit(f"Input CSV missing column: {col}")
        proba = artifact.predict_proba(df[ALL_FEATURES])
        out = df.copy()
        out["default_probability"] = proba
        out["risk_band"] = [BUSINESS_RULES.risk_band(p) for p in proba]
        out["decision"] = [BUSINESS_RULES.decision(p) for p in proba]
        target = args.output or Path(args.csv).with_suffix(".scored.csv")
        out.to_csv(target, index=False)
        print(f"Wrote {len(out)} scored rows to {target}")
    else:
        applicant: dict[str, object] = {}
        for f in ALL_FEATURES:
            applicant[f] = getattr(args, f, None)
        result = artifact.score_applicant(applicant)
        print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="creditguard", description="CreditGuard AI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train all models and persist the best one.")
    train_p.add_argument("--no-optuna", action="store_true", help="Skip hyperparameter tuning.")
    train_p.add_argument("--trials", type=int, default=TRAINING_CONFIG.optuna_trials)
    train_p.add_argument("--timeout", type=int, default=TRAINING_CONFIG.optuna_timeout_seconds)
    train_p.add_argument("--models", nargs="+", default=None)
    train_p.add_argument("--no-openml", action="store_true", help="Skip OpenML download.")
    train_p.add_argument("--synthetic-rows", type=int, default=None)
    train_p.add_argument("--no-shap", action="store_true", help="Skip SHAP computation.")
    train_p.set_defaults(func=_cmd_train)

    pred_p = sub.add_parser("predict", help="Score a CSV or a single applicant.")
    pred_p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    pred_p.add_argument("--csv", help="Path to applicant CSV (batch mode).")
    pred_p.add_argument("--output", help="Where to write the scored CSV.")
    for f in ALL_FEATURES:
        pred_p.add_argument(f"--{f.replace('_', '-')}", dest=f, default=None)
    pred_p.set_defaults(func=_cmd_predict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
