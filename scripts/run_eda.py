"""
Diagnostic health check on a model's configured features (removes nothing).
Reports events-per-feature, sparsity, missingness, and redundant pairs.

Usage:  python scripts/run_eda.py --model suspect|prospect_plus|all
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_all
from src.features.preprocessing import configured_features
from src.logging_config import setup_logging
from src.models.train import load_model_dataset
from src.utils import models_from_arg, parse_model_args

logger = setup_logging("run_eda")
SPARSITY, MISSINGNESS, CORRELATION = 0.98, 0.50, 0.90


def eda(model_name: str, data_cfg: dict, features_cfg: dict) -> None:
    df = load_model_dataset(data_cfg["output"]["processed_dir"], model_name, "train")
    features = [c for c in configured_features(model_name, features_cfg) if c in df.columns]
    wins = df.loc[df["LABEL_WON"] == 1, "OPPORTUNITY_ID"].nunique()
    logger.info("=" * 70)
    logger.info("%s: %d rows, %d distinct opps, %d features, %d won opps -> %.2f events per feature",
                model_name.upper(), len(df), df["OPPORTUNITY_ID"].nunique(), len(features), wins,
                wins / max(len(features), 1))

    numeric = df[features].select_dtypes(include="number")
    sparse = [(c, (numeric[c] == 0).mean()) for c in numeric.columns if (numeric[c] == 0).mean() >= SPARSITY]
    logger.info("Sparse (>= %.0f%% zero): %d", SPARSITY * 100, len(sparse))
    for c, pct in sparse:
        logger.info("  %-40s %.1f%% zero", c, pct * 100)

    missing = [(c, df[c].isna().mean()) for c in features if df[c].isna().mean() >= MISSINGNESS]
    logger.info("High missingness (>= %.0f%% NaN): %d", MISSINGNESS * 100, len(missing))
    for c, pct in missing:
        logger.info("  %-40s %.1f%% NaN", c, pct * 100)

    corr = numeric.corr().abs()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
             if pd.notna(corr.loc[a, b]) and corr.loc[a, b] >= CORRELATION]
    logger.info("Redundant pairs (|r| >= %.2f): %d", CORRELATION, len(pairs))
    for a, b, r in sorted(pairs, key=lambda x: -x[2])[:30]:
        logger.info("  %-35s <-> %-35s r=%.3f", a, b, r)

    for c in features_cfg["categorical_columns"]:
        if c in features:
            logger.info("Cardinality %-20s %d distinct values", c, df[c].nunique())


def main() -> None:
    args = parse_model_args("EDA diagnostics", allow_all=True)
    data_cfg, features_cfg, _ = load_all()
    for m in models_from_arg(args.model):
        eda(m, data_cfg, features_cfg)


if __name__ == "__main__":
    main()
