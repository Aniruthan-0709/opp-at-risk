"""
Stage 1 feature selection (deterministic): starts from the model's
configured feature list and drops only true zero-variance columns.
Writes configs/<model>_feature_list.json, which every later step reads.

Usage:  python scripts/select_features.py --model suspect|prospect_plus|all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_all
from src.features.preprocessing import configured_features
from src.logging_config import setup_logging
from src.models.train import load_model_dataset
from src.utils import feature_list_path, models_from_arg, parse_model_args

logger = setup_logging("select_features")


def select(model_name: str, data_cfg: dict, features_cfg: dict) -> None:
    df = load_model_dataset(data_cfg["output"]["processed_dir"], model_name, "train")
    configured = configured_features(model_name, features_cfg)
    missing = [c for c in configured if c not in df.columns]
    if missing:
        raise ValueError(f"{model_name}: configured features missing from processed data: {missing}")
    zero_var = [c for c in configured if df[c].nunique(dropna=False) <= 1]
    selected = [c for c in configured if c not in zero_var]
    leaked = set(selected) & set(features_cfg["metadata_columns"])
    if leaked:
        raise ValueError(f"{model_name}: metadata columns must never be features: {sorted(leaked)}")

    feature_list_path(model_name).write_text(json.dumps({"feature_columns": selected}, indent=2))
    logger.info("%s: %d configured -> %d selected (dropped zero-variance: %s) -> %s",
                model_name, len(configured), len(selected), zero_var or "none", feature_list_path(model_name))


def main() -> None:
    args = parse_model_args("Stage 1 feature selection", allow_all=True)
    data_cfg, features_cfg, _ = load_all()
    for m in models_from_arg(args.model):
        select(m, data_cfg, features_cfg)


if __name__ == "__main__":
    main()
