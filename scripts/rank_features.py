"""
Stage 2 (diagnostic only): mutual information of each selected feature with
LABEL_WON. Univariate — can't see interactions; SHAP (explain_model.py) is
the real arbiter. Removes nothing.

Usage:  python scripts/rank_features.py --model suspect|prospect_plus|all
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.config import load_all
from src.logging_config import setup_logging
from src.models.train import load_model_dataset
from src.utils import load_feature_list, models_from_arg, parse_model_args

logger = setup_logging("rank_features")


def rank(model_name: str, data_cfg: dict, features_cfg: dict) -> None:
    df = load_model_dataset(data_cfg["output"]["processed_dir"], model_name, "train")
    features = load_feature_list(model_name)
    X = df[features].copy()
    discrete = []
    for c in features:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c])[0]
            discrete.append(True)
        else:
            discrete.append(False)
    X = X.fillna(-1.0)  # diagnostic-only sentinel
    mi = mutual_info_classif(X, df["LABEL_WON"], discrete_features=discrete, random_state=42)
    ranking = pd.DataFrame({"feature": features, "mutual_info": mi}).sort_values("mutual_info", ascending=False)

    out = Path(data_cfg["output"]["reports_dir"]) / f"{model_name}_mutual_info_ranking.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out, index=False)
    logger.info("=" * 70)
    logger.info("%s MUTUAL INFORMATION RANKING", model_name.upper())
    for _, r in ranking.iterrows():
        logger.info("  %-40s %.5f", r["feature"], r["mutual_info"])
    logger.info("Saved to %s", out)


def main() -> None:
    args = parse_model_args("Mutual information ranking", allow_all=True)
    data_cfg, features_cfg, _ = load_all()
    for m in models_from_arg(args.model):
        rank(m, data_cfg, features_cfg)


if __name__ == "__main__":
    main()
