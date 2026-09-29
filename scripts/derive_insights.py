"""
Stakeholder-facing insights: for EVERY feature (ordered by SHAP importance),
bin its real values and report row count, real win rate (Won and Lost
mixed), and mean SHAP per bin. A bin is a trustworthy finding when win rate
vs. baseline and mean SHAP point the same way; when they disagree, suspect
confounding. Bins under min_confident_bin_size are flagged LOW CONFIDENCE.

Output: reports/<model>_insights.csv (+ the log)

Usage:  python scripts/derive_insights.py --model suspect|prospect_plus|all
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import load_all
from src.logging_config import setup_logging
from src.models.explain import compute_shap_values, feature_dependence_bins, rank_features_by_shap
from src.models.train import apply_saved_categories, load_artifact, load_model_dataset, weights_or_none
from src.utils import models_from_arg, parse_model_args

logger = setup_logging("derive_insights")


def insights(model_name: str, data_cfg: dict, features_cfg: dict) -> None:
    min_n = features_cfg["thresholds"]["min_confident_bin_size"]
    artifact = load_artifact(data_cfg["output"]["models_dir"], model_name)
    features = artifact["feature_columns"]
    df = load_model_dataset(data_cfg["output"]["processed_dir"], model_name, "train")
    X = apply_saved_categories(df[features], artifact["categories"])
    y, weights = df[features_cfg["target"]], weights_or_none(df)
    shap_values = compute_shap_values(artifact["model"], X)

    baseline = np.average(y, weights=weights) * 100
    logger.info("=" * 70)
    logger.info("%s — baseline win rate %.1f%% (compare every bin against this)", model_name.upper(), baseline)

    rows = []
    for feature in rank_features_by_shap(shap_values, features, weights)["feature"]:
        table = feature_dependence_bins(shap_values, X, y, feature, weights, min_n)
        logger.info("-" * 70)
        logger.info("DEPENDENCE: %s", feature)
        for _, r in table.iterrows():
            agrees = (r["win_rate_pct"] > baseline) == (r["mean_shap"] > 0)
            flag = "  <- LOW CONFIDENCE" if r["low_confidence"] else ("" if agrees else "  <- win rate and SHAP disagree")
            logger.info("  %-28s n=%-6d win_rate=%5.1f%%  mean_shap=%+.4f%s",
                        r["bin"], r["n"], r["win_rate_pct"], r["mean_shap"], flag)
        rows.append(table.assign(feature=feature, baseline_win_rate_pct=round(baseline, 1)))

    out = Path(data_cfg["output"]["reports_dir"]) / f"{model_name}_insights.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows)[["feature", "bin", "n", "win_rate_pct", "baseline_win_rate_pct", "mean_shap", "low_confidence"]] \
        .to_csv(out, index=False)
    logger.info("Saved %s", out)


def main() -> None:
    args = parse_model_args("Bin-level insights", allow_all=True)
    data_cfg, features_cfg, _ = load_all()
    for m in models_from_arg(args.model):
        insights(m, data_cfg, features_cfg)


if __name__ == "__main__":
    main()
