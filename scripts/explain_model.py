"""
Global SHAP explanation for one model: weighted importance ranking, the
call-vs-meeting comparison (prospect_plus only, since Suspect has no
activity features), and direction per feature.

Outputs: reports/<model>_shap_ranking.csv, reports/<model>_shap_summary.png

Usage:  python scripts/explain_model.py --model suspect|prospect_plus|all
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_all
from src.logging_config import setup_logging
from src.models.explain import call_vs_meeting_summary, compute_shap_values, directional_summary, rank_features_by_shap
from src.models.train import apply_saved_categories, load_artifact, load_model_dataset, weights_or_none
from src.utils import models_from_arg, parse_model_args

logger = setup_logging("explain_model")


def explain(model_name: str, data_cfg: dict) -> None:
    reports = Path(data_cfg["output"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    artifact = load_artifact(data_cfg["output"]["models_dir"], model_name)
    features = artifact["feature_columns"]
    df = load_model_dataset(data_cfg["output"]["processed_dir"], model_name, "train")
    X = apply_saved_categories(df[features], artifact["categories"])
    weights = weights_or_none(df)

    logger.info("Computing SHAP for %s on %d train rows...", model_name, len(X))
    shap_values = compute_shap_values(artifact["model"], X)

    ranking = rank_features_by_shap(shap_values, features, weights)
    ranking.to_csv(reports / f"{model_name}_shap_ranking.csv", index=False)
    logger.info("=" * 70)
    logger.info("%s SHAP IMPORTANCE RANKING", model_name.upper())
    for _, r in ranking.iterrows():
        logger.info("  %-40s %.5f  (%.1f%%)", r["feature"], r["mean_abs_shap"], r["pct_of_total"])

    if model_name == "prospect_plus":
        cvm = call_vs_meeting_summary(ranking)
        ratio = cvm["meet_total_importance"] / cvm["call_total_importance"] if cvm["call_total_importance"] else float("inf")
        logger.info("=" * 70)
        logger.info("CALL vs MEETING: calls %.4f | meetings %.4f -> meetings/calls = %.2fx",
                    cvm["call_total_importance"], cvm["meet_total_importance"], ratio)

    logger.info("=" * 70)
    logger.info("DIRECTION (weighted mean signed SHAP; skewed toward LOST by the ~88%% loss base rate)")
    for _, r in directional_summary(shap_values, features, weights).iterrows():
        logger.info("  %-40s %+.5f  (%s)", r["feature"], r["mean_signed_shap"], r["direction"])

    import shap
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(reports / f"{model_name}_shap_summary.png", dpi=150)
    plt.close()
    logger.info("Saved %s and %s", reports / f"{model_name}_shap_ranking.csv", reports / f"{model_name}_shap_summary.png")


def main() -> None:
    args = parse_model_args("Global SHAP explanation", allow_all=True)
    data_cfg, _, _ = load_all()
    for m in models_from_arg(args.model):
        explain(m, data_cfg)


if __name__ == "__main__":
    main()
