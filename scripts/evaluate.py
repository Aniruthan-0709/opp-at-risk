"""
Evaluates both saved models. Nothing is trained here.

  1. Each model on its holdout test set and on the out-of-time eval window (Q3)
  2. Combined routed system on eval deals: prospect_plus for deals that reached
     Prospect+, suspect (day-0 snapshot) for deals that never left Suspect
  3. Calibration: predicted vs. actual win rate per probability bin, per model,
     for RAW and CALIBRATED scores. After calibration "10%" should mean ~10%.
     Ranking metrics (AUC-PR, ROC-AUC) should be the same for both; only the
     percentages, and so precision/recall at the 50% cut, change.
  4. Average predicted win probability by outcome type (Won / Lost / AutoClosed)

Outputs: reports/evaluation_summary.json, reports/<model>_calibration_{raw,calibrated}.csv

Usage:  python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import load_all
from src.logging_config import setup_logging
from src.models.train import (
    MODEL_NAMES, calibration_table, compute_metrics, load_artifact, load_model_dataset, predict_proba, weights_or_none,
)

logger = setup_logging("evaluate")


def main() -> None:
    data_cfg, features_cfg, _ = load_all()
    processed, reports = data_cfg["output"]["processed_dir"], Path(data_cfg["output"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    target, threshold = features_cfg["target"], features_cfg["thresholds"]["win_probability"]
    summary: dict = {}
    eval_scored: dict[str, pd.DataFrame] = {}

    for m in MODEL_NAMES:
        artifact = load_artifact(data_cfg["output"]["models_dir"], m)
        summary[m] = {}
        for split in ["test", "eval"]:
            df = load_model_dataset(processed, m, split)
            if df.empty:
                logger.warning("%s %s set is empty — skipping", m, split)
                continue
            w = weights_or_none(df)
            raw = predict_proba(artifact, df, calibrated=False)
            proba = predict_proba(artifact, df)
            summary[m][split] = {"raw": compute_metrics(df[target], raw, threshold, w),
                                 "calibrated": compute_metrics(df[target], proba, threshold, w)}
            note = " (calibrator was fitted on this set)" if split == "test" else ""
            logger.info("%-14s %-5s raw        %s", m, split, summary[m][split]["raw"])
            logger.info("%-14s %-5s calibrated %s%s", m, split, summary[m][split]["calibrated"], note)
            if split == "eval":
                eval_scored[m] = df.assign(WIN_PROBABILITY=proba)
                for label, scores in [("raw", raw), ("calibrated", proba)]:
                    cal = calibration_table(df[target], scores, w)
                    cal.to_csv(reports / f"{m}_calibration_{label}.csv", index=False)
                    logger.info("%s %s calibration on eval (predicted vs actual):\n%s",
                                m, label.upper(), cal.to_string(index=False))

    # Combined routed system on the out-of-time window
    if {"suspect", "prospect_plus"} <= eval_scored.keys():
        pp = eval_scored["prospect_plus"]
        sus = eval_scored["suspect"]
        sus_day0 = sus[(sus["SNAPSHOT_AGE_DAYS"] == 0) & (sus["MAX_STAGE_REACHED"] <= 1)]
        combined = pd.concat([pp.assign(MODEL_USED="prospect_plus"), sus_day0.assign(MODEL_USED="suspect")])
        summary["combined_routed_eval"] = compute_metrics(combined[target], combined["WIN_PROBABILITY"].to_numpy(), threshold)
        logger.info("=" * 70)
        logger.info("COMBINED routed system (eval window, calibrated): %s", summary["combined_routed_eval"])
        logger.info("  routed to prospect_plus: %d deals | suspect: %d deals", len(pp), len(sus_day0))

        by_outcome = combined.groupby("OUTCOME_TYPE")["WIN_PROBABILITY"].agg(["count", "mean"]).round(4)
        summary["mean_win_probability_by_outcome"] = by_outcome["mean"].to_dict()
        logger.info("Average predicted win probability by actual outcome:\n%s", by_outcome.to_string())

    out = reports / "evaluation_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()