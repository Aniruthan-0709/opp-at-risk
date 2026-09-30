"""
Scores every currently open opportunity from data/input/open_opps.csv.

  1. Owner gate: inactive human owner -> not scored, written to
     reports/owner_reassignment_alerts.csv for the sales manager.
     Re-scored automatically on a later run once reassigned.
  2. Route: MAX_STAGE_REACHED == 1 -> suspect model (DAYS_IN_SUSPECT = days open,
     clipped to the trained range); >= 2 -> prospect_plus model.
  3. Score (WIN_PROBABILITY is calibrated; RAW_SCORE kept for transparency), and flag Prospect Desk (untriaged) and Dead Queue opps.
  4. Append every score to reports/prediction_log.csv (date + model used),
     so predictions can be checked against real outcomes once deals close.

Outputs:
  reports/open_opp_scores.csv                  one row per scored opp
  reports/owner_reassignment_alerts.csv        opps skipped by the owner gate
  data/processed/open_<model>_scored.csv       features + scores, used by reasons.py
  reports/prediction_log.csv                   appended every run

Usage:  python scripts/score_open_opps.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_all
from src.features.preprocessing import prepare_open
from src.features.snapshots import suspect_scoring_age
from src.logging_config import setup_logging
from src.models.train import MODEL_NAMES, load_artifact, predict_proba
from src.serving.routing import owner_gate_mask, route

logger = setup_logging("score_open_opps")

OUTPUT_COLUMNS = [
    "OPPORTUNITY_ID", "ACCOUNTID", "VERTICAL", "INITIATIVE__C", "CURRENT_STAGENAME", "MAX_STAGE_REACHED",
    "AMOUNT", "DAYS_OPEN", "OWNER_NAME", "MODEL_USED", "WIN_PROBABILITY", "RAW_SCORE", "PREDICTED_OUTCOME",
    "IS_PROSPECT_DESK", "IS_DEAD_QUEUE",
]
ALERT_COLUMNS = [
    "OPPORTUNITY_ID", "ACCOUNTID", "VERTICAL", "INITIATIVE__C", "CURRENT_STAGENAME",
    "OWNER_NAME", "AMOUNT", "DAYS_OPEN",
]


def main() -> None:
    data_cfg, features_cfg, _ = load_all()
    out = data_cfg["output"]
    reports, processed = Path(out["reports_dir"]), Path(out["processed_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    threshold = features_cfg["thresholds"]["win_probability"]
    max_age = features_cfg["suspect"]["max_snapshot_age_days"]

    raw = pd.read_csv(data_cfg["input"]["open_opps_csv"])
    opps = prepare_open(raw, features_cfg)

    # 1. Owner gate
    gated = owner_gate_mask(opps)
    alerts = opps.loc[gated, ALERT_COLUMNS].assign(
        ALERT="Owner is an inactive user - reassign; no prediction until an active owner is assigned",
        ALERT_DATE=date.today().isoformat(),
    )
    alerts.to_csv(reports / "owner_reassignment_alerts.csv", index=False)
    logger.info("Owner gate: %d of %d open opps need reassignment (not scored) -> %s",
                len(alerts), len(opps), reports / "owner_reassignment_alerts.csv")

    # 2. Route + 3. Score
    scoreable = opps.loc[~gated].copy()
    scoreable["MODEL_USED"] = route(scoreable)
    scoreable["DAYS_IN_SUSPECT"] = suspect_scoring_age(scoreable["DAYS_OPEN"], max_age)
    parts = []
    for m in MODEL_NAMES:
        part = scoreable[scoreable["MODEL_USED"] == m].copy()
        if part.empty:
            logger.info("No open opps routed to %s", m)
            (processed / f"open_{m}_scored.csv").unlink(missing_ok=True)  # don't leave a stale file for reasons.py
            continue
        artifact = load_artifact(out["models_dir"], m)
        part["RAW_SCORE"] = predict_proba(artifact, part, calibrated=False)
        part["WIN_PROBABILITY"] = predict_proba(artifact, part)   # calibrated
        part["MODEL_TRAINED_AT"] = artifact["trained_at"]
        part.to_csv(processed / f"open_{m}_scored.csv", index=False)
        logger.info("%s: scored %d opps (mean win probability %.3f)", m, len(part), part["WIN_PROBABILITY"].mean())
        parts.append(part)

    if not parts:
        logger.warning("No opps were scored (all gated or input empty).")
        return
    scored = pd.concat(parts, ignore_index=True)
    scored["PREDICTED_OUTCOME"] = (scored["WIN_PROBABILITY"] > threshold).map({True: "Win", False: "Lose"})
    scored = scored.sort_values("WIN_PROBABILITY", ascending=False)
    scored[OUTPUT_COLUMNS].to_csv(reports / "open_opp_scores.csv", index=False)
    logger.info("Saved %d scores to %s | Prospect Desk (untriaged): %d | Dead Queue: %d",
                len(scored), reports / "open_opp_scores.csv",
                int(scored["IS_PROSPECT_DESK"].fillna(0).sum()), int(scored["IS_DEAD_QUEUE"].fillna(0).sum()))

    # 4. Prediction log (append)
    log_path = Path(out["prediction_log"])
    log_rows = scored[["OPPORTUNITY_ID", "CURRENT_STAGENAME", "MAX_STAGE_REACHED", "MODEL_USED",
                       "WIN_PROBABILITY", "RAW_SCORE", "MODEL_TRAINED_AT"]].assign(SCORED_AT=date.today().isoformat())
    log_rows.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    logger.info("Appended %d rows to %s", len(log_rows), log_path)


if __name__ == "__main__":
    main()