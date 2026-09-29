"""
Summary of the latest scoring run (reads reports/open_opp_scores.csv; no re-scoring):
bucketed score distribution, by model, by stage (sanity check: should rise
with stage), counts of flagged opps, and the top 25 by win probability.

Usage:  python scripts/summarize_scores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_all
from src.logging_config import setup_logging

logger = setup_logging("summarize_scores")
BINS = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
LABELS = ["<1%", "1-5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-100%"]


def main() -> None:
    data_cfg, _, _ = load_all()
    reports = Path(data_cfg["output"]["reports_dir"])
    df = pd.read_csv(reports / "open_opp_scores.csv")
    alerts_path = reports / "owner_reassignment_alerts.csv"
    n_alerts = len(pd.read_csv(alerts_path)) if alerts_path.exists() else 0
    logger.info("%d scored open opps | %d skipped for owner reassignment", len(df), n_alerts)

    df["bucket"] = pd.cut(df["WIN_PROBABILITY"], bins=BINS, labels=LABELS, include_lowest=True)
    logger.info("=" * 70)
    logger.info("SCORE DISTRIBUTION")
    for b, n in df["bucket"].value_counts().reindex(LABELS).items():
        logger.info("  %-8s %6d (%.1f%%)", b, n, n / len(df) * 100)

    logger.info("=" * 70)
    logger.info("BY MODEL")
    for m, g in df.groupby("MODEL_USED"):
        logger.info("  %-14s n=%-6d mean=%.3f  predicted win=%d", m, len(g), g["WIN_PROBABILITY"].mean(),
                    int((g["PREDICTED_OUTCOME"] == "Win").sum()))

    logger.info("=" * 70)
    logger.info("BY CURRENT STAGE (sanity check: should rise with stage)")
    stage = df.groupby("CURRENT_STAGENAME")["WIN_PROBABILITY"].agg(["count", "mean"]).sort_values("mean")
    for s, r in stage.iterrows():
        logger.info("  %-30s n=%-6d mean=%.4f", s, int(r["count"]), r["mean"])

    logger.info("=" * 70)
    logger.info("FLAGS: Prospect Desk (untriaged) %d | Dead Queue %d",
                int(df["IS_PROSPECT_DESK"].fillna(0).sum()), int(df["IS_DEAD_QUEUE"].fillna(0).sum()))

    top = df.head(25)
    top.to_csv(reports / "top_opps_priority_list.csv", index=False)
    logger.info("=" * 70)
    logger.info("TOP 25 BY WIN PROBABILITY")
    for _, r in top.iterrows():
        logger.info("  %-20s %-28s %-6s $%10.0f  %5.1f%%  [%s]", r["OPPORTUNITY_ID"], r["CURRENT_STAGENAME"],
                    r["VERTICAL"], r["AMOUNT"] if pd.notna(r["AMOUNT"]) else 0, r["WIN_PROBABILITY"] * 100, r["MODEL_USED"])


if __name__ == "__main__":
    main()
