"""
WHERE risk and strength sit: every Vertical x Initiative among scored open
opps, with average win probability and % predicted to lose. Every
initiative is shown (thin cells flagged LOW CONFIDENCE, never hidden). The
system-wide top-15 riskiest / strongest lists use reliable cells only.

Reads reports/open_opp_scores.csv (no re-scoring).
Output: reports/segment_risk_summary.csv

Usage:  python scripts/segment_risk_summary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_all
from src.logging_config import setup_logging

logger = setup_logging("segment_risk_summary")
TOP_N = 15


def log_row(r, with_vertical: bool = False) -> None:
    flag = "  <- LOW CONFIDENCE" if r["low_confidence"] else ""
    prefix = f"{r['VERTICAL']:<8} " if with_vertical else "  "
    logger.info("%s%-45s n=%-6d avg_win_prob=%5.1f%%  predicted_lost=%5.1f%%  suspect_model=%3.0f%%%s",
                prefix, r["INITIATIVE__C"], r["n"], r["avg_win_prob"] * 100, r["pct_predicted_lost"] * 100,
                r["pct_suspect_model"] * 100, flag)


def main() -> None:
    data_cfg, features_cfg, _ = load_all()
    min_n = features_cfg["thresholds"]["min_confident_bin_size"]
    reports = Path(data_cfg["output"]["reports_dir"])
    df = pd.read_csv(reports / "open_opp_scores.csv")
    df["INITIATIVE__C"] = df["INITIATIVE__C"].fillna("(no initiative)")

    g = df.groupby(["VERTICAL", "INITIATIVE__C"]).agg(
        n=("OPPORTUNITY_ID", "size"),
        avg_win_prob=("WIN_PROBABILITY", "mean"),
        pct_predicted_lost=("PREDICTED_OUTCOME", lambda s: (s == "Lose").mean()),
        pct_suspect_model=("MODEL_USED", lambda s: (s == "suspect").mean()),
    ).reset_index()
    g["low_confidence"] = g["n"] < min_n
    g.to_csv(reports / "segment_risk_summary.csv", index=False)

    logger.info("=" * 70)
    logger.info("BY VERTICAL, every initiative, riskiest first")
    for vertical, vg in g.groupby("VERTICAL"):
        logger.info("-" * 70)
        logger.info("VERTICAL: %s", vertical)
        for _, r in vg.sort_values("avg_win_prob").iterrows():
            log_row(r)

    reliable = g[~g["low_confidence"]]
    for title, ascending in [("RISKIEST", True), ("STRONGEST", False)]:
        logger.info("=" * 70)
        logger.info("TOP %d %s INITIATIVES SYSTEM-WIDE (n >= %d)", TOP_N, title, min_n)
        for _, r in reliable.sort_values("avg_win_prob", ascending=ascending).head(TOP_N).iterrows():
            log_row(r, with_vertical=True)
    logger.info("Saved %s. For WHY, see reasons.py.", reports / "segment_risk_summary.csv")


if __name__ == "__main__":
    main()
