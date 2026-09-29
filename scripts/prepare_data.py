"""
Builds every processed dataset both models need from data/input/closed_opps.csv.

  1. Clean + derive features (src/features/preprocessing.py)
  2. Split by TRUE_CLOSE_DATE: pool (<= train_cutoff) vs. out-of-time eval window
  3. Split the pool into train/test at the OPPORTUNITY level, stratified by
     FY quarter x outcome type (Won / Lost / AutoClosed)
  4. For each of train / test / eval:
       suspect_*.csv       -> every deal, expanded into Suspect snapshots
       prospect_plus_*.csv -> deals that reached Prospect+ (MAX_STAGE_REACHED >= 2)

Usage:  python scripts/prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_all
from src.features.preprocessing import prepare_closed
from src.features.snapshots import build_suspect_snapshots
from src.labeling.split import stratified_opp_split
from src.logging_config import setup_logging

logger = setup_logging("prepare_data")


def summarize(name: str, df: pd.DataFrame) -> None:
    if "SAMPLE_WEIGHT" in df.columns:
        n_opps = df["OPPORTUNITY_ID"].nunique()
        win_rate = (df["LABEL_WON"] * df["SAMPLE_WEIGHT"]).sum() / df["SAMPLE_WEIGHT"].sum()
        logger.info("  %-22s %6d rows from %5d opps | win rate %.1f%%", name, len(df), n_opps, win_rate * 100)
    else:
        logger.info("  %-22s %6d rows              | win rate %.1f%%", name, len(df), df["LABEL_WON"].mean() * 100)


def main() -> None:
    data_cfg, features_cfg, _ = load_all()
    windows, split_cfg, sus_cfg = data_cfg["windows"], features_cfg["split"], features_cfg["suspect"]
    out_dir = Path(data_cfg["output"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(data_cfg["input"]["closed_opps_csv"])
    logger.info("Loaded %s: %d rows", data_cfg["input"]["closed_opps_csv"], len(raw))
    closed = prepare_closed(raw, features_cfg)

    cutoff = pd.Timestamp(windows["train_cutoff"])
    eval_start, eval_end = pd.Timestamp(windows["eval_start"]), pd.Timestamp(windows["eval_end"])
    pool = closed[closed["TRUE_CLOSE_DATE"] <= cutoff]
    eval_df = closed[(closed["TRUE_CLOSE_DATE"] >= eval_start) & (closed["TRUE_CLOSE_DATE"] <= eval_end)]
    logger.info("Train/test pool (closed <= %s): %d opps | Eval window (%s..%s): %d opps",
                cutoff.date(), len(pool), eval_start.date(), eval_end.date(), len(eval_df))
    if eval_df.empty:
        logger.warning("Eval window is empty — check windows in configs/data.yaml.")

    train_df, test_df = stratified_opp_split(
        pool, split_cfg["test_fraction"], split_cfg["random_state"], split_cfg["min_stratum_size"],
    )

    logger.info("=" * 70)
    logger.info("Writing processed datasets to %s", out_dir)
    for split_name, part in [("train", train_df), ("test", test_df), ("eval", eval_df)]:
        suspect = build_suspect_snapshots(part, sus_cfg["snapshot_interval_days"], sus_cfg["max_snapshot_age_days"])
        prospect_plus = part[part["MAX_STAGE_REACHED"] >= 2]
        suspect.to_csv(out_dir / f"suspect_{split_name}.csv", index=False)
        prospect_plus.to_csv(out_dir / f"prospect_plus_{split_name}.csv", index=False)
        summarize(f"suspect_{split_name}", suspect)
        summarize(f"prospect_plus_{split_name}", prospect_plus)


if __name__ == "__main__":
    main()
