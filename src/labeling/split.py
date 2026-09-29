"""
Train/test split for the closed-deal pool (everything closed on/before the
train cutoff).

Split at the OPPORTUNITY level before any Suspect snapshots are generated,
so all snapshots of one deal land on the same side, and both models share
the same test deals.

Stratified by FY quarter x outcome type (Won / Lost / AutoClosed), so each
quarter's mix of wins, genuine losses, and auto-closed losses is preserved
in both train and test. Strata smaller than min_stratum_size are merged
into an outcome-only stratum rather than failing the split.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def build_strata(df: pd.DataFrame, min_stratum_size: int) -> pd.Series:
    strata = df["FY_QUARTER"] + "_" + df["OUTCOME_TYPE"]
    sizes = strata.map(strata.value_counts())
    merged = strata.where(sizes >= min_stratum_size, "ANY_" + df["OUTCOME_TYPE"])
    n_merged = int((sizes < min_stratum_size).sum())
    if n_merged:
        logger.info("Merged %d rows from small quarter strata into outcome-only strata", n_merged)
    # Final guard: any stratum still under 2 rows can't be stratified at all.
    final_sizes = merged.map(merged.value_counts())
    return merged.where(final_sizes >= 2, "ANY")


def stratified_opp_split(df: pd.DataFrame, test_fraction: float, random_state: int,
                         min_stratum_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = build_strata(df, min_stratum_size)
    train_df, test_df = train_test_split(
        df, test_size=test_fraction, random_state=random_state, stratify=strata,
    )
    logger.info("Split: %d train opps / %d test opps", len(train_df), len(test_df))
    for name, part in [("train", train_df), ("test", test_df)]:
        mix = part["OUTCOME_TYPE"].value_counts(normalize=True).mul(100).round(1).to_dict()
        logger.info("  %-5s outcome mix %%: %s", name, mix)
    return train_df.copy(), test_df.copy()
