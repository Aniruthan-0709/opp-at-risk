"""
Suspect-model training snapshots.

A single day-one snapshot per deal would give DAYS_IN_SUSPECT = 0 for every
training row, so the model could never learn that "still in Suspect after
90 days" matters. Instead each historical closed deal contributes one row at
age 0, interval, 2*interval, ... for as long as it was genuinely still
sitting in Suspect (age < DAYS_IN_SUSPECT_TOTAL), every row labelled with
the deal's EVENTUAL outcome.

Rules:
  - Age 0 is always included, so every deal is represented once.
  - Ages at or past the day the deal left Suspect / closed are not created
    (that state never existed).
  - SAMPLE_WEIGHT = 1 / (snapshots for that deal), so long-lived deals don't
    outweigh short ones. Weights sum to the number of deals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def snapshot_counts(days_in_suspect_total: pd.Series, interval: int, max_age: int) -> np.ndarray:
    total = pd.to_numeric(days_in_suspect_total, errors="coerce").fillna(0).clip(lower=0).to_numpy()
    k_by_time = np.ceil(total / interval).astype(int) - 1      # largest k with k*interval < total
    k_max = np.minimum(k_by_time, max_age // interval)
    return 1 + np.maximum(k_max, 0)


def build_suspect_snapshots(df: pd.DataFrame, interval: int, max_age: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)  # cumcount below relies on one unique index label per deal
    counts = snapshot_counts(df["DAYS_IN_SUSPECT_TOTAL"], interval, max_age)
    out = df.loc[df.index.repeat(counts)].copy()
    ages = out.groupby(level=0).cumcount().to_numpy() * interval
    out["SNAPSHOT_AGE_DAYS"] = ages
    out["DAYS_IN_SUSPECT"] = ages
    out["SAMPLE_WEIGHT"] = 1.0 / np.repeat(counts, counts)
    return out.reset_index(drop=True)


def suspect_scoring_age(days_open: pd.Series, max_age: int) -> pd.Series:
    """Live DAYS_IN_SUSPECT for an open opp that has never left Suspect, clipped to the trained range."""
    return pd.to_numeric(days_open, errors="coerce").fillna(0).clip(lower=0, upper=max_age)
