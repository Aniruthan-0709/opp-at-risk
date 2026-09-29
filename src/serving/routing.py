"""
Live-scoring rules. Deterministic business rules, never model features.

1. Owner gate: an open opp owned by an INACTIVE HUMAN user is not scored.
   Nobody is working it, so a score would rest on stale information; it is
   reported for reassignment instead and scored on a later run once it has
   an active owner. Queue owners (non-human users) are not gated.
2. Routing: MAX_STAGE_REACHED == 1 (never left Suspect) -> suspect model;
   >= 2 (reached Prospect or later at any point, even if it has since moved
   back to Suspect) -> prospect_plus model, so already-collected amount and
   activity data are never thrown away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def owner_gate_mask(df: pd.DataFrame) -> pd.Series:
    """True = needs reassignment (inactive human owner). Unknown owner info is not gated."""
    is_human = df["OWNER_IS_NON_HUMAN"].fillna(1) == 0
    is_inactive = df["OWNER_IS_ACTIVE"].fillna(1) == 0
    return is_human & is_inactive


def route(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(df["MAX_STAGE_REACHED"] <= 1, "suspect", "prospect_plus"), index=df.index)
