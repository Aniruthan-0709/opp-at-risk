"""
Shared preprocessing for closed (training/eval) and open (scoring) opps.

Also the single source of truth for WHICH features each model uses
(suspect_features / prospect_plus_features), so no script has to rebuild
the list itself.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ACTIVITY_SUBTYPES = ["TASK", "CADENCE", "LISTEMAIL", "EMAIL", "CALL"]
STAGE_NUMBERS = [1, 2, 3, 4, 5]

ACTIVITY_COLUMNS = [
    f"ACCT_{s}_COUNT_STAGE{n}" for s in ACTIVITY_SUBTYPES for n in STAGE_NUMBERS
] + [f"ACCT_MEET_COUNT_STAGE{n}" for n in STAGE_NUMBERS]

DERIVED_ACTIVITY_COLUMNS = (
    ["acct_total_activity", "acct_call_share", "acct_meet_share"]
    + [f"acct_call_share_stage{n}" for n in STAGE_NUMBERS]
    + [f"acct_meet_share_stage{n}" for n in STAGE_NUMBERS]
)

# Count-like columns where "missing" genuinely means zero (no contacts on
# the account, no activity in that window, no prior wins).
ZERO_FILL_COLUMNS = [
    "ACTIVE_CONTACT_COUNT", "PRIOR_ACCOUNT_WIN_COUNT", "IS_REMOTE_GEO",
    "JOBCAT_EXECUTIVE_COUNT", "JOBCAT_FINANCE_PROCUREMENT_COUNT", "JOBCAT_OPERATIONS_COUNT",
    "JOBCAT_IT_COUNT", "JOBCAT_PEOPLE_GOVERNANCE_COUNT", "JOBCAT_MARKETING_COUNT",
    "JOBCAT_VERTICAL_ROLE_COUNT", "JOBCAT_UNKNOWN_COUNT", "BACKWARD_MOVE_COUNT",
    *ACTIVITY_COLUMNS,
]
# Left as NaN on purpose (XGBoost handles missing natively):
#   HEADCOUNT               - unknown size is not "0 employees"
#   PCT_CONTACTS_CAN_CONTACT - no contacts at all is not "0% reachable"
#   share ratios            - no activity is not "0% calls"

CLOSED_REQUIRED = [
    "OPPORTUNITY_ID", "ACCOUNTID", "INITIATIVE__C", "TRUE_CYCLE_START", "TRUE_CLOSE_DATE",
    "TRUE_CLOSE_FY", "TRUE_CLOSE_QUARTER_NUM", "IS_AUTO_CLOSED", "DAYS_IN_SUSPECT_TOTAL", "LABEL_WON",
    "MAX_STAGE_REACHED",
]
OPEN_REQUIRED = [
    "OPPORTUNITY_ID", "ACCOUNTID", "INITIATIVE__C", "CURRENT_STAGENAME", "TRUE_CYCLE_START",
    "DAYS_OPEN", "OWNER_IS_ACTIVE", "OWNER_IS_NON_HUMAN", "IS_PROSPECT_DESK", "IS_DEAD_QUEUE",
    "MAX_STAGE_REACHED",
]


# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
def suspect_features(features_cfg: dict) -> list[str]:
    return list(features_cfg["static_features"]) + list(features_cfg["suspect"]["extra_features"])


def prospect_plus_features(features_cfg: dict) -> list[str]:
    cols = list(features_cfg["static_features"]) + list(features_cfg["prospect_plus"]["extra_features"])
    if features_cfg["prospect_plus"].get("include_activity_features", True):
        cols += ACTIVITY_COLUMNS + DERIVED_ACTIVITY_COLUMNS
    return cols


def configured_features(model_name: str, features_cfg: dict) -> list[str]:
    if model_name == "suspect":
        return suspect_features(features_cfg)
    if model_name == "prospect_plus":
        return prospect_plus_features(features_cfg)
    raise ValueError(f"Unknown model '{model_name}' (expected 'suspect' or 'prospect_plus')")


# ---------------------------------------------------------------------------
# Validation + cleaning
# ---------------------------------------------------------------------------
def validate_columns(df: pd.DataFrame, required: list[str], source: str, features_cfg: dict) -> None:
    expected = set(required) | set(features_cfg["static_features"]) | set(ACTIVITY_COLUMNS)
    expected |= {"AMOUNT", "AMOUNT_BUCKET", "MAX_STAGE_REACHED", "BACKWARD_MOVE_COUNT", "CYCLE_LENGTH_DAYS"}
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing expected columns (check the SQL/CSV export): {missing}")
    n_dupes = df["OPPORTUNITY_ID"].duplicated().sum()
    if n_dupes:
        raise ValueError(f"{source} has {n_dupes} duplicate OPPORTUNITY_ID rows — check the SQL joins.")


def clean(df: pd.DataFrame, features_cfg: dict) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_FILL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in features_cfg["categorical_columns"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    return df


def add_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lifetime and per-stage call/meeting shares. NaN (not 0) when there's no activity."""
    df = df.copy()
    task_cols = [f"ACCT_{s}_COUNT_STAGE{n}" for s in ACTIVITY_SUBTYPES for n in STAGE_NUMBERS]
    meet_cols = [f"ACCT_MEET_COUNT_STAGE{n}" for n in STAGE_NUMBERS]
    df["acct_total_activity"] = df[task_cols + meet_cols].sum(axis=1)
    total = df["acct_total_activity"].replace(0, np.nan)
    df["acct_call_share"] = df[[f"ACCT_CALL_COUNT_STAGE{n}" for n in STAGE_NUMBERS]].sum(axis=1) / total
    df["acct_meet_share"] = df[meet_cols].sum(axis=1) / total
    for n in STAGE_NUMBERS:
        stage_total = df[[f"ACCT_{s}_COUNT_STAGE{n}" for s in ACTIVITY_SUBTYPES] + [f"ACCT_MEET_COUNT_STAGE{n}"]].sum(axis=1)
        stage_total = stage_total.replace(0, np.nan)
        df[f"acct_call_share_stage{n}"] = df[f"ACCT_CALL_COUNT_STAGE{n}"] / stage_total
        df[f"acct_meet_share_stage{n}"] = df[f"ACCT_MEET_COUNT_STAGE{n}"] / stage_total
    return df


def add_closed_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """OUTCOME_TYPE (Won / Lost / AutoClosed) and FY_QUARTER, used for stratification + reporting."""
    df = df.copy()
    df["LABEL_WON"] = df["LABEL_WON"].astype(int)
    df["IS_AUTO_CLOSED"] = pd.to_numeric(df["IS_AUTO_CLOSED"], errors="coerce").fillna(0).astype(int)
    df["OUTCOME_TYPE"] = np.where(
        df["LABEL_WON"] == 1, "Won", np.where(df["IS_AUTO_CLOSED"] == 1, "AutoClosed", "Lost")
    )
    df["FY_QUARTER"] = (
        "FY" + df["TRUE_CLOSE_FY"].astype(int).astype(str).str[-2:]
        + "Q" + df["TRUE_CLOSE_QUARTER_NUM"].astype(int).astype(str)
    )
    return df


def prepare_closed(raw: pd.DataFrame, features_cfg: dict) -> pd.DataFrame:
    validate_columns(raw, CLOSED_REQUIRED, "closed_opps.csv", features_cfg)
    if raw["LABEL_WON"].isna().any():
        raise ValueError("closed_opps.csv has NULL LABEL_WON rows — the SQL should only return resolved deals.")
    df = add_activity_features(clean(raw, features_cfg))
    df = add_closed_metadata(df)
    for col in ["TRUE_CYCLE_START", "TRUE_CLOSE_DATE"]:
        df[col] = pd.to_datetime(df[col])
    logger.info("Prepared %d closed opps (win rate %.1f%%)", len(df), df["LABEL_WON"].mean() * 100)
    return df


def prepare_open(raw: pd.DataFrame, features_cfg: dict) -> pd.DataFrame:
    validate_columns(raw, OPEN_REQUIRED, "open_opps.csv", features_cfg)
    df = add_activity_features(clean(raw, features_cfg))
    for col in ["OWNER_IS_ACTIVE", "OWNER_IS_NON_HUMAN", "IS_PROSPECT_DESK", "IS_DEAD_QUEUE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Prepared %d open opps", len(df))
    return df
