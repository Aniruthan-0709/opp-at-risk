"""
SHAP-based explanations, shared by explain_model.py, derive_insights.py
and reasons.py.

Everything accepts optional sample weights: the Suspect model trains on
several snapshots per deal, and without weights a deal that sat in Suspect
for a year would count ~12x more than one that moved on quickly.

Direction: "loss" ranks features by most NEGATIVE mean SHAP (what pulls
toward losing); "win" ranks by most POSITIVE (what pushes toward winning).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Business-defined bin edges (pd.cut, right=False). Anything not listed
# falls back to _auto_bin. HEADCOUNT tiers are pending business input.
CURATED_BINS = {
    "ACTIVE_CONTACT_COUNT": ([0, 1, 3, 6, 1e9], ["0", "1-2", "3-5", "6+"]),
    "PCT_CONTACTS_CAN_CONTACT": ([0, 0.5, 0.99, 1.01], ["<50%", "50-99%", "100%"]),
    "PRIOR_ACCOUNT_WIN_COUNT": ([0, 1, 2, 5, 1e9], ["0 (new account)", "1", "2-4", "5+"]),
    "DAYS_IN_SUSPECT": ([0, 30, 60, 90, 180, 1e9], ["<30 days", "30-60 days", "60-90 days", "90-180 days", "180+ days"]),
    "CYCLE_LENGTH_DAYS": ([0, 30, 60, 90, 180, 1e9], ["<30 days", "30-60 days", "60-90 days", "90-180 days", "180+ days"]),
    "AMOUNT": ([0, 1, 5000, 100000, 750000, 1e12], ["$0 (no amount yet)", "1-5K", "5K-100K", "100K-750K", "750K+"]),
    "MAX_STAGE_REACHED": ([1, 2, 3, 4, 5, 6], ["1 (Suspect)", "2 (Prospect)", "3 (Engaged)", "4 (Negotiating)", "5 (Contracting)"]),
    "IS_REMOTE_GEO": ([-0.5, 0.5, 1.5], ["Not Remote Geo", "Remote Geo"]),
    "acct_meet_share": ([0, 0.01, 0.3, 0.6, 1.01], ["0 (none)", "low (0-30%)", "mid (30-60%)", "high (60%+)"]),
    "acct_call_share": ([0, 0.01, 0.3, 0.6, 1.01], ["0 (none)", "low (0-30%)", "mid (30-60%)", "high (60%+)"]),
}
MISSING_LABEL = "Missing / no data"


def _w(weights, n: int) -> np.ndarray:
    return np.ones(n) if weights is None else np.asarray(weights, dtype=float)


def _mask(mask, n: int) -> np.ndarray:
    return np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


# ---------------------------------------------------------------------------
# SHAP computation + global ranking
# ---------------------------------------------------------------------------
def compute_shap_values(model, X: pd.DataFrame):
    """Exact TreeExplainer SHAP values (log-odds / margin space)."""
    import shap
    return shap.TreeExplainer(model)(X)


def rank_features_by_shap(shap_values, feature_columns: list[str], weights=None, mask=None) -> pd.DataFrame:
    """Weighted mean |SHAP| per feature: how much each feature moves predictions."""
    vals = shap_values.values
    m = _mask(mask, len(vals))
    w = _w(weights, len(vals))[m]
    mean_abs = np.average(np.abs(vals[m]), axis=0, weights=w)
    ranking = pd.DataFrame({"feature": feature_columns, "mean_abs_shap": mean_abs})
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = ranking["mean_abs_shap"].sum()
    ranking["pct_of_total"] = ranking["mean_abs_shap"] / total * 100 if total else 0.0
    return ranking


def rank_reasons(shap_values, feature_columns: list[str], direction: str, weights=None, mask=None) -> pd.DataFrame:
    """Weighted mean SIGNED SHAP within a population, sorted toward `direction`."""
    vals = shap_values.values
    m = _mask(mask, len(vals))
    w = _w(weights, len(vals))[m]
    mean_signed = np.average(vals[m], axis=0, weights=w)
    ranking = pd.DataFrame({"feature": feature_columns, "mean_signed_shap": mean_signed})
    return ranking.sort_values("mean_signed_shap", ascending=(direction == "loss")).reset_index(drop=True)


def top_reasons_per_row(shap_values, feature_columns: list[str], row_idx: int, direction: str,
                        top_n: int = 5) -> list[tuple[str, float]]:
    row = shap_values.values[row_idx]
    order = np.argsort(row) if direction == "loss" else np.argsort(row)[::-1]
    return [(feature_columns[i], float(row[i])) for i in order[:top_n]]


def call_vs_meeting_summary(ranking: pd.DataFrame) -> dict:
    call = ranking["feature"].str.contains("CALL", case=False)
    meet = ranking["feature"].str.contains("MEET", case=False)
    return {
        "call_total_importance": float(ranking.loc[call, "mean_abs_shap"].sum()),
        "meet_total_importance": float(ranking.loc[meet, "mean_abs_shap"].sum()),
    }


def directional_summary(shap_values, feature_columns: list[str], weights=None) -> pd.DataFrame:
    """Weighted mean signed SHAP for every feature (coarse direction; skewed by the ~88% loss base rate)."""
    ranking = rank_reasons(shap_values, feature_columns, "win", weights)
    ranking["direction"] = np.where(ranking["mean_signed_shap"] > 0, "toward WON", "toward LOST")
    return ranking


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def _auto_bin(values: pd.Series, n_bins: int) -> pd.Series:
    """
    Fallback binning. Plain qcut collapses to one bin for low-cardinality
    flags and heavily zero-skewed counts, so those get special handling.
    """
    present = values.dropna()
    result = pd.Series(MISSING_LABEL, index=values.index, dtype=object)
    if present.empty:
        return result
    if present.nunique() <= n_bins:
        result.loc[present.index] = present.astype(str)
        return result
    if (present == 0).mean() >= 0.5:
        result.loc[present.index] = "0"
        nonzero = present[present > 0]
        if nonzero.nunique() >= max(n_bins - 1, 1):
            try:
                result.loc[nonzero.index] = pd.qcut(nonzero, q=n_bins - 1, duplicates="drop").astype(str)
            except ValueError:
                result.loc[nonzero.index] = "nonzero"
        else:
            result.loc[nonzero.index] = "nonzero"
        return result
    result.loc[present.index] = pd.qcut(present, q=n_bins, duplicates="drop").astype(str)
    return result


def bin_feature(values: pd.Series, feature: str, n_bins: int = 4) -> pd.Series:
    """Curated business bins if defined, else auto bins. NaN gets its own labelled bin."""
    if not pd.api.types.is_numeric_dtype(values):
        return values.astype(object).where(values.notna(), MISSING_LABEL).astype(str)
    if feature in CURATED_BINS:
        edges, labels = CURATED_BINS[feature]
        binned = pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=False)
        return binned.astype(object).where(binned.notna(), MISSING_LABEL).astype(str)
    return _auto_bin(values, n_bins)


def _ordered(summary: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Keep curated bins in their natural order; otherwise leave as given."""
    if feature in CURATED_BINS:
        order = {lab: i for i, lab in enumerate(CURATED_BINS[feature][1] + [MISSING_LABEL])}
        return summary.assign(_o=summary["bin"].map(order).fillna(99)).sort_values("_o").drop(columns="_o")
    return summary


def feature_dependence_bins(shap_values, X: pd.DataFrame, y: pd.Series, feature: str,
                            weights=None, min_confident_n: int = 15) -> pd.DataFrame:
    """
    Mixed-outcome table (Won AND Lost together): per bin, row count, weighted
    real win rate, and weighted mean SHAP. A bin is trustworthy when win rate
    vs. baseline and mean SHAP point the same way.
    """
    idx = list(X.columns).index(feature)
    w = _w(weights, len(X))
    df = pd.DataFrame({
        "bin": bin_feature(X[feature].reset_index(drop=True), feature).to_numpy(),
        "shap": shap_values.values[:, idx], "y": np.asarray(y), "w": w,
    })
    df["wy"], df["ws"] = df["w"] * df["y"], df["w"] * df["shap"]
    g = df.groupby("bin").agg(n=("y", "size"), w=("w", "sum"), wy=("wy", "sum"), ws=("ws", "sum")).reset_index()
    out = pd.DataFrame({
        "bin": g["bin"], "n": g["n"],
        "win_rate_pct": (g["wy"] / g["w"] * 100).round(1),
        "mean_shap": (g["ws"] / g["w"]).round(4),
    })
    out["low_confidence"] = out["n"] < min_confident_n
    return _ordered(out, feature).reset_index(drop=True)


def reason_bins(shap_values, X: pd.DataFrame, mask, feature: str, direction: str,
                weights=None, min_confident_n: int = 15) -> pd.DataFrame:
    """
    Within an already-restricted population (all losses, or all wins), which
    value range of `feature` pulls hardest in `direction`. No win-rate column:
    it would be 0% or 100% everywhere by construction.
    """
    m = _mask(mask, len(X))
    idx = list(X.columns).index(feature)
    w = _w(weights, len(X))[m]
    vals = X[feature].reset_index(drop=True)[m]
    df = pd.DataFrame({"bin": bin_feature(vals, feature).to_numpy(), "shap": shap_values.values[m, idx], "w": w})
    df["ws"] = df["w"] * df["shap"]
    g = df.groupby("bin").agg(n=("shap", "size"), w=("w", "sum"), ws=("ws", "sum")).reset_index()
    out = pd.DataFrame({"bin": g["bin"], "n": g["n"], "mean_shap": (g["ws"] / g["w"]).round(4)})
    out["low_confidence"] = out["n"] < min_confident_n
    return out.sort_values("mean_shap", ascending=(direction == "loss")).reset_index(drop=True)