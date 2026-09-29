"""
WHY deals win or lose, by Vertical with every Initiative nested underneath.
Replaces the four loss/win x historical/open scripts:

  --direction   loss | win
  --population  historical  (closed training deals: LABEL_WON = 0 / 1)
                open        (scored open opps: WIN_PROBABILITY < / > threshold)
  --model       suspect | prospect_plus | all

For each segment (overall -> Vertical -> each Initiative in that Vertical):
  - every feature ranked by weighted mean signed SHAP (most negative first
    for loss, most positive first for win)
  - bin-level detail for the top N features: WHICH value range does the pulling
  - segments / bins below the confidence thresholds are flagged, never hidden
For --population open, also writes one row per opp with its own top reasons.

Outputs (reports/):
  <direction>_reasons_<population>_<model>.csv          segment x feature ranking
  <direction>_reason_bins_<population>_<model>.csv      segment x feature x bin
  <direction>_reasons_per_opp_<model>.csv               (open only)

Usage:
  python scripts/reasons.py --direction loss --population open --model all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import load_all
from src.logging_config import setup_logging
from src.models.explain import compute_shap_values, rank_reasons, reason_bins, top_reasons_per_row
from src.models.train import apply_saved_categories, load_artifact, load_model_dataset, weights_or_none
from src.utils import MODEL_CHOICES, models_from_arg

logger = setup_logging("reasons")


def load_population(model_name: str, direction: str, population: str, data_cfg: dict, features_cfg: dict) -> pd.DataFrame:
    if population == "historical":
        df = load_model_dataset(data_cfg["output"]["processed_dir"], model_name, "train")
        return df[df[features_cfg["target"]] == (1 if direction == "win" else 0)].reset_index(drop=True)
    path = Path(data_cfg["output"]["processed_dir"]) / f"open_{model_name}_scored.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/score_open_opps.py first.")
    df = pd.read_csv(path)
    thr = features_cfg["thresholds"]["win_probability"]
    keep = df["WIN_PROBABILITY"] > thr if direction == "win" else df["WIN_PROBABILITY"] < thr
    return df[keep].reset_index(drop=True)


def report_segment(level: str, vertical: str, initiative: str, mask: np.ndarray, ctx: dict,
                   rank_rows: list, bin_rows: list) -> None:
    n = int(mask.sum())
    if n == 0:
        return
    low = n < ctx["min_segment"]
    label = f"{vertical}" if level == "vertical" else (f"  > {initiative}" if level == "initiative" else "ALL")
    logger.info("-" * 70)
    logger.info("%s  n=%d%s", label, n, "  <- LOW CONFIDENCE" if low else "")

    ranking = rank_reasons(ctx["shap"], ctx["features"], ctx["direction"], ctx["weights"], mask)
    for i, r in ranking.iterrows():
        logger.info("    %-40s mean_signed_shap=%+.4f", r["feature"], r["mean_signed_shap"])
        rank_rows.append({"level": level, "vertical": vertical, "initiative": initiative, "n": n,
                          "low_confidence": low, "rank": i + 1, "feature": r["feature"],
                          "mean_signed_shap": round(r["mean_signed_shap"], 5)})

    logger.info("    --- bin detail, top %d features ---", ctx["top_bins"])
    for feature in ranking["feature"].head(ctx["top_bins"]):
        bins = reason_bins(ctx["shap"], ctx["X"], mask, feature, ctx["direction"], ctx["weights"], ctx["min_bin"])
        logger.info("    %s:", feature)
        for _, b in bins.iterrows():
            logger.info("      %-28s n=%-6d mean_shap=%+.4f%s", b["bin"], b["n"], b["mean_shap"],
                        "  <- LOW CONFIDENCE" if b["low_confidence"] else "")
            bin_rows.append({"level": level, "vertical": vertical, "initiative": initiative, "feature": feature,
                             "bin": b["bin"], "n": int(b["n"]), "mean_shap": b["mean_shap"],
                             "low_confidence": bool(b["low_confidence"])})


def run(model_name: str, direction: str, population: str, data_cfg: dict, features_cfg: dict) -> None:
    thr = features_cfg["thresholds"]
    reports = Path(data_cfg["output"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    artifact = load_artifact(data_cfg["output"]["models_dir"], model_name)
    features = artifact["feature_columns"]

    df = load_population(model_name, direction, population, data_cfg, features_cfg)
    logger.info("=" * 70)
    logger.info("%s reasons | %s | %s model | %d rows", direction.upper(), population, model_name, len(df))
    if df.empty:
        logger.warning("Population is empty — nothing to explain.")
        return

    X = apply_saved_categories(df[features], artifact["categories"])
    ctx = {
        "shap": compute_shap_values(artifact["model"], X), "X": X, "features": features, "direction": direction,
        "weights": weights_or_none(df), "min_segment": thr["min_segment_size"],
        "min_bin": thr["min_confident_bin_size"], "top_bins": thr["top_n_features_for_bins"],
    }
    df["INITIATIVE__C"] = df["INITIATIVE__C"].fillna("(no initiative)")
    rank_rows, bin_rows = [], []

    report_segment("overall", "ALL", "ALL", np.ones(len(df), dtype=bool), ctx, rank_rows, bin_rows)
    for vertical in sorted(df["VERTICAL"].dropna().unique()):
        v_mask = (df["VERTICAL"] == vertical).to_numpy()
        logger.info("=" * 70)
        report_segment("vertical", vertical, "ALL", v_mask, ctx, rank_rows, bin_rows)
        for initiative in df.loc[v_mask, "INITIATIVE__C"].value_counts().index:
            i_mask = v_mask & (df["INITIATIVE__C"] == initiative).to_numpy()
            report_segment("initiative", vertical, initiative, i_mask, ctx, rank_rows, bin_rows)

    tag = f"{population}_{model_name}"
    pd.DataFrame(rank_rows).to_csv(reports / f"{direction}_reasons_{tag}.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(reports / f"{direction}_reason_bins_{tag}.csv", index=False)
    logger.info("Saved %s and %s", reports / f"{direction}_reasons_{tag}.csv", reports / f"{direction}_reason_bins_{tag}.csv")

    if population == "open":
        rows = []
        for i in range(len(df)):
            row = {c: df.loc[i, c] for c in ["OPPORTUNITY_ID", "ACCOUNTID", "VERTICAL", "INITIATIVE__C",
                                             "CURRENT_STAGENAME", "WIN_PROBABILITY"]}
            for rank, (feat, val) in enumerate(
                    top_reasons_per_row(ctx["shap"], features, i, direction, thr["top_n_reasons_per_opp"]), start=1):
                row[f"REASON_{rank}"], row[f"REASON_{rank}_SHAP"] = feat, round(val, 4)
            rows.append(row)
        per_opp = pd.DataFrame(rows).sort_values("WIN_PROBABILITY", ascending=(direction == "loss"))
        path = reports / f"{direction}_reasons_per_opp_{model_name}.csv"
        per_opp.to_csv(path, index=False)
        logger.info("Saved %d per-opp reasons to %s", len(per_opp), path)


def main() -> None:
    p = argparse.ArgumentParser(description="Win/loss reasons by Vertical and nested Initiative")
    p.add_argument("--direction", choices=["loss", "win"], required=True)
    p.add_argument("--population", choices=["historical", "open"], required=True)
    p.add_argument("--model", choices=MODEL_CHOICES + ["all"], default="all")
    args = p.parse_args()
    data_cfg, features_cfg, _ = load_all()
    for m in models_from_arg(args.model):
        run(m, args.direction, args.population, data_cfg, features_cfg)


if __name__ == "__main__":
    main()
