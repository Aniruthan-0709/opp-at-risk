"""
Training, persistence, and evaluation for the two models:
  suspect       - static features + DAYS_IN_SUSPECT, trained on snapshots
  prospect_plus - full feature set, trained on deals that reached Prospect+

scripts/train_baseline.py is the ONLY place models are fit. Everything
else loads the saved artifact (model + fitted category mappings + feature
list), so scoring always uses exactly what was trained.

Metrics: precision / recall / ROC-AUC / AUC-PR (primary), never plain
accuracy, given the ~12% win rate.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

MODEL_NAMES = ("suspect", "prospect_plus")


def artifact_path(models_dir: str, model_name: str) -> Path:
    return Path(models_dir) / f"{model_name}_model.joblib"


def load_model_dataset(processed_dir: str, model_name: str, split: str) -> pd.DataFrame:
    """split: 'train' | 'test' | 'eval'. Files are written by scripts/prepare_data.py."""
    path = Path(processed_dir) / f"{model_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/prepare_data.py first.")
    return pd.read_csv(path)


def weights_or_none(df: pd.DataFrame) -> pd.Series | None:
    return df["SAMPLE_WEIGHT"] if "SAMPLE_WEIGHT" in df.columns else None


# ---------------------------------------------------------------------------
# Categorical handling
# ---------------------------------------------------------------------------
def fit_categories(X: pd.DataFrame, categorical_columns: list[str]) -> dict[str, list]:
    return {
        col: sorted(X[col].astype(str).unique().tolist())
        for col in categorical_columns if col in X.columns
    }


def apply_saved_categories(X: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """Apply train-fitted categories. Unseen values become NaN (XGBoost treats as missing)."""
    X = X.copy()
    for col, cats in categories.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col].astype(str), categories=cats)
    return X


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_logistic_pipeline(feature_columns: list[str], categorical_columns: list[str], lr_cfg: dict) -> Pipeline:
    """
    Sanity-check baseline. StandardScaler is REQUIRED: without it AMOUNT's
    scale vs. 0-1 ratios breaks lbfgs convergence (this regressed twice).
    """
    categorical = [c for c in categorical_columns if c in feature_columns]
    numeric = [c for c in feature_columns if c not in categorical]
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = LogisticRegression(
        max_iter=lr_cfg["max_iter"], class_weight=lr_cfg["class_weight"], random_state=lr_cfg["random_state"],
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_xgboost(X: pd.DataFrame, y: pd.Series, xgb_cfg: dict, sample_weight: pd.Series | None = None):
    import xgboost as xgb

    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    pos_w, neg_w = w[y.to_numpy() == 1].sum(), w[y.to_numpy() == 0].sum()
    scale_pos_weight = neg_w / pos_w if pos_w > 0 else 1.0
    logger.info("XGBoost scale_pos_weight = %.2f (weighted neg=%.0f, pos=%.0f)", scale_pos_weight, neg_w, pos_w)

    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        random_state=xgb_cfg["random_state"],
        tree_method=xgb_cfg["tree_method"],
        enable_categorical=True,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_and_save_xgboost(model_name: str, df: pd.DataFrame, feature_columns: list[str],
                           categorical_columns: list[str], target: str, xgb_cfg: dict, models_dir: str):
    X = df[feature_columns]
    categories = fit_categories(X, categorical_columns)
    X_cat = apply_saved_categories(X, categories)
    model = train_xgboost(X_cat, df[target], xgb_cfg, weights_or_none(df))

    path = artifact_path(models_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model_name": model_name,
        "model": model,
        "categories": categories,
        "feature_columns": feature_columns,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train_rows": len(df),
    }, path)
    logger.info("Saved %s model (%d features) to %s", model_name, len(feature_columns), path)
    return model, categories


def load_artifact(models_dir: str, model_name: str) -> dict:
    path = artifact_path(models_dir, model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run 'python scripts/train_baseline.py' first "
            "(and again whenever features, config, or input data change)."
        )
    artifact = joblib.load(path)
    logger.info("Loaded %s model from %s (trained %s, %d features)",
                model_name, path, artifact["trained_at"], len(artifact["feature_columns"]))
    return artifact


class Calibrator:
    """
    Maps raw model scores to real win rates. Fitted on the holdout test set.
    Needed because scale_pos_weight (class weighting) deliberately inflates
    raw probabilities so the model pays attention to rare wins.
      sigmoid : Platt scaling on the log-odds — strictly increasing, so the
                ranking of deals is unchanged (AUC-PR / ROC-AUC unchanged)
      isotonic: flexible step curve — may tie a few deals together
    """
    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unknown calibration method '{method}'")
        self.method = method

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y, sample_weight=None) -> "Calibrator":
        if self.method == "sigmoid":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw), np.asarray(y),
                                                                       sample_weight=sample_weight)
        else:
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
                raw, np.asarray(y), sample_weight=sample_weight)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            return self.model_.predict_proba(self._logit(raw))[:, 1]
        return self.model_.predict(raw)


def raw_scores(artifact: dict, df: pd.DataFrame) -> np.ndarray:
    X = apply_saved_categories(df[artifact["feature_columns"]], artifact["categories"])
    return artifact["model"].predict_proba(X)[:, 1]


def predict_proba(artifact: dict, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
    """Calibrated win probability by default; raw model score if calibrated=False or no calibrator saved."""
    raw = raw_scores(artifact, df)
    calibrator = artifact.get("calibrator")
    if calibrated and calibrator is not None:
        return calibrator.transform(raw)
    if calibrated:
        logger.warning("%s has no calibrator — returning RAW scores. Re-run train_baseline.py.",
                       artifact["model_name"])
    return raw


def fit_and_attach_calibrator(model_name: str, test_df: pd.DataFrame, target: str,
                              models_dir: str, method: str) -> dict:
    """Fit the calibrator on the holdout test set and save it inside the model artifact."""
    path = artifact_path(models_dir, model_name)
    artifact = joblib.load(path)
    if method == "none":
        artifact["calibrator"] = None
        logger.info("%s: calibration disabled (method: none)", model_name)
    else:
        raw = raw_scores(artifact, test_df)
        artifact["calibrator"] = Calibrator(method).fit(raw, test_df[target], weights_or_none(test_df))
        logger.info("%s: fitted %s calibration on %d test rows and saved it with the model",
                    model_name, method, len(test_df))
    joblib.dump(artifact, path)
    return artifact


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_metrics(y: pd.Series, proba: np.ndarray, threshold: float = 0.5,
                    sample_weight: pd.Series | None = None) -> dict:
    pred = (proba >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        logger.warning("Only one class present — ROC-AUC / AUC-PR undefined for this set.")
        return {"n": int(len(y)), "win_rate": float(np.average(y, weights=sample_weight))}
    return {
        "n": int(len(y)),
        "win_rate": round(float(np.average(y, weights=sample_weight)), 4),
        "precision": round(precision_score(y, pred, zero_division=0, sample_weight=sample_weight), 4),
        "recall": round(recall_score(y, pred, zero_division=0, sample_weight=sample_weight), 4),
        "roc_auc": round(roc_auc_score(y, proba, sample_weight=sample_weight), 4),
        "auc_pr": round(average_precision_score(y, proba, sample_weight=sample_weight), 4),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series, label: str, threshold: float = 0.5,
             sample_weight: pd.Series | None = None) -> dict:
    metrics = compute_metrics(y, model.predict_proba(X)[:, 1], threshold, sample_weight)
    logger.info("%s metrics: %s", label, metrics)
    return metrics


def calibration_table(y: pd.Series, proba: np.ndarray, sample_weight: pd.Series | None = None,
                      n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs. actual win rate per probability decile. Well calibrated = the two columns match."""
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    df = pd.DataFrame({"y": np.asarray(y), "p": proba, "w": w})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    df["wp"], df["wy"] = df["w"] * df["p"], df["w"] * df["y"]
    g = df.groupby("bin", observed=True).agg(n=("p", "size"), w=("w", "sum"), wp=("wp", "sum"), wy=("wy", "sum"))
    return pd.DataFrame({
        "bin": g.index.astype(str),
        "n": g["n"].to_numpy(),
        "mean_predicted": (g["wp"] / g["w"]).round(4).to_numpy(),
        "actual_win_rate": (g["wy"] / g["w"]).round(4).to_numpy(),
    })