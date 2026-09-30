"""
Trains and SAVES both models. The only script in the project that fits a
model — everything downstream loads models/<model>_model.joblib.

For each model:
  - logistic regression sanity check (not saved; confirms the features carry signal)
  - XGBoost, saved with its fitted category mappings + feature list
  - holdout test metrics (weighted by SAMPLE_WEIGHT for the Suspect snapshots)

Re-run after any change to input data, features, or config.

Usage:  python scripts/train_baseline.py [--model suspect|prospect_plus|all]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.config import load_all
from src.logging_config import setup_logging
from src.models.train import (
    apply_saved_categories,
    build_logistic_pipeline,
    compute_metrics,
    evaluate,
    fit_and_attach_calibrator,
    load_model_dataset,
    predict_proba,
    train_and_save_xgboost,
    weights_or_none,
)
from src.utils import load_feature_list, models_from_arg, parse_model_args

logger = setup_logging("train_baseline")


def train_one(model_name: str, data_cfg: dict, features_cfg: dict, model_cfg: dict) -> None:
    processed = data_cfg["output"]["processed_dir"]
    target = features_cfg["target"]
    threshold = features_cfg["thresholds"]["win_probability"]
    categorical = features_cfg["categorical_columns"]
    features = load_feature_list(model_name)

    train_df = load_model_dataset(processed, model_name, "train")
    test_df = load_model_dataset(processed, model_name, "test")
    w_train, w_test = weights_or_none(train_df), weights_or_none(test_df)
    logger.info("=" * 70)
    logger.info("%s: %d train rows / %d test rows, %d features",
                model_name.upper(), len(train_df), len(test_df), len(features))

    with mlflow.start_run(run_name=f"{model_name}_logistic"):
        pipeline = build_logistic_pipeline(features, categorical, model_cfg["logistic_regression"])
        fit_kwargs = {"model__sample_weight": w_train} if w_train is not None else {}
        pipeline.fit(train_df[features], train_df[target], **fit_kwargs)
        metrics = evaluate(pipeline, test_df[features], test_df[target], f"{model_name} logistic (test)",
                           threshold, w_test)
        mlflow.log_params({"model": model_name, "type": "logistic", "n_features": len(features)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=["numpy.dtype"])

    with mlflow.start_run(run_name=f"{model_name}_xgboost"):
        model, categories = train_and_save_xgboost(
            model_name, train_df, features, categorical, target, model_cfg["xgboost"], data_cfg["output"]["models_dir"],
        )
        X_test = apply_saved_categories(test_df[features], categories)
        metrics = evaluate(model, X_test, test_df[target], f"{model_name} XGBoost (test, raw)", threshold, w_test)

        # Calibration is fitted on the test set, so calibrated TEST metrics are
        # in-sample for the calibrator — judge calibration on Q3 in evaluate.py.
        method = model_cfg.get("calibration", {}).get("method", "sigmoid")
        artifact = fit_and_attach_calibrator(model_name, test_df, target, data_cfg["output"]["models_dir"], method)
        cal_metrics = compute_metrics(test_df[target], predict_proba(artifact, test_df), threshold, w_test)
        logger.info("%s XGBoost (test, calibrated — fitted on this set) metrics: %s", model_name, cal_metrics)

        mlflow.log_params({"model": model_name, "type": "xgboost", "n_features": len(features),
                           "calibration": method, **model_cfg["xgboost"]})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(f"configs/{model_name}_feature_list.json")
        mlflow.xgboost.log_model(model, name="model")


def main() -> None:
    args = parse_model_args("Train and save models", allow_all=True)
    data_cfg, features_cfg, model_cfg = load_all()
    mlflow.set_tracking_uri(model_cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(model_cfg["mlflow"]["experiment_name"])
    for m in models_from_arg(args.model):
        train_one(m, data_cfg, features_cfg, model_cfg)
    logger.info("Done. Compare runs with: mlflow ui --host 127.0.0.1 --backend-store-uri %s",
                model_cfg["mlflow"]["tracking_uri"])


if __name__ == "__main__":
    main()