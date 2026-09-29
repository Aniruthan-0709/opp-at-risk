"""Small helpers shared by the scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_CHOICES = ["suspect", "prospect_plus"]


def feature_list_path(model_name: str) -> Path:
    return Path("configs") / f"{model_name}_feature_list.json"


def load_feature_list(model_name: str) -> list[str]:
    path = feature_list_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run: python scripts/select_features.py --model {model_name}")
    return json.loads(path.read_text())["feature_columns"]


def parse_model_args(description: str, allow_all: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    choices = MODEL_CHOICES + (["all"] if allow_all else [])
    parser.add_argument("--model", choices=choices, default="all" if allow_all else None,
                        required=not allow_all, help="Which model to run for")
    return parser.parse_args()


def models_from_arg(value: str) -> list[str]:
    return MODEL_CHOICES if value == "all" else [value]
