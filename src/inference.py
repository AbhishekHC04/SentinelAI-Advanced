
"""Shared model loading and prediction functions."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import os

ARTIFACT_DIR = Path(
    os.getenv("SENTINELAI_ARTIFACT_DIR", str(ROOT / "artifacts"))
)
MODEL_PATH = ARTIFACT_DIR / "sentinelai_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python -m src.train` first."
        )
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata


def align_features(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    clean = frame.copy()
    clean.columns = [str(c).strip() for c in clean.columns]
    clean = clean.loc[:, ~clean.columns.duplicated()]
    clean = clean.drop(
        columns=[c for c in clean.columns if c.lower() == "label"],
        errors="ignore",
    )

    for column in clean.columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.replace([np.inf, -np.inf], np.nan)
    aligned = pd.DataFrame(index=clean.index)

    for feature in feature_names:
        aligned[feature] = (
            clean[feature] if feature in clean.columns else np.nan
        )

    if aligned.dropna(how="all").empty:
        raise ValueError(
            "No numeric feature values found in the supplied records."
        )

    return aligned


def predict_frame(frame: pd.DataFrame, model=None, metadata=None) -> pd.DataFrame:
    if model is None or metadata is None:
        model, metadata = load_artifacts()

    X = align_features(frame, metadata["features"])
    probabilities = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]

    suspicious_index = next(
        (i for i, label in enumerate(classes) if label == "Suspicious"),
        None,
    )
    benign_index = next(
        (i for i, label in enumerate(classes) if label == "Benign"),
        None,
    )

    if suspicious_index is None or benign_index is None:
        raise ValueError(
            "Model must contain both 'Benign' and 'Suspicious' classes."
        )

    # The default threshold preserves compatibility with existing artifacts.
    threshold = float(metadata.get("decision_threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("decision_threshold must be between 0 and 1.")

    suspicious_probability = probabilities[:, suspicious_index]
    labels = np.where(
        suspicious_probability >= threshold, "Suspicious", "Benign"
    )

    result = frame.copy()
    result["SentinelAI_prediction"] = labels
    result["suspicious_probability"] = suspicious_probability
    result["decision_threshold"] = threshold
    return result
