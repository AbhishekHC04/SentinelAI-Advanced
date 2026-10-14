
"""Train SentinelAI on a CSV or multiple CIC-IDS2017 CSV files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data import make_demo_dataset, load_labeled_csv
from src.model import build_model, evaluate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data" / "MachineLearningCVE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="Path to one labeled CSV. Omit for synthetic demo.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="CSV filenames inside data/MachineLearningCVE for multi-day training.",
    )
    parser.add_argument("--target", default="Label")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Suspicious probability threshold. Default 0.5.",
    )
    args = parser.parse_args()

    if args.data and args.files:
        parser.error("Use either --data or --files, not both.")
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5.")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    synthetic = False

    if args.files:
        frames, labels = [], []
        for filename in args.files:
            path = DATA_DIR / filename
            if not path.exists():
                parser.error(f"Dataset file not found: {path}")
            print(f"Loading: {filename}")
            X_part, y_part = load_labeled_csv(str(path), args.target)
            frames.append(X_part)
            labels.append(y_part)

        common = set(frames[0].columns)
        for frame in frames[1:]:
            common.intersection_update(frame.columns)
        features = [c for c in frames[0].columns if c in common]
        if not features:
            raise SystemExit("No common numeric features across input files.")

        X = pd.concat(
            [frame[features] for frame in frames],
            ignore_index=True,
        )
        y = pd.concat(labels, ignore_index=True)

        # Remove exact duplicate feature/label records.
        combined = X.copy()
        combined["__label__"] = y.to_numpy()
        before = len(combined)
        combined = combined.drop_duplicates()
        print(f"Removed duplicate records: {before - len(combined):,}")
        y = combined.pop("__label__")
        X = combined
        source = "; ".join(args.files)

    elif args.data:
        X, y = load_labeled_csv(args.data, args.target)
        source = str(Path(args.data).resolve())

        # Remove duplicate feature/label records.
        combined = X.copy()
        combined["__label__"] = y.to_numpy()
        before = len(combined)
        combined = combined.drop_duplicates()
        print(f"Removed duplicate records: {before - len(combined):,}")
        y = combined.pop("__label__")
        X = combined

    else:
        frame = make_demo_dataset(random_state=args.seed)
        X, y = frame.drop(columns="Label"), frame["Label"]
        source = "synthetic demo dataset (not real network traffic)"
        synthetic = True

    X = X.copy()
    X.columns = [str(c).strip() for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    if X.empty or X.shape[1] == 0:
        raise SystemExit("No numeric features found.")
    if y.nunique() != 2:
        raise SystemExit(
            "Combined training data must contain both Benign and Suspicious labels."
        )

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    model = build_model(args.seed)
    model.fit(X_train, y_train)

    # Report default-threshold metrics separately from configured-threshold
    # metrics so threshold changes are transparent.
    probabilities = model.predict_proba(X_test)
    classes = [str(c) for c in model.classes_]
    suspicious_index = classes.index("Suspicious")
    suspicious_probability = probabilities[:, suspicious_index]
    predictions = np.where(
        suspicious_probability >= args.threshold,
        "Suspicious",
        "Benign",
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision_suspicious": float(
            precision_score(
                y_test, predictions, pos_label="Suspicious", zero_division=0
            )
        ),
        "recall_suspicious": float(
            recall_score(
                y_test, predictions, pos_label="Suspicious", zero_division=0
            )
        ),
        "f1_suspicious": float(
            f1_score(
                y_test, predictions, pos_label="Suspicious", zero_division=0
            )
        ),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS / "sentinelai_model.joblib"
    metadata_path = ARTIFACTS / "metadata.json"
    joblib.dump(model, model_path)

    classifier = model.named_steps["classifier"]
    importance = sorted(
        [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(X.columns, classifier.feature_importances_)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    metadata = {
        "project": "SentinelAI Advanced",
        "data_source": source,
        "synthetic_demo_data": synthetic,
        "target_column": args.target,
        "features": list(X.columns),
        "classes": classes,
        "rows": int(len(X)),
        "features_count": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "seed": args.seed,
        "decision_threshold": float(args.threshold),
        "metrics": metrics,
        "feature_importance": importance,
        "evaluation_method": "random stratified split",
        "caveat": (
            "Synthetic demo metrics are not evidence of real-world detection performance."
            if synthetic
            else
            "Metrics use a random split of the supplied data. "
            "For CIC-IDS2017, use separate-day validation and test data "
            "for stronger generalization estimates."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSentinelAI Advanced training complete")
    print(f"Source: {source}")
    print(f"Rows: {len(X):,} | Features: {X.shape[1]}")
    print(f"Decision threshold: {args.threshold:.3f}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print("Important: predictions are triage signals, not proof of an attack.")


if __name__ == "__main__":
    main()
