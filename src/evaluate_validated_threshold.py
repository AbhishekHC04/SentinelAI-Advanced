from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.data import load_labeled_csv
from src.model import build_model


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MachineLearningCVE"
OUTPUT = ROOT / "artifacts" / "candidate"

TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]

VALIDATION_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

TEST_FILE = "Wednesday-workingHours.pcap_ISCX.csv"

CANDIDATE_THRESHOLDS = np.arange(0.01, 0.501, 0.01)


def load_files(files):
    xs, ys = [], []

    for name in files:
        path = DATA / name
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        print(f"Loading {name}...")
        X, y = load_labeled_csv(str(path), "Label")
        xs.append(X)
        ys.append(y)

    return xs, ys


def deduplicate_training(X, y):
    combined = X.copy()
    combined["__label__"] = y.to_numpy()

    before = len(combined)
    combined = combined.drop_duplicates()
    print(f"Removed duplicate training records: {before - len(combined):,}")

    y_out = combined.pop("__label__")
    return combined, y_out


def get_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision_suspicious": float(
            precision_score(
                y_true, y_pred,
                pos_label="Suspicious",
                zero_division=0,
            )
        ),
        "recall_suspicious": float(
            recall_score(
                y_true, y_pred,
                pos_label="Suspicious",
                zero_division=0,
            )
        ),
        "f1_suspicious": float(
            f1_score(
                y_true, y_pred,
                pos_label="Suspicious",
                zero_division=0,
            )
        ),
    }


def main():
    train_xs, train_ys = load_files(TRAIN_FILES)
    val_xs, val_ys = load_files(VALIDATION_FILES)
    test_xs, test_ys = load_files([TEST_FILE])

    X_test, y_test = test_xs[0], test_ys[0]

    # Use only numeric features shared by every dataset partition.
    common = set(train_xs[0].columns)
    for X in train_xs[1:] + val_xs + test_xs:
        common.intersection_update(X.columns)

    features = [
        column for column in train_xs[0].columns
        if column in common
    ]

    if not features:
        raise SystemExit("No common features across the dataset files.")

    X_train = pd.concat(
        [X[features] for X in train_xs],
        ignore_index=True,
    )
    y_train = pd.concat(train_ys, ignore_index=True)

    X_val = pd.concat(
        [X[features] for X in val_xs],
        ignore_index=True,
    )
    y_val = pd.concat(val_ys, ignore_index=True)

    X_test = X_test[features].copy()

    X_train, y_train = deduplicate_training(X_train, y_train)

    # Remove features that are entirely missing in the training partition.
    usable = X_train.columns[
        X_train.notna().any(axis=0)
    ].tolist()

    if not usable:
        raise SystemExit("No usable numeric training features.")

    X_train = X_train[usable]
    X_val = X_val[usable]
    X_test = X_test[usable]

    if set(y_train.unique()) != {"Benign", "Suspicious"}:
        raise SystemExit(
            "Training data must contain both Benign and Suspicious labels."
        )

    print("\nTraining candidate model on Monday, Tuesday, and Thursday...")
    model = build_model(42)
    model.fit(X_train, y_train)

    classes = list(model.classes_)
    if "Suspicious" not in classes:
        raise SystemExit("Model is missing the Suspicious class.")

    suspicious_index = classes.index("Suspicious")
    val_prob = model.predict_proba(X_val)[:, suspicious_index]

    # Select threshold using Friday validation only.
    threshold_scores = []

    for threshold in CANDIDATE_THRESHOLDS:
        pred = np.where(
            val_prob >= threshold,
            "Suspicious",
            "Benign",
        )
        metrics = get_metrics(y_val, pred)
        threshold_scores.append(
            (metrics["f1_suspicious"], float(threshold), metrics)
        )

    # If thresholds tie, choose the higher threshold to reduce alerts.
    best_f1 = max(item[0] for item in threshold_scores)
    tied = [item for item in threshold_scores if item[0] == best_f1]
    _, best_threshold, validation_metrics = max(
        tied, key=lambda item: item[1]
    )

    print(f"\nSelected Friday threshold: {best_threshold:.2f}")
    print("Friday validation metrics:")
    for name, value in validation_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Evaluate Wednesday only after threshold selection is complete.
    test_prob = model.predict_proba(X_test)[:, suspicious_index]
    test_pred = np.where(
        test_prob >= best_threshold,
        "Suspicious",
        "Benign",
    )
    test_metrics = get_metrics(y_test, test_pred)

    print("\nWednesday held-out test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save to a separate candidate directory; do not overwrite the live model.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT / "sentinelai_model.joblib"
    metadata_path = OUTPUT / "metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_type": "RandomForestClassifier",
        "dataset": "CIC-IDS2017",
        "features": usable,
        "classes": [str(label) for label in model.classes_],
        "decision_threshold": best_threshold,
        "threshold_selection_metric": "Friday validation F1 for Suspicious",
        "training_files": TRAIN_FILES,
        "validation_files": VALIDATION_FILES,
        "test_file": TEST_FILE,
        "evaluation_method": (
            "Day-separated experiment: train Monday/Tuesday/Thursday, "
            "select threshold on Friday, evaluate on Wednesday"
        ),
        "validation_metrics": validation_metrics,
        "held_out_test_metrics": test_metrics,
        "warning": (
            "Offline CIC-IDS2017 benchmark results; not proof of "
            "real-world performance. Validate on additional data before "
            "production deployment."
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nCandidate model saved successfully:")
    print(f"  Model: {model_path}")
    print(f"  Metadata: {metadata_path}")
    print("\nThe existing artifacts/sentinelai_model.joblib was not changed.")


if __name__ == "__main__":
    main()
