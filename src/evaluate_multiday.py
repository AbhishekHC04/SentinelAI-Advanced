
"""Train on multiple CIC-IDS2017 days and test on Wednesday."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.data import load_labeled_csv
from src.model import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MachineLearningCVE"

TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]
TEST_FILE = "Wednesday-workingHours.pcap_ISCX.csv"


def load_day(filename):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    X, y = load_labeled_csv(str(path), "Label")
    # Keep exact duplicate rows out of this experiment.
    return X, y


def main():
    X_parts, y_parts = [], []

    for filename in TRAIN_FILES:
        print(f"\nLoading training file: {filename}")
        X, y = load_day(filename)
        X_parts.append(X)
        y_parts.append(y)
        print(f"Rows: {len(X):,} | Features: {X.shape[1]}")

    print(f"\nLoading held-out test file: {TEST_FILE}")
    X_test, y_test = load_day(TEST_FILE)
    print(f"Test rows: {len(X_test):,} | Features: {X_test.shape[1]}")

    # Align feature columns across all days.
    common = set(X_parts[0].columns)
    for X in X_parts[1:]:
        common.intersection_update(X.columns)
    common.intersection_update(X_test.columns)
    features = [c for c in X_parts[0].columns if c in common]

    if not features:
        raise SystemExit("No common features across training and test days.")

    X_train = pd.concat([X[features] for X in X_parts], ignore_index=True)
    y_train = pd.concat(y_parts, ignore_index=True)
    X_test = X_test[features].copy()

    # Remove exact duplicate feature/label records across the combined training
    # data. Do not use Wednesday to fit or select the model.
    combined = X_train.copy()
    combined["__label__"] = y_train.to_numpy()
    before = len(combined)
    combined = combined.drop_duplicates()
    print(f"\nRemoved duplicate training records: {before - len(combined):,}")
    y_train = combined.pop("__label__")
    X_train = combined

    # Drop columns that are entirely missing in training.
    usable = X_train.columns[X_train.notna().any(axis=0)].tolist()
    X_train = X_train[usable]
    X_test = X_test[usable]

    print("\nTraining multi-day model...")
    model = build_model(42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nMULTI-DAY → WEDNESDAY RESULTS")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Features: {len(usable)}")
    print("\nTraining class counts:")
    print(y_train.value_counts().to_string())
    print("\nTest class counts:")
    print(y_test.value_counts().to_string())

    labels = ["Benign", "Suspicious"]
    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )
    print(
        "Suspicious precision: "
        f"{precision_score(y_test, predictions, pos_label='Suspicious', zero_division=0):.4f}"
    )
    print(
        "Suspicious recall: "
        f"{recall_score(y_test, predictions, pos_label='Suspicious', zero_division=0):.4f}"
    )
    print(
        "Suspicious F1: "
        f"{f1_score(y_test, predictions, pos_label='Suspicious', zero_division=0):.4f}"
    )
    print("\nConfusion matrix (actual rows, predicted columns):")
    print("Label order:", labels)
    print(confusion_matrix(y_test, predictions, labels=labels))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=labels, zero_division=0, digits=4
    ))
    print(
        "\nCaution: this is a benchmark experiment on CIC-IDS2017, "
        "not proof of real-world detection performance."
    )


if __name__ == "__main__":
    main()
