
"""Evaluate SentinelAI by training on Tuesday and testing on Wednesday."""
from __future__ import annotations

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import load_labeled_csv
from src.model import build_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MachineLearningCVE"

TUESDAY = DATA / "Tuesday-WorkingHours.pcap_ISCX.csv"
WEDNESDAY = DATA / "Wednesday-workingHours.pcap_ISCX.csv"


def main() -> None:
    for path in (TUESDAY, WEDNESDAY):
        if not path.exists():
            raise SystemExit(f"Dataset file not found: {path}")

    print("Loading Tuesday training data...")
    X_train, y_train = load_labeled_csv(str(TUESDAY), "Label")

    print("\nLoading Wednesday test data...")
    X_test, y_test = load_labeled_csv(str(WEDNESDAY), "Label")

    # Use only features available in both datasets, preserving
    # Tuesday's column order.
    common = [column for column in X_train.columns if column in X_test.columns]
    if not common:
        raise SystemExit("No common numeric features found between the datasets.")

    X_train = X_train[common].copy()
    X_test = X_test[common].copy()

    # Remove features that contain no usable training values.
    usable = X_train.columns[X_train.notna().any(axis=0)].tolist()
    X_train = X_train[usable]
    X_test = X_test[usable]

    if X_train.empty or len(usable) == 0:
        raise SystemExit("No usable training features remain.")

    print("\nTraining on Tuesday and evaluating on Wednesday...")
    model = build_model(42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = ["Benign", "Suspicious"]
    print("\nCROSS-DAY EVALUATION")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Common features: {len(usable)}")
    print(f"Training classes: {y_train.value_counts().to_dict()}")
    print(f"Test classes: {y_test.value_counts().to_dict()}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
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

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("Order:", labels)
    print(confusion_matrix(y_test, predictions, labels=labels))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
            digits=4,
        )
    )

    print(
        "Note: this is a cross-day benchmark, not proof of "
        "real-world network detection performance."
    )


if __name__ == "__main__":
    main()
