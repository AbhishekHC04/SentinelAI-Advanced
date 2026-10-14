
"""Diagnose predicted probabilities on the held-out Wednesday dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
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
    X, y = load_labeled_csv(str(DATA / filename), "Label")
    return X, y


def main():
    train_X, train_y = [], []

    for filename in TRAIN_FILES:
        print(f"Loading {filename}")
        X, y = load_day(filename)
        train_X.append(X)
        train_y.append(y)

    X_test, y_test = load_day(TEST_FILE)

    common = set(train_X[0].columns)
    for X in train_X[1:]:
        common.intersection_update(X.columns)
    common.intersection_update(X_test.columns)
    features = [c for c in train_X[0].columns if c in common]

    X_train = pd.concat(
        [X[features] for X in train_X], ignore_index=True
    )
    y_train = pd.concat(train_y, ignore_index=True)

    training = X_train.copy()
    training["__label__"] = y_train.to_numpy()
    training = training.drop_duplicates()
    y_train = training.pop("__label__")
    X_train = training

    usable = X_train.columns[X_train.notna().any(axis=0)].tolist()
    X_train = X_train[usable]
    X_test = X_test[usable]

    model = build_model(42)
    model.fit(X_train, y_train)

    suspicious_index = list(model.classes_).index("Suspicious")
    probabilities = model.predict_proba(X_test)[:, suspicious_index]

    print("\nTHRESHOLD DIAGNOSTIC — WEDNESDAY TEST SET")
    print("These results are diagnostic, not final model-selection results.")
    print(f"Test records: {len(y_test):,}")
    print(f"Actual suspicious records: {(y_test == 'Suspicious').sum():,}")
    print(f"Maximum suspicious probability: {probabilities.max():.4f}")
    print(f"Median suspicious probability: {np.median(probabilities):.4f}")
    print("\nThreshold | Precision | Recall | F1 | Balanced accuracy")

    for threshold in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        predictions = np.where(
            probabilities >= threshold, "Suspicious", "Benign"
        )
        precision = precision_score(
            y_test, predictions, pos_label="Suspicious", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, pos_label="Suspicious", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, pos_label="Suspicious", zero_division=0
        )
        balanced = balanced_accuracy_score(y_test, predictions)
        print(
            f"{threshold:9.2f} | {precision:9.4f} | "
            f"{recall:6.4f} | {f1:5.4f} | {balanced:17.4f}"
        )


if __name__ == "__main__":
    main()
