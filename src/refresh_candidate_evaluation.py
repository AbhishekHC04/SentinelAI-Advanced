
"""Recalculate candidate-model metrics on the Wednesday benchmark."""
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.data import load_labeled_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MachineLearningCVE"
CANDIDATE = ROOT / "artifacts" / "candidate"

MODEL_PATH = CANDIDATE / "sentinelai_model.joblib"
METADATA_PATH = CANDIDATE / "metadata.json"
TEST_FILE = DATA / "Wednesday-workingHours.pcap_ISCX.csv"


def main():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Candidate model or metadata is missing.")

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    print(f"Loading benchmark: {TEST_FILE.name}")
    X, y = load_labeled_csv(str(TEST_FILE), "Label")

    features = metadata["features"]
    X = X.reindex(columns=features)

    probabilities = model.predict_proba(X)
    classes = [str(label) for label in model.classes_]
    suspicious_index = classes.index("Suspicious")

    threshold = float(metadata["decision_threshold"])
    predictions = np.where(
        probabilities[:, suspicious_index] >= threshold,
        "Suspicious",
        "Benign",
    )

    labels = ["Benign", "Suspicious"]
    matrix = confusion_matrix(y, predictions, labels=labels)
    report = classification_report(
        y,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    metadata["confusion_matrix"] = matrix.tolist()
    metadata["confusion_matrix_labels"] = labels
    metadata["classification_report"] = report
    metadata["held_out_test_metrics"] = {
        "accuracy": float(np.mean(predictions == y.to_numpy())),
        "balanced_accuracy": float(
            (report["Benign"]["recall"]
             + report["Suspicious"]["recall"]) / 2
        ),
        "precision_suspicious": float(
            report["Suspicious"]["precision"]
        ),
        "recall_suspicious": float(
            report["Suspicious"]["recall"]
        ),
        "f1_suspicious": float(
            report["Suspicious"]["f1-score"]
        ),
    }
    metadata["evaluation_note"] = (
        "Exploratory Wednesday cross-day benchmark. Wednesday data "
        "was inspected during earlier development and is not a pristine "
        "independent holdout. Rows use the existing CSV loader's "
        "deduplication and label-normalization behavior."
    )

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    tn, fp, fn, tp = matrix.ravel()
    print("\nConfusion matrix (actual rows, predicted columns):")
    print("                  Predicted Benign  Predicted Suspicious")
    print(f"Actual Benign     {tn:16,}  {fp:19,}")
    print(f"Actual Suspicious {fn:16,}  {tp:19,}")
    print("\nUpdated benchmark metrics:")
    for name, value in metadata["held_out_test_metrics"].items():
        print(f"{name}: {value:.4f}")

    print(f"\nUpdated metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
