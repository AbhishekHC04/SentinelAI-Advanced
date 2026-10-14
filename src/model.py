"""Training pipeline and evaluation helpers."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)

def build_model(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )),
    ])

def evaluate(model, X_test, y_test) -> dict:
    predicted = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, average="binary", pos_label="Suspicious", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "precision_suspicious": float(precision),
        "recall_suspicious": float(recall),
        "f1_suspicious": float(f1),
        "labels": ["Benign", "Suspicious"],
        "confusion_matrix": confusion_matrix(
            y_test, predicted, labels=["Benign", "Suspicious"]
        ).tolist(),
        "classification_report": classification_report(
            y_test, predicted, labels=["Benign", "Suspicious"],
            output_dict=True, zero_division=0
        ),
    }
