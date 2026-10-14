"""Data generation and CSV preparation for SentinelAI Advanced."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

FEATURES = [
    "flow_duration", "total_forward_packets", "total_backward_packets",
    "flow_bytes_per_second", "flow_packets_per_second",
    "forward_packet_length_mean", "backward_packet_length_mean",
    "packet_length_std", "syn_flag_count", "ack_flag_count",
]

BENIGN_WORDS = {"benign", "normal", "0", "legitimate", "clean"}

def make_demo_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """Create synthetic numeric flow-like features for a reproducible demo only."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURES),
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        weights=[0.72, 0.28],
        class_sep=1.15,
        flip_y=0.025,
        random_state=random_state,
    )
    frame = pd.DataFrame(X, columns=FEATURES)
    for column in FEATURES:
        values = frame[column]
        frame[column] = (values - values.min() + 0.001).astype(float)
    frame["Label"] = np.where(y == 1, "Suspicious", "Benign")
    return frame

def normalize_label(value: object) -> str:
    """Map common benign labels to Benign; map every other label to Suspicious."""
    label = str(value).strip().lower()
    return "Benign" if label in BENIGN_WORDS or "benign" in label or "normal" in label else "Suspicious"

def load_labeled_csv(path: str, target: str = "Label") -> tuple[pd.DataFrame, pd.Series]:
    """Load a labeled CSV, normalize binary labels, and retain numeric features only."""
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    target_col = next((c for c in frame.columns if c.lower() == target.strip().lower()), None)
   
    if target_col is None:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"Columns: {', '.join(frame.columns[:30])}"
        )

    frame = frame.dropna(subset=[target_col]).copy()

    # Remove exact duplicate records before preparing features.
    rows_before = len(frame)
    frame = frame.drop_duplicates().copy()
    print(f"Removed exact duplicate rows: {rows_before - len(frame)}")

    y = frame[target_col].map(normalize_label)
    X = frame.drop(columns=[target_col])
    # Avoid common identifier/time columns. Review dataset-specific fields before training.
    excluded = [c for c in X.columns if any(token in c.lower() for token in
                ("flow id", "timestamp", "src ip", "dst ip", "source ip", "destination ip"))]
    X = X.drop(columns=excluded, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns remain after cleaning.")
    if y.nunique() < 1:
        raise ValueError("The target contains no usable labels.")
    return X, y
