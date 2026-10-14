"""Streamlit dashboard for SentinelAI Advanced."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.inference import load_artifacts, predict_frame

ROOT = Path(__file__).resolve().parent
META_PATH = ROOT / "artifacts" / "metadata.json"

st.set_page_config(page_title="SentinelAI Advanced", page_icon="🛡️", layout="wide")
st.title("🛡️ SentinelAI Advanced")
st.caption("Machine-learning network-flow triage | Model evaluation | Batch inference")
st.warning("Educational prototype. Predictions are not proof of compromise. Validate on representative, independently held-out data.")

try:
    model, metadata = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Model card")
    
    st.write(
        f"**Training source:** "
        f"{metadata.get('data_source', metadata.get('dataset', 'Unknown'))}"
    )
    st.write(f"**Feature count:** {len(metadata['features'])}")
    st.write(
        f"**Decision threshold:** "
        f"{metadata.get('decision_threshold', 0.5):.2f}"
    )
    if "test_rows" in metadata:
        st.write(f"**Test rows:** {metadata['test_rows']}")
    if "seed" in metadata:
        st.write(f"**Seed:** {metadata['seed']}")

    page = st.radio("Page", ["Overview", "Batch prediction", "Model card"])

if page == "Overview":
    st.subheader("Held-out test metrics")
    metrics = metadata.get(
        "metrics", metadata.get("held_out_test_metrics", {})
    )
    cards = st.columns(5)
    items = [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Suspicious precision", "precision_suspicious"),
        ("Suspicious recall", "recall_suspicious"),
        ("Suspicious F1", "f1_suspicious"),
    ]
    for card, (label, key) in zip(cards, items):
        card.metric(label, f"{metrics[key]:.1%}")

    
    
    st.subheader("Confusion matrix")

    matrix = metadata.get("confusion_matrix")

    if matrix is not None:
        st.write("Rows = actual labels; columns = predicted labels")

        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Benign", "Actual Suspicious"],
            columns=["Predicted Benign", "Predicted Suspicious"],
        )

        st.dataframe(matrix_df, use_container_width=True)

        tn, fp = matrix[0]
        fn, tp = matrix[1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True negatives", f"{tn:,}")
        c2.metric("False positives", f"{fp:,}")
        c3.metric("False negatives", f"{fn:,}")
        c4.metric("True positives", f"{tp:,}")
    else:
        st.warning(
            "Confusion matrix is missing from the metadata loaded by "
            "this dashboard. Check the configured artifact directory."
        )


    st.subheader("Top model feature importances")
    importance = metadata.get("feature_importance", [])[:10]
    if importance:
        imp = pd.DataFrame(importance).sort_values("importance")
        st.bar_chart(imp.set_index("feature"))
        st.info(
        metadata.get(
            "caveat",
            metadata.get("warning", "Benchmark results; validate on representative data.")
        )
    )


elif page == "Batch prediction":
    st.subheader("Upload network-flow records")
    st.write(
        "Upload a CSV with numeric features matching the training schema. "
        "A `Label` column is optional and ignored for inference."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded:
        try:
            input_frame = pd.read_csv(uploaded, low_memory=False)
            st.dataframe(input_frame.head(10), use_container_width=True)

            results = predict_frame(input_frame, model, metadata)

            # Traffic summary
            total_flows = len(results)
            suspicious_flows = int(
                (results["SentinelAI_prediction"] == "Suspicious").sum()
            )
            benign_flows = total_flows - suspicious_flows
            suspicious_percentage = (
                suspicious_flows / total_flows * 100
                if total_flows else 0.0
            )

            st.subheader("Traffic analysis summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total flows", f"{total_flows:,}")
            c2.metric("Suspicious flows", f"{suspicious_flows:,}")
            c3.metric("Benign flows", f"{benign_flows:,}")

            st.metric(
                "Suspicious flow percentage",
                f"{suspicious_percentage:.2f}%",
            )

            if suspicious_flows > 0:
                st.error(
                    f"⚠️ ALERT: {suspicious_flows:,} flow(s) flagged as "
                    f"suspicious ({suspicious_percentage:.2f}%). "
                    "Review these records before drawing conclusions."
                )
            else:
                st.success(
                    "No flows were flagged as suspicious in this batch. "
                    "This does not guarantee that all traffic is benign."
                )

            # Detailed predictions
            st.subheader("Detailed predictions")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="sentinelai_predictions.csv",
                mime="text/csv",
            )

        except Exception as error:
            st.error(f"Could not process CSV: {error}")

else:
    st.subheader("Model card")
    
    st.json({
        "project": metadata.get("project", "SentinelAI Advanced"),
        "model_type": metadata.get("model_type", "RandomForestClassifier"),
        "data_source": metadata.get(
            "data_source", metadata.get("dataset", "Unknown")
        ),
        "features": metadata["features"],
        "features_count": len(metadata["features"]),
        "classes": metadata.get("classes", ["Benign", "Suspicious"]),
        "decision_threshold": metadata.get("decision_threshold", 0.5),
        "training_files": metadata.get("training_files", []),
        "validation_files": metadata.get("validation_files", []),
        "test_file": metadata.get("test_file"),
        "validation_metrics": metadata.get("validation_metrics", {}),
        "held_out_test_metrics": metadata.get(
            "held_out_test_metrics", metadata.get("metrics", {})
        ),
        "evaluation_method": metadata.get("evaluation_method", "Not specified"),
        "caveat": metadata.get(
            "caveat", metadata.get("warning", "Validate on representative data.")
        ),
    })

    
    
    st.markdown("""
    **Scope:** Supervised classification of network-flow records as
    Benign or Suspicious using a Random Forest model.

    **Dataset:** CIC-IDS2017. Training used Monday, Tuesday, and
    Thursday traffic. Friday was used to select the classification
    threshold, and Wednesday was used for cross-day benchmarking.

    **Evaluation caveat:** Wednesday was inspected during earlier
    development experiments. These results are exploratory and should
    not be described as a pristine, independent final test.

    **Threshold:** The decision threshold is selected using Friday
    validation data. Lower thresholds can increase suspicious-flow
    recall but may also increase false alerts.

    **Limitations:** CIC-IDS2017 may not represent current production
    networks. Dataset artifacts, class imbalance, traffic distribution
    shifts, and differences between attack types can affect performance.
    Model feature importance does not establish causation.

    **Next validation step:** Evaluate on a separate, authorized network
    capture or independently sourced dataset that was not used for
    training, threshold selection, or development decisions. Report
    precision, recall, F1, balanced accuracy, and the confusion matrix.

    **Safe use:** This is an educational prototype, not a standalone
    intrusion-detection or incident-confirmation system. Investigate
    alerts using authorized telemetry and human review.
    """)

