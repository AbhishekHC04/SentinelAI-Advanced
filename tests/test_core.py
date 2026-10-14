import pandas as pd
import pytest

from src.data import make_demo_dataset, normalize_label
from src.model import build_model
from src.inference import align_features, predict_frame

def test_demo_data_has_two_classes():
    frame = make_demo_dataset(n_samples=240)
    assert len(frame) == 240
    assert set(frame["Label"].unique()) == {"Benign", "Suspicious"}

def test_label_normalization():
    assert normalize_label("BENIGN") == "Benign"
    assert normalize_label("Normal") == "Benign"
    assert normalize_label("DoS Hulk") == "Suspicious"

def test_pipeline_trains_and_predicts():
    frame = make_demo_dataset(n_samples=240)
    X, y = frame.drop(columns="Label"), frame["Label"]
    model = build_model()
    model.fit(X.iloc[:200], y.iloc[:200])
    predictions = model.predict(X.iloc[200:])
    assert len(predictions) == 40
    assert set(predictions).issubset({"Benign", "Suspicious"})

def test_feature_alignment_ignores_label_and_adds_missing_feature():
    frame = pd.DataFrame({"flow_duration": [1, 2], "Label": ["Benign", "Suspicious"], "extra": [4, 5]})
    result = align_features(frame, ["flow_duration", "missing_feature"])
    assert list(result.columns) == ["flow_duration", "missing_feature"]
    assert result["flow_duration"].tolist() == [1, 2]
    assert result["missing_feature"].isna().all()

def test_feature_alignment_rejects_non_numeric_only():
    with pytest.raises(ValueError):
        align_features(pd.DataFrame({"name": ["alice", "bob"]}), ["flow_duration"])

def test_predict_frame_adds_prediction_columns():
    frame = make_demo_dataset(n_samples=240)
    X, y = frame.drop(columns="Label"), frame["Label"]
    model = build_model()
    model.fit(X.iloc[:200], y.iloc[:200])
    metadata = {"features": list(X.columns)}
    result = predict_frame(X.iloc[200:].copy(), model, metadata)
    assert "SentinelAI_prediction" in result
    assert "suspicious_probability" in result
