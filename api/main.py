"""FastAPI inference service for SentinelAI Advanced."""
from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_artifacts, predict_frame

app = FastAPI(
    title="SentinelAI API",
    description="Educational API for classifying network-flow feature records.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=500)


@app.get("/health")
def health():
    try:
        model, metadata = load_artifacts()
        return {
            "status": "ok",
            "model_loaded": True,
            "features": len(metadata["features"]),
            "decision_threshold": metadata.get("decision_threshold", 0.5),
            "classes": [str(label) for label in model.classes_],
        }
    except FileNotFoundError:
        return {"status": "not_ready", "model_loaded": False}


@app.get("/model")
def model_info():
    try:
        model, metadata = load_artifacts()
        return {
            "project": metadata.get("project", "SentinelAI Advanced"),
            "model_type": metadata.get("model_type", "Unknown"),
            "features": metadata["features"],
            "features_count": len(metadata["features"]),
            "classes": [str(label) for label in model.classes_],
            "data_source": metadata.get(
                "data_source", metadata.get("dataset", "Unknown")
            ),
            "decision_threshold": metadata.get("decision_threshold", 0.5),
            "metrics": metadata.get(
                "metrics", metadata.get("held_out_test_metrics", {})
            ),
            "validation_metrics": metadata.get("validation_metrics", {}),
            "caveat": metadata.get(
                "caveat", metadata.get("warning", "")
            ),
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model, metadata = load_artifacts()
        frame = pd.DataFrame(request.records)
        results = predict_frame(frame, model, metadata)
        return {"predictions": results.to_dict(orient="records")}
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
