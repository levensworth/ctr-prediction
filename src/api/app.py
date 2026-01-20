"""FastAPI application for CTR prediction service."""

import time
import uuid

from fastapi import FastAPI, HTTPException

from src.api.config import app_config
from src.api.service import CTRPredictionService
from src.pipelines.prediction import create_prediction_pipeline

from .dtos import InvokeRequest, InvokeResponse, PredictionOutput



app = FastAPI(
    title="CTR Prediction API",
    description="API for predicting Click-Through Rate for publication-campaign pairs",
    version="1.0.0",
)

# initialize the prediction pipeline and service
pipeline = create_prediction_pipeline(app_config.artifacts_dir)
service = CTRPredictionService(pipeline, app_config.feature_set_name)

@app.post("/invoke", response_model=InvokeResponse)
def invoke(request: InvokeRequest) -> InvokeResponse:
    """Generate CTR predictions for publication-campaign pairs."""
    start_time = time.perf_counter()

    predictions: list[PredictionOutput] = []
    
    # TODO: maybe have a special endpoint for single predictions?
    # errors: list[str] = []

    # for idx, input_data in enumerate(request.data):
    #     try:
    #         prediction = service.predict_single(input_data)
    #         predictions.append(prediction)
    #     except ValueError as e:
    #         errors.append(f"Item {idx}: {str(e)}")

    # if errors and not predictions:
    #     raise HTTPException(
    #         status_code=404,
    #         detail=f"No predictions could be generated: {'; '.join(errors)}",
    #     )

    predictions = service.predict_batch(request.data)

    elapsed_seconds = time.perf_counter() - start_time

    return InvokeResponse(
        predictions=predictions,
        execution_time_ms=elapsed_seconds * 1000,
        response_id=uuid.uuid4(),
    )

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}




