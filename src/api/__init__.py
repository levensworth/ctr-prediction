"""API module for CTR prediction service."""

from .app import app
from .dtos import InvokeRequest, InvokeResponse, PredictionInput, PredictionOutput

__all__ = [
    "app",
    "InvokeRequest",
    "InvokeResponse",
    "PredictionInput",
    "PredictionOutput",
]
