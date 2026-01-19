"""Data Transfer Objects for the CTR prediction API."""

from uuid import UUID

from pydantic import BaseModel, Field
import pydantic


class PredictionInput(BaseModel):
    """Single prediction request input."""

    publication_id: UUID
    campaign_id: UUID


class InvokeRequest(BaseModel):
    """Request payload for the /invoke endpoint."""

    data: list[PredictionInput] = Field(
        ..., min_length=1, description="List of publication-campaign pairs to predict"
    )


class PredictionOutput(BaseModel):
    """Single prediction result."""

    estimated_ctr: float
    model_id: str


class InvokeResponse(BaseModel):
    """Response payload from the /invoke endpoint."""

    predictions: list[PredictionOutput]
    execution_time_ms: float 
    response_id: UUID

    @pydantic.field_validator("execution_time_ms", mode="before")
    def round_execution_time_ms(cls, v: float) -> float:
        return round(v, 2)
