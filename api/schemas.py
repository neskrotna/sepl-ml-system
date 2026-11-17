
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WineInput(BaseModel):
    """Input measurements for a white wine sample."""

    model_config = ConfigDict(extra="forbid")

    fixed_acidity: float = Field(ge=0)
    volatile_acidity: float = Field(ge=0)
    citric_acid: float = Field(ge=0)
    residual_sugar: float = Field(ge=0)
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0)
    total_sulfur_dioxide: float = Field(ge=0)
    density: float = Field(gt=0)
    pH: float = Field(ge=0, le=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Response returned after submitting a prediction."""

    task_id: str
    status: Literal["queued"]


class ResultResponse(BaseModel):
    """Response containing the current prediction status."""

    task_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    prediction: str | None = None
    class_id: int | None = None
    confidence: float | None = None
    error: str | None = None