from typing import Optional

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)


class SegmentResult(BaseModel):
    start: int
    end: int
    text_preview: str
    score: float
    label: str


class DetectData(BaseModel):
    overall_score: float
    classification: str
    confidence: float
    breakdown: dict
    segments: list[SegmentResult]
    metadata: dict


class DetectResponse(BaseModel):
    status: str = "success"
    data: DetectData


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
    gpu_available: bool
