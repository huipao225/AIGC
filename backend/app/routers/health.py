import time

import torch
from fastapi import APIRouter, Request

from app.config import settings

router = APIRouter(tags=["health"])
_start_time = time.time()


@router.get("/api/health")
async def health(request: Request) -> dict:
    detector = request.app.state.detector
    return {
        "status": "healthy" if detector.loaded else "loading",
        "version": settings.app_version,
        "models_loaded": detector.get_models_loaded(),
        "gpu_available": torch.cuda.is_available(),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }
