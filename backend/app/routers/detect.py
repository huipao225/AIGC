import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.pydantic_schemas.schemas import DetectRequest, ErrorDetail, ErrorResponse
from app.services.detector_service import DetectorService
from app.services.statistical_service import StatisticalService
from app.services.text_processor import chunk_text, clean_text

logger = logging.getLogger(__name__)
router = APIRouter(tags=["detection"])


@router.post(
    "/api/detect",
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def detect(request: Request, body: DetectRequest) -> JSONResponse:
    detector: DetectorService = request.app.state.detector

    if not detector.loaded:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="MODELS_LOADING",
                    message="模型正在加载中，请稍候。",
                ),
            ).model_dump(),
        )

    if len(body.text) > settings.max_text_length:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="TEXT_TOO_LONG",
                    message=f"文本超出最大长度限制 {settings.max_text_length} 字符。",
                ),
            ).model_dump(),
        )

    preview = body.text[:200].replace("\n", "\\n")
    logger.info("检测请求 — 长度=%d 预览: %s", len(body.text), preview)

    t0 = time.time()

    cleaned = clean_text(body.text)
    chunks = chunk_text(cleaned, detector.tokenizer)

    # Chinese RoBERTa classification per chunk
    segments = []
    for ch in chunks:
        result = detector.classify(ch["text_preview"])
        segments.append(
            {
                "start": ch["start"],
                "end": ch["end"],
                "text_preview": ch["text_preview"],
                "score": result["score"],
                "label": result["label"],
            }
        )

    avg_roberta = (
        round(sum(s["score"] for s in segments) / len(segments), 4)
        if segments
        else 0.5
    )

    # Multi-method analysis
    stat = StatisticalService()
    perplexity_result = detector.compute_perplexity(cleaned[:512])
    burstiness_result = stat.compute_burstiness(cleaned)
    chinese_features = stat.compute_chinese_features(cleaned)

    # Ensemble with Chinese-specific weights
    final_score = stat.ensemble_score(
        avg_roberta,
        perplexity_result["score"],
        burstiness_result["score"],
        chinese_features["score"],
    )
    confidence = round(abs(final_score - 0.5) * 2, 4)

    processing_ms = round((time.time() - t0) * 1000, 2)

    return JSONResponse(
        content={
            "status": "success",
            "data": {
                "overall_score": final_score,
                "classification": (
                    "AI-generated" if final_score > 0.5 else "Human-written"
                ),
                "confidence": confidence,
                "breakdown": {
                    "roberta": {"score": avg_roberta},
                    "perplexity": perplexity_result,
                    "burstiness": burstiness_result,
                    "chinese_features": chinese_features,
                },
                "segments": segments,
                "metadata": {
                    "text_length": len(cleaned),
                    "chunks_analyzed": len(chunks),
                    "processing_time_ms": processing_ms,
                },
            },
        }
    )
