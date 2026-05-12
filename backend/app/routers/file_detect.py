import logging
import time

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.pydantic_schemas.schemas import ErrorDetail, ErrorResponse
from app.services.detector_service import DetectorService
from app.services.statistical_service import StatisticalService
from app.services.text_processor import chunk_text, clean_text
from app.utils.file_parser import FileParseError, extract_text
from app.utils.save_text import save_submission

logger = logging.getLogger(__name__)
router = APIRouter(tags=["file-detection"])

ALLOWED_EXTENSIONS = {"txt", "docx", "pdf"}


@router.post(
    "/api/detect/file",
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def detect_file(request: Request, file: UploadFile = File(...)) -> JSONResponse:
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

    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="UNSUPPORTED_FORMAT",
                    message=f"不支持的文件格式 .{ext}，请上传 .txt / .docx / .pdf 文件。",
                ),
            ).model_dump(),
        )

    content = await file.read()

    try:
        text = extract_text(file.filename, content)
    except FileParseError as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                status="error", error=ErrorDetail(code=e.code, message=e.message)
            ).model_dump(),
        )

    if len(text) > settings.max_text_length:
        text = text[: settings.max_text_length]

    if len(text.strip()) < 10:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                status="error",
                error=ErrorDetail(
                    code="TEXT_TOO_SHORT",
                    message="提取的文本太短（不足 10 个字符），无法进行有效检测。",
                ),
            ).model_dump(),
        )

    preview = text[:200].replace("\n", "\\n")
    logger.info("文件检测 — 文件=%s 长度=%d 预览: %s", file.filename, len(text), preview)

    t0 = time.time()

    cleaned = clean_text(text)
    chunks = chunk_text(cleaned, detector.tokenizer)

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

    stat = StatisticalService()
    perplexity_result = detector.compute_perplexity(cleaned[:512])
    burstiness_result = stat.compute_burstiness(cleaned)
    chinese_features = stat.compute_chinese_features(cleaned)

    final_score = stat.ensemble_score(
        avg_roberta,
        perplexity_result["score"],
        burstiness_result["score"],
        chinese_features["score"],
    )
    confidence = round(abs(final_score - 0.5) * 2, 4)

    processing_ms = round((time.time() - t0) * 1000, 2)

    classification = "AI-generated" if final_score > 0.5 else "Human-written"
    save_submission(text, classification, final_score, file.filename)

    return JSONResponse(
        content={
            "status": "success",
            "data": {
                "overall_score": final_score,
                "classification": classification,
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
                    "source_file": file.filename,
                },
            },
        }
    )
