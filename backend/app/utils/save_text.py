import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# 北京时间
TZ = timezone(timedelta(hours=8))
SAVE_DIR = Path(__file__).parent.parent.parent / "logs" / "detections"


def save_submission(text: str, classification: str, score: float, source: str = "text") -> str:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{classification}_{source}.txt"
    filepath = SAVE_DIR / filename

    header = (
        f"时间: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"来源: {source}\n"
        f"判定: {classification}\n"
        f"得分: {score}\n"
        f"长度: {len(text)} 字符\n"
        f"{'=' * 50}\n\n"
    )

    filepath.write_text(header + text, encoding="utf-8")
    logger.info("文本已保存: %s", filepath.name)
    return str(filepath)
