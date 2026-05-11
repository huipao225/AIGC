import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class FileParseError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def extract_text(filename: str, content: bytes) -> str:
    if len(content) > MAX_FILE_SIZE:
        raise FileParseError("FILE_TOO_LARGE", "文件大小超过 10MB 限制。")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "txt":
        return _extract_txt(content)
    elif ext == "docx":
        return _extract_docx(content)
    elif ext == "pdf":
        return _extract_pdf(content)
    else:
        raise FileParseError(
            "UNSUPPORTED_FORMAT",
            f"不支持的文件格式 .{ext}，请上传 .txt / .docx / .pdf 文件。",
        )


def _extract_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("gbk", errors="replace")


def _extract_docx(content: bytes) -> str:
    try:
        from docx import Document

        doc = Document(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            raise FileParseError(
                "EMPTY_DOCUMENT", "文档中未提取到文本内容，请检查文件。"
            )
        return "\n".join(paragraphs)
    except FileParseError:
        raise
    except Exception as e:
        logger.error("Failed to parse docx: %s", e)
        raise FileParseError("PARSE_ERROR", "无法解析 .docx 文件，文件可能已损坏。")


def _extract_pdf(content: bytes) -> str:
    try:
        import fitz

        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text.strip())
        doc.close()
        if not pages:
            raise FileParseError(
                "EMPTY_DOCUMENT", "PDF 中未提取到文本内容，可能是扫描件或图片。"
            )
        return "\n".join(pages)
    except FileParseError:
        raise
    except Exception as e:
        logger.error("Failed to parse pdf: %s", e)
        raise FileParseError(
            "PARSE_ERROR", "无法解析 .pdf 文件，文件可能已损坏或加密。"
        )
