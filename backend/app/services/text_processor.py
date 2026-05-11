import re

from app.config import settings


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, tokenizer) -> list[dict]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= settings.chunk_max_tokens:
        return [
            {
                "start": 0,
                "end": len(text),
                "text_preview": text[:100],
                "tokens": tokens,
            }
        ]

    chunks = []
    for i in range(0, len(tokens), settings.chunk_stride):
        chunk_tokens = tokens[i : i + settings.chunk_max_tokens]
        chunk_text_decoded = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        char_start = text.find(chunk_text_decoded[:50]) if chunk_text_decoded[:50] else i
        if char_start == -1:
            char_start = i
        chunks.append(
            {
                "start": char_start,
                "end": char_start + len(chunk_text_decoded),
                "text_preview": chunk_text_decoded[:100],
                "tokens": chunk_tokens,
            }
        )
        if i + settings.chunk_max_tokens >= len(tokens):
            break

    return chunks
