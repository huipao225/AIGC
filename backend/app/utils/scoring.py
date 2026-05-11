import math


def compute_burstiness(sentences: list[str]) -> dict:
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 3:
        return {"cv": 0.0, "score": 0.5}

    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std = math.sqrt(variance)
    cv = round(std / mean, 4) if mean > 0 else 0.0

    score = 1.0 - min(cv / 1.5, 1.0)
    return {"cv": cv, "score": round(score, 4)}


def ensemble_score(roberta_score: float, burstiness_score: float) -> float:
    return round(0.75 * roberta_score + 0.25 * burstiness_score, 4)


def split_sentences(text: str) -> list[str]:
    import re

    return re.split(r"(?<=[.!?。！？\n])\s*", text)
