import logging
import math
import re

import numpy as np

logger = logging.getLogger(__name__)

# === Chinese AI Writing Pattern Libraries ===

# Category 1: Structured transition chains (AI hallmark)
AI_TRANSITION_CHAINS = [
    "首先", "其次", "再次", "最后", "此外", "另外", "总之",
    "综上所述", "总而言之", "简而言之", "换句话说", "也就是说",
    "因此", "所以", "因而", "从而", "进而", "可见",
    "同时", "与此同时", "另一方面", "与此相对",
    "不仅", "而且", "并且", "以及",
    "譬如", "例如", "比如", "即", "诸如",
]

# Category 2: Balanced argument markers
AI_BALANCE_MARKERS = [
    "一方面", "另一方面", "换言之", "也就是说",
    "从某种意义", "某种程度上",
]

# Category 3: Formal academic formulaic openers
AI_OPENERS = [
    "在", "从", "通过", "随着", "基于", "针对", "对于", "关于",
    "根据", "按照", "作为",
]

# Category 4: Overused academic buzzwords (per 100 chars)
AI_BUZZWORDS = [
    "赋能", "落地", "闭环", "抓手", "对标", "对齐", "打通",
    "深度", "广度", "范式", "维度", "层面", "领域", "视角",
    "核心", "关键", "重要", "显著", "有效", "充分", "全面",
    "数字化", "智能化", "信息化", "网络化",
    "发展", "突破", "创新", "提升", "优化", "推动", "促进",
    "挑战", "机遇", "趋势", "格局",
]

# Category 5: Evidence-claim formulaic patterns
AI_EVIDENCE_PATTERNS = [
    "研究表明", "数据表明", "实践表明", "调查显示",
    "事实证明", "这说明", "这意味着", "足以说明",
    "由此可见", "可以看出", "不难发现", "显而易见",
]

# Category 6: Transition words (per sentence count)
AI_CONJUNCTIONS = [
    "然而", "但是", "不过", "尽管", "虽然",
    "因为", "由于", "所以", "因此", "因而",
    "如果", "那么", "只有", "只要",
    "并且", "而且", "不仅", "还",
    "同时", "也", "又", "此外",
]

# Ensemble weights (sum = 1.0)
W_ROBERTA = 0.15       # Minimal — unreliable on Chinese academic text
W_PERPLEXITY = 0.05     # Trace — English model, near-zero Chinese signal
W_BURSTINESS = 0.15     # Sentence length uniformity
W_CHINESE = 0.65        # Primary — best Chinese AI indicator


class StatisticalService:
    @staticmethod
    def compute_burstiness(text: str) -> dict:
        sentences = _split_chinese_sentences(text)
        lengths = [len(s) for s in sentences if len(s) >= 2]
        if len(lengths) < 3:
            return {"cv": 0.0, "score": 0.5}

        mean = np.mean(lengths)
        std = np.std(lengths)
        cv = round(float(std / mean) if mean > 0 else 0.0, 4)

        # Chinese AI: CV 0.2-0.5; human: 0.4-1.2
        # Map: lower CV → higher AI score
        if cv < 0.25:
            score = 0.85
        elif cv < 0.35:
            score = 0.75
        elif cv < 0.45:
            score = 0.60
        elif cv < 0.55:
            score = 0.45
        elif cv < 0.70:
            score = 0.30
        else:
            score = 0.15

        return {"cv": cv, "score": round(score, 4)}

    @staticmethod
    def compute_chinese_features(text: str) -> dict:
        """Comprehensive Chinese-specific AIGC detection features."""
        sentences = _split_chinese_sentences(text)
        n_sentences = max(len(sentences), 1)
        total_chars = max(len(text.replace("\n", "").replace(" ", "")), 1)

        # 1. Discourse marker density (per sentence)
        marker_count = sum(text.count(m) for m in AI_TRANSITION_CHAINS)
        marker_density = round(marker_count / n_sentences, 4)

        # 2. Balance marker density
        balance_count = sum(text.count(m) for m in AI_BALANCE_MARKERS)
        balance_density = round(balance_count / n_sentences, 4)

        # 3. Paragraph opener ratio
        opener_count = sum(
            1 for s in sentences if s.strip() and s.strip()[0] in AI_OPENERS
        )
        opener_ratio = round(opener_count / n_sentences, 4)

        # 4. Buzzword density (per 100 chars)
        buzzword_count = sum(text.count(t) for t in AI_BUZZWORDS)
        buzzword_density = round(buzzword_count / total_chars * 100, 4)

        # 5. Evidence-pattern density
        evidence_count = sum(text.count(p) for p in AI_EVIDENCE_PATTERNS)
        evidence_density = round(evidence_count / n_sentences, 4)

        # 6. Conjunction density (per sentence)
        conj_count = sum(text.count(c) for c in AI_CONJUNCTIONS)
        conj_density = round(conj_count / n_sentences, 4)

        # 7. Paragraph length uniformity
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if len(paragraphs) >= 2:
            para_lengths = [len(p) for p in paragraphs]
            para_cv = float(np.std(para_lengths) / np.mean(para_lengths)) if np.mean(para_lengths) > 0 else 0
        else:
            para_cv = 0.0
        para_uniformity = 1.0 - min(para_cv / 0.6, 1.0)  # Uniform paragraphs → AI

        # 8. Average sentences per paragraph
        s_per_para = n_sentences / max(len(paragraphs), 1)

        # 9. Em-dash / colon frequency (AI uses more structured punctuation)
        dash_colon = text.count("——") + text.count("：")
        structured_punct_density = round(dash_colon / total_chars * 100, 4)

        # === Sub-scores ===
        # Marker sub-score (transition chains + balance markers)
        marker_score = min(marker_density / 0.5, 1.0) if n_sentences >= 3 else marker_density / 0.5
        balance_score = min(balance_density / 0.2, 1.0)

        # Opener sub-score
        opener_score = min(opener_ratio / 0.3, 1.0)

        # Buzzword sub-score
        buzzword_score = min(buzzword_density / 4.0, 1.0)

        # Evidence pattern sub-score
        evidence_score = min(evidence_density / 0.3, 1.0)

        # Conjunction sub-score
        conj_score = min(conj_density / 0.4, 1.0)

        # Structure sub-score
        structure_score = 0.5 * para_uniformity + 0.3 * min(s_per_para / 3.0, 1.0) + 0.2 * min(structured_punct_density / 2.0, 1.0)

        # Weighted combination of sub-scores
        combined = (
            0.25 * marker_score
            + 0.10 * balance_score
            + 0.15 * opener_score
            + 0.20 * buzzword_score
            + 0.10 * evidence_score
            + 0.10 * conj_score
            + 0.10 * structure_score
        )

        # Calibration boost: if 3+ sub-scores are high, it's very likely AI
        high_subs = sum(
            1 for s in [marker_score, balance_score, opener_score, buzzword_score, evidence_score, conj_score]
            if s > 0.5
        )
        if high_subs >= 3:
            combined = max(combined, 0.65)
        if high_subs >= 4:
            combined = max(combined, 0.80)

        return {
            "marker_density": marker_density,
            "opener_ratio": opener_ratio,
            "buzzword_density": buzzword_density,
            "sentences_per_para": round(s_per_para, 2),
            "evidence_density": evidence_density,
            "conj_density": conj_density,
            "score": round(min(combined, 1.0), 4),
        }

    @staticmethod
    def ensemble_score(
        roberta_score: float,
        perplexity_score: float,
        burstiness_score: float,
        chinese_features_score: float,
    ) -> float:
        return round(
            W_ROBERTA * roberta_score
            + W_PERPLEXITY * perplexity_score
            + W_BURSTINESS * burstiness_score
            + W_CHINESE * chinese_features_score,
            4,
        )


def _split_chinese_sentences(text: str) -> list[str]:
    """Split Chinese text into sentences."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences = re.split(r"(?<=[。！？!?\n])\s*", text)
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > 120:
            sub = re.split(r"(?<=[；;，,])", s)
            result.extend(seg.strip() for seg in sub if seg.strip())
        else:
            result.append(s)
    return result
