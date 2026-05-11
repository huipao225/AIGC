import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings

logger = logging.getLogger(__name__)


class DetectorService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier: AutoModelForSequenceClassification | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.perplexity_model: AutoModelForCausalLM | None = None
        self.perplexity_tokenizer: AutoTokenizer | None = None
        self._loaded = False

    def load_models(self) -> None:
        start = time.time()
        logger.info("Loading models on %s ...", self.device)

        # Primary: Chinese RoBERTa classifier
        logger.info("Loading Chinese classifier: %s", settings.model_primary)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_primary, cache_dir=settings.model_cache_dir
        )
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            settings.model_primary, cache_dir=settings.model_cache_dir
        )
        self.classifier.to(self.device).eval()

        # Perplexity model (DistilGPT-2, lightweight reference)
        logger.info("Loading perplexity model: distilgpt2")
        self.perplexity_tokenizer = AutoTokenizer.from_pretrained(
            "distilgpt2", cache_dir=settings.model_cache_dir
        )
        self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token
        self.perplexity_model = AutoModelForCausalLM.from_pretrained(
            "distilgpt2", cache_dir=settings.model_cache_dir
        )
        self.perplexity_model.to(self.device).eval()

        self._loaded = True
        elapsed = time.time() - start
        logger.info("All models loaded in %.1fs", elapsed)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @torch.no_grad()
    def classify(self, text: str) -> dict:
        if not self._loaded:
            raise RuntimeError("Models not loaded yet")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.chunk_max_tokens,
        ).to(self.device)
        outputs = self.classifier(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        ai_prob = round(probs[0, 1].item(), 4)
        return {
            "score": ai_prob,
            "label": "AI-generated" if ai_prob > 0.5 else "Human-written",
        }

    @torch.no_grad()
    def compute_perplexity(self, text: str) -> dict:
        if not self._loaded:
            raise RuntimeError("Models not loaded yet")
        inputs = self.perplexity_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        perplexity = round(torch.exp(torch.tensor(loss)).item(), 2)

        if perplexity < 10:
            score = 0.9
        elif perplexity < 30:
            score = 0.8
        elif perplexity < 60:
            score = 0.65
        elif perplexity < 100:
            score = 0.5
        elif perplexity < 200:
            score = 0.35
        elif perplexity < 400:
            score = 0.2
        else:
            score = 0.1

        return {"perplexity": perplexity, "score": round(score, 4)}

    def get_models_loaded(self) -> list[str]:
        models = []
        if self.classifier is not None:
            models.append(settings.model_primary)
        if self.perplexity_model is not None:
            models.append("distilgpt2")
        return models
