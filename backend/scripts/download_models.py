"""Pre-download HuggingFace models during Docker build."""
import os
import sys

# Use HF mirror in China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

MODELS = [
    ("classifier", "Hello-SimpleAI/chatgpt-detector-roberta-chinese"),
    ("causal", "distilgpt2"),
]

for name, model_id in MODELS:
    print(f"Downloading {name}: {model_id} ...")
    try:
        if name == "classifier":
            AutoTokenizer.from_pretrained(model_id)
            AutoModelForSequenceClassification.from_pretrained(model_id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            AutoModelForCausalLM.from_pretrained(model_id)
        print(f"  {name} downloaded successfully.")
    except Exception as e:
        print(f"  ERROR downloading {name}: {e}", file=sys.stderr)

print("All models downloaded.")
