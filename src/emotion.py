"""Per-sentence emotion: one of anger / fear / joy / sadness + confidence.

Same reasoning as sentiment.py: one multilingual model instead of a
per-language one. `MilaNLProc/xlm-emo-t` is XLM-T (the same base as the
sentiment model) fine-tuned for emotion, so it slots into the same
lru_cache-singleton pattern with no new dependency and no per-language
branching.

Only 4 labels (no "neutral"/"disgust"/"surprise") -- that's what this model
was trained to output, not a simplification made here.
"""
from functools import lru_cache

_MODEL_NAME = "MilaNLProc/xlm-emo-t"


@lru_cache(maxsize=1)
def _get_pipeline():
    from transformers import pipeline
    return pipeline("text-classification", model=_MODEL_NAME)


def score(text: str) -> dict:
    if not text.strip():
        return {"label": "none", "confidence": 0.0}
    result = _get_pipeline()(text, truncation=True)[0]
    return {"label": result["label"].lower(), "confidence": round(result["score"], 4)}
