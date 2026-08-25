"""Per-sentence sentiment: polarity label + signed intensity in [-1, 1].

One multilingual model (XLM-R fine-tuned on tweets in 8 languages including
English, Hindi and Spanish) covers every sentence regardless of language, so
there's no per-language branching here -- that's the point of using a
multilingual model instead of four separate ones.

Telugu isn't one of the model's 8 fine-tuning languages. XLM-R's pretraining
still covers it, so it runs and returns a score, but accuracy on Telugu is
unverified -- RESULTS.md reports English/Hindi/Spanish numbers only and calls
Telugu out as best-effort. That's the honest way to ship a "supports 4
languages" feature without 4 languages of labelled eval data.
"""
from functools import lru_cache

_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_LABEL_TO_SIGN = {"negative": -1, "neutral": 0, "positive": 1}


@lru_cache(maxsize=1)
def _get_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=_MODEL_NAME)


def score(text: str) -> dict:
    if not text.strip():
        return {"label": "neutral", "intensity": 0.0}
    result = _get_pipeline()(text, truncation=True)[0]
    label = result["label"].lower()
    sign = _LABEL_TO_SIGN.get(label, 0)
    return {"label": label, "intensity": round(sign * result["score"], 4)}
