"""Named entity recognition, one document -> list of (text, label, start, end).

English gets spaCy's dedicated en_core_web_sm model (trained on English news
text, decent precision). Every other language falls back to xx_ent_wiki_sm,
spaCy's multilingual NER trained on Wikipedia -- it only tags PER/LOC/ORG/MISC
(no DATE, unlike en_core_web_sm), which is a real quality gap the README
states rather than hides.
"""
from functools import lru_cache

_MULTILINGUAL_MODEL = "xx_ent_wiki_sm"
_ENGLISH_MODEL = "en_core_web_sm"


@lru_cache(maxsize=2)
def _get_nlp(lang: str):
    import spacy
    model = _ENGLISH_MODEL if lang == "en" else _MULTILINGUAL_MODEL
    return spacy.load(model, disable=["parser", "tagger", "lemmatizer"])


def extract(text: str, lang: str) -> list[dict]:
    if not text.strip():
        return []
    doc = _get_nlp(lang)(text)
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
