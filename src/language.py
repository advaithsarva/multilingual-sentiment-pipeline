"""Language detection for a whole document.

Invariant: detect_language always returns a two-letter ISO code, never raises,
never returns None. langdetect throws on empty/short/symbol-only text, and a
crash here would kill the whole pipeline over one bad paragraph, so we fall
back to 'en' rather than propagate the exception.
"""
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # langdetect is non-deterministic by default; pin it


def detect_language(text: str) -> str:
    text = text.strip()
    if not text:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"
