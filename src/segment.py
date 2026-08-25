"""Sentence segmentation, one document -> list of (text, start, end) spans.

Invariant: offsets are character offsets into the ORIGINAL text, and
`text[start:end] == sentence` always holds. That's what lets the HTML report
highlight sentences in place instead of re-flowing the document.

pysbd covers en/es/hi (and more) with real linguistic rules. It doesn't ship
a Telugu model, so unsupported languages fall back to a plain regex splitter
on ., !, ? and Telugu's own sentence-ending danda (।). The fallback is worse
on abbreviations and quotes -- documented in README, not hidden.
"""
import re
import pysbd
from pysbd.languages import LANGUAGE_CODES

_SEGMENTERS: dict[str, pysbd.Segmenter] = {}

_FALLBACK_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def _pysbd_segmenter(lang: str) -> pysbd.Segmenter:
    if lang not in _SEGMENTERS:
        _SEGMENTERS[lang] = pysbd.Segmenter(language=lang, clean=False, char_span=True)
    return _SEGMENTERS[lang]


def segment(text: str, lang: str) -> list[tuple[str, int, int]]:
    if not text.strip():
        return []

    if lang in LANGUAGE_CODES:
        spans = _pysbd_segmenter(lang).segment(text)
        return [(s.sent, s.start, s.end) for s in spans]

    spans = []
    pos = 0
    for piece in _FALLBACK_SPLIT.split(text):
        if not piece.strip():
            pos += len(piece)
            continue
        start = text.index(piece, pos)
        end = start + len(piece)
        spans.append((piece, start, end))
        pos = end
    return spans
