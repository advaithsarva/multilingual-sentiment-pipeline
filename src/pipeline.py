"""Wires language detection -> segmentation -> sentiment -> NER into one call.

Invariant (the one every stage above is built to preserve): every sentence
span's (start, end) indexes the ORIGINAL document text, never the sentence's
own local string. This is what lets the HTML report and the entity timeline
line entities and sentiment back up to exact character positions instead of
re-searching the text.
"""
from dataclasses import dataclass, field

from src.language import detect_language
from src.segment import segment
from src.sentiment import score as score_sentiment
from src.entities import extract as extract_entities


@dataclass
class SentenceResult:
    text: str
    start: int
    end: int
    sentiment: dict
    entities: list[dict]


@dataclass
class DocumentResult:
    text: str
    language: str
    sentences: list[SentenceResult] = field(default_factory=list)
    label: str = "document"


def analyze(text: str, label: str = "document") -> DocumentResult:
    lang = detect_language(text)
    sentences = [
        SentenceResult(
            text=sent_text,
            start=start,
            end=end,
            sentiment=score_sentiment(sent_text),
            entities=extract_entities(sent_text, lang),
        )
        for sent_text, start, end in segment(text, lang)
    ]
    return DocumentResult(text=text, language=lang, sentences=sentences, label=label)
