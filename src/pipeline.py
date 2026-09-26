"""Wires language detection -> segmentation -> sentiment -> NER into one call.

Invariant (the one every stage above is built to preserve): every sentence
span's (start, end) indexes the ORIGINAL document text, never the sentence's
own local string. This is what lets the HTML report and the entity timeline
line entities and sentiment back up to exact character positions instead of
re-searching the text.
"""
from collections import Counter
from dataclasses import dataclass, field

from src.language import detect_language
from src.segment import segment
from src.sentiment import score as score_sentiment
from src.emotion import score as score_emotion
from src.entities import extract as extract_entities


@dataclass
class SentenceResult:
    text: str
    start: int
    end: int
    sentiment: dict
    emotion: dict
    entities: list[dict]


@dataclass
class DocumentResult:
    text: str
    language: str
    sentences: list[SentenceResult] = field(default_factory=list)
    label: str = "document"

    def summary(self) -> dict:
        """Document-level verdict: mean sentence intensity + label breakdown.

        Not stored as a field -- it's cheap to derive from `sentences` and
        this way it can never drift out of sync with them.
        """
        if not self.sentences:
            return {"score": 0.0, "verdict": "neutral", "positive": 0, "negative": 0, "neutral": 0, "emotions": {}}
        intensities = [s.sentiment["intensity"] for s in self.sentences]
        score = round(sum(intensities) / len(intensities), 4)
        counts = Counter(s.sentiment["label"] for s in self.sentences)
        if score > 0.15:
            verdict = "mostly positive"
        elif score < -0.15:
            verdict = "mostly negative"
        else:
            verdict = "mixed"
        return {
            "score": score,
            "verdict": verdict,
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral": counts["neutral"],
            "emotions": dict(Counter(s.emotion["label"] for s in self.sentences)),
        }


def analyze(text: str, label: str = "document") -> DocumentResult:
    lang = detect_language(text)
    sentences = [
        SentenceResult(
            text=sent_text,
            start=start,
            end=end,
            sentiment=score_sentiment(sent_text),
            emotion=score_emotion(sent_text),
            entities=extract_entities(sent_text, lang),
        )
        for sent_text, start, end in segment(text, lang)
    ]
    return DocumentResult(text=text, language=lang, sentences=sentences, label=label)
