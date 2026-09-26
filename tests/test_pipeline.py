"""End-to-end smoke test with the real models. Slow on first run (downloads
the sentiment model + spaCy models the first time), fast after that since
they're cached. Not run as part of a pre-commit hook for that reason.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import analyze, DocumentResult, SentenceResult
from src.report import render


def test_positive_and_negative_sentences_get_opposite_signs():
    doc = analyze("I absolutely love this product. This is the worst thing I ever bought.")
    assert len(doc.sentences) == 2
    assert doc.sentences[0].sentiment["label"] == "positive"
    assert doc.sentences[1].sentiment["label"] == "negative"
    assert doc.sentences[0].sentiment["intensity"] > 0
    assert doc.sentences[1].sentiment["intensity"] < 0


def test_english_ner_finds_a_person():
    doc = analyze("Barack Obama visited Berlin in 2016.")
    labels = {e["label"] for s in doc.sentences for e in s.entities}
    assert "PERSON" in labels


def test_summary_verdict_from_intensities():
    """Offline -- no model needed, just exercises DocumentResult.summary()."""
    doc = DocumentResult(text="", language="en", sentences=[
        SentenceResult(text="a", start=0, end=1, sentiment={"label": "positive", "intensity": 0.9},
                       emotion={"label": "joy", "confidence": 0.8}, entities=[]),
        SentenceResult(text="b", start=1, end=2, sentiment={"label": "negative", "intensity": -0.1},
                       emotion={"label": "sadness", "confidence": 0.6}, entities=[]),
    ])
    s = doc.summary()
    assert s["verdict"] == "mostly positive"
    assert s["positive"] == 1 and s["negative"] == 1
    assert s["score"] == 0.4
    assert s["emotions"] == {"joy": 1, "sadness": 1}


def test_angry_sentence_gets_anger_emotion():
    doc = analyze("I am furious and disgusted, this is an outrage!")
    assert doc.sentences[0].emotion["label"] == "anger"


def test_report_renders_all_documents():
    docs = [analyze("Good news today.", label="a.txt"), analyze("Bad news today.", label="b.txt")]
    out = render(docs)
    assert "a.txt" in out and "b.txt" in out
    assert out.count('class="doc"') == 2


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
