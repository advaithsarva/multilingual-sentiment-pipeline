"""End-to-end smoke test with the real models. Slow on first run (downloads
the sentiment model + spaCy models the first time), fast after that since
they're cached. Not run as part of a pre-commit hook for that reason.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import analyze
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
