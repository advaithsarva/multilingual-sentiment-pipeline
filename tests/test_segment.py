"""Fast, offline tests for the one invariant everything downstream depends on:
sentence spans must index back into the original text exactly.
No network, no model loading -- runs in well under a second.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segment import segment
from src.language import detect_language


def test_offsets_roundtrip_english():
    text = "I love this movie. It was great! Dr. Smith agreed too."
    spans = segment(text, "en")
    assert len(spans) >= 2
    for sent_text, start, end in spans:
        assert text[start:end] == sent_text


def test_offsets_roundtrip_telugu_fallback():
    # pysbd has no Telugu model, this exercises the regex fallback path
    text = "ఇది చాలా బాగుంది. నాకు చాలా ఇష్టం."
    spans = segment(text, "te")
    assert len(spans) == 2
    for sent_text, start, end in spans:
        assert text[start:end] == sent_text


def test_empty_text_gives_no_spans():
    assert segment("", "en") == []
    assert segment("   ", "en") == []


def test_language_detection_never_raises_on_garbage():
    assert detect_language("") == "en"
    assert detect_language("!!! ??? ...") == "en"
    assert detect_language("This is clearly English text.") == "en"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
