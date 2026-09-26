"""Render DocumentResults to one self-contained HTML file.

Static HTML+CSS, no build step, no JS framework: the spec's "interactive
heatmap" needs hover tooltips and a document toggle, both of which are a
handful of lines of vanilla JS. A React app would need a bundler, a dev
server and a build step to show the same thing.
"""
import html
from src.pipeline import DocumentResult

_SENTIMENT_COLORS = {"positive": "76, 175, 80", "negative": "244, 67, 54", "neutral": "158, 158, 158"}

_EMOTION_EMOJI = {"anger": "\U0001F620", "fear": "\U0001F628", "joy": "\U0001F604", "sadness": "\U0001F622"}

_ENTITY_COLORS = {
    "PERSON": "#fff3cd", "PER": "#fff3cd",
    "ORG": "#cce5ff",
    "GPE": "#d4edda", "LOC": "#d4edda",
    "DATE": "#f8d7da",
    "MISC": "#e2d9f3",
}


def _mark_entities(text: str, entities: list[dict]) -> str:
    """Wrap entity spans in <mark>, escaping everything else."""
    out, pos = [], 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        out.append(html.escape(text[pos:ent["start"]]))
        color = _ENTITY_COLORS.get(ent["label"], "#eee")
        out.append(
            f'<mark style="background:{color}" title="{html.escape(ent["label"])}">'
            f'{html.escape(text[ent["start"]:ent["end"]])}</mark>'
        )
        pos = ent["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _sentence_html(sent) -> str:
    rgb = _SENTIMENT_COLORS.get(sent.sentiment["label"], _SENTIMENT_COLORS["neutral"])
    alpha = max(0.08, abs(sent.sentiment["intensity"]))
    emoji = _EMOTION_EMOJI.get(sent.emotion["label"], "")
    tooltip = f'{sent.sentiment["label"]} ({sent.sentiment["intensity"]:+.2f}) · {sent.emotion["label"]}'
    badge = f'<sup title="{sent.emotion["label"]} ({sent.emotion["confidence"]:.2f})">{emoji}</sup>' if emoji else ""
    return (
        f'<span class="sent" style="background:rgba({rgb},{alpha})" '
        f'title="{html.escape(tooltip)}">{_mark_entities(sent.text, sent.entities)}{badge}</span>'
    )


def _timeline_html(doc: DocumentResult) -> str:
    seen: dict[str, list[float]] = {}
    for i, sent in enumerate(doc.sentences):
        for ent in sent.entities:
            seen.setdefault(ent["text"], []).append(i)
    if not seen:
        return "<p><em>No entities found.</em></p>"
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{', '.join(str(i) for i in idxs)}</td></tr>"
        for name, idxs in sorted(seen.items(), key=lambda kv: -len(kv[1]))
    )
    return f'<table class="timeline"><tr><th>Entity</th><th>Sentence #</th></tr>{rows}</table>'


def _summary_html(doc: DocumentResult) -> str:
    s = doc.summary()
    total = len(doc.sentences) or 1
    pct = {k: s[k] / total * 100 for k in ("positive", "neutral", "negative")}
    bar = "".join(
        f'<span style="width:{pct[k]:.1f}%;background:rgba({_SENTIMENT_COLORS[k]},0.7)"></span>'
        for k in ("positive", "neutral", "negative") if pct[k]
    )
    emotions = "".join(
        f'{_EMOTION_EMOJI.get(label, "")} {html.escape(label)} &times;{n} &nbsp; '
        for label, n in sorted(s["emotions"].items(), key=lambda kv: -kv[1])
    )
    return (
        f'<div class="summary"><strong>Overall: {s["verdict"]} ({s["score"]:+.2f})</strong>'
        f'<div class="bar">{bar}</div>'
        f'<small>{s["positive"]} positive &middot; {s["neutral"]} neutral &middot; {s["negative"]} negative</small>'
        f'<div><small>{emotions}</small></div></div>'
    )


_STYLE = """
body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; }
.sent { padding: 0.1rem 0; }
mark { padding: 0 2px; border-radius: 2px; }
.doc { border: 1px solid #ddd; border-radius: 6px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; }
.doc h2 { margin-top: 0; font-size: 1.1rem; color: #444; }
table.timeline { border-collapse: collapse; width: 100%; }
table.timeline td, table.timeline th { border: 1px solid #ddd; padding: 4px 8px; text-align: left; font-size: 0.9rem; }
.legend span { padding: 2px 8px; border-radius: 3px; margin-right: 8px; font-size: 0.85rem; }
.summary { margin-bottom: 0.75rem; }
.summary .bar { display: flex; height: 10px; border-radius: 4px; overflow: hidden; margin: 4px 0; background: #eee; }
.summary .bar span { display: block; }
.summary small { color: #666; }
"""


def render(docs: list[DocumentResult], title: str = "Sentiment Report") -> str:
    """Multi-document report: one heatmap + entity timeline per document."""
    sections = []
    for doc in docs:
        body = "".join(_sentence_html(s) for s in doc.sentences)
        sections.append(
            f'<div class="doc"><h2>{html.escape(doc.label)} '
            f'<small>({html.escape(doc.language)}, {len(doc.sentences)} sentences)</small></h2>'
            f'{_summary_html(doc)}'
            f'<p>{body}</p>{_timeline_html(doc)}</div>'
        )
    legend = "".join(
        f'<span style="background:rgba({rgb},0.4)">{label}</span>'
        for label, rgb in _SENTIMENT_COLORS.items()
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><p class='legend'>{legend}</p>"
        f"{''.join(sections)}</body></html>"
    )
