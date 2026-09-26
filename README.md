# multilingual-sentiment-pipeline

Feed it a document in English, Hindi, Spanish or (best-effort) Telugu; it
detects the language, splits it into sentences, scores each sentence's
sentiment and emotion, tags named entities, and renders a single
self-contained HTML report with a color-coded heatmap, per-sentence emotion
badges, a document-level verdict, and an entity timeline.

## What it does

```
text --> language detection --> sentence segmentation --> [sentiment, emotion, NER] per sentence --> HTML report
```

- **Language detection** — `langdetect`, seeded for determinism.
- **Sentence segmentation** — `pysbd` (real linguistic rules) for languages
  it supports (en, es, hi, and 20 others); a regex fallback on `. ! ? ।` for
  the rest, including Telugu, which `pysbd` doesn't ship a model for.
- **Sentiment** — one multilingual model,
  [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment),
  fine-tuned on tweets in 8 languages including English, Hindi and Spanish.
  No per-language branching needed for the languages it was tuned on.
- **Emotion** — one multilingual model,
  [`MilaNLProc/xlm-emo-t`](https://huggingface.co/MilaNLProc/xlm-emo-t)
  (same XLM-T base as the sentiment model), classifying each sentence as
  anger, fear, joy or sadness. A second signal alongside sentiment, not a
  replacement for it — a sentence can be `negative` + `fear` or `negative` +
  `anger`, which the polarity score alone can't distinguish.
- **NER** — spaCy's `en_core_web_sm` for English, `xx_ent_wiki_sm`
  (multilingual, Wikipedia-trained) for everything else.
- **Report** — static HTML/CSS/JS, no build step. Sentence background color
  and opacity encode sentiment label + intensity, a small emoji badge shows
  the dominant emotion, entities are highlighted inline, and each document
  gets an overall verdict bar plus an entity-mention timeline.

## Honest limitations

- **Telugu sentiment is unverified.** The model wasn't fine-tuned on Telugu,
  only pretrained on it as one of XLM-R's 100 languages. It runs and returns
  a score, but there's no labelled Telugu eval data behind that score — see
  [RESULTS.md](RESULTS.md).
- **Non-English NER loses DATE.** `xx_ent_wiki_sm` only tags
  PER/ORG/LOC/MISC; `en_core_web_sm` also catches dates. This is a real
  quality gap between English and everything else, not a bug.
- **Telugu segmentation is a regex fallback**, not linguistic rules like the
  other three languages get from `pysbd`. It handles `.`/`!`/`?`/`।` but not
  abbreviations.
- **Emotion has no confirmed Hindi/Telugu training data.** `xlm-emo-t`'s
  labelled data (the XLM-EMO paper's 19 languages) is mostly European; Hindi
  and Telugu get whatever XLM-T's pretraining transfers zero-shot, same
  caveat as Telugu sentiment. Our eval's 1.00 Hindi score (RESULTS.md) is 6
  unambiguous sentences, not proof of real Hindi emotion accuracy.
- **Emotion has only 4 labels** (anger/fear/joy/sadness) — no "neutral", no
  "disgust"/"surprise". That's what the model outputs, not a simplification
  made here; a calm sentence still gets forced into one of the four.

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm

python -m src.cli samples/doc_en.txt samples/doc_es.txt samples/doc_hi.txt -o report.html
```

Open `report.html`. First run downloads the sentiment model (~1.1GB) and the
emotion model (~1.1GB) from Hugging Face; both are cached locally after
that.

## Tests

```bash
python tests/test_segment.py   # offline, <1s: the offset invariant
python tests/test_pipeline.py  # needs the real models, ~10s once cached
```

## Results

See [RESULTS.md](RESULTS.md) for measured sentiment accuracy per language
and per-sentence latency, with the exact command that produced each number.

## Project origin

One of three portfolio projects specced in `domains/nlp/`, built the same
day as [graph-rag-knowledge-system](https://github.com/advaithsarva/graph-rag-knowledge-system)
and [fact-checking-agent](https://github.com/advaithsarva/fact-checking-agent).
