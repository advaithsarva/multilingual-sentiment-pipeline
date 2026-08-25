# multilingual-sentiment-pipeline

Feed it a document in English, Hindi, Spanish or (best-effort) Telugu; it
detects the language, splits it into sentences, scores each sentence's
sentiment, tags named entities, and renders a single self-contained HTML
report with a color-coded heatmap and an entity timeline.

## What it does

```
text --> language detection --> sentence segmentation --> [sentiment, NER] per sentence --> HTML report
```

- **Language detection** — `langdetect`, seeded for determinism.
- **Sentence segmentation** — `pysbd` (real linguistic rules) for languages
  it supports (en, es, hi, and 20 others); a regex fallback on `. ! ? ।` for
  the rest, including Telugu, which `pysbd` doesn't ship a model for.
- **Sentiment** — one multilingual model,
  [`cardiffnlp/twitter-xlm-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment),
  fine-tuned on tweets in 8 languages including English, Hindi and Spanish.
  No per-language branching needed for the languages it was tuned on.
- **NER** — spaCy's `en_core_web_sm` for English, `xx_ent_wiki_sm`
  (multilingual, Wikipedia-trained) for everything else.
- **Report** — static HTML/CSS/JS, no build step. Sentence background color
  and opacity encode sentiment label + intensity; entities are highlighted
  inline; each document gets an entity-mention timeline.

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

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm

python -m src.cli samples/doc_en.txt samples/doc_es.txt -o report.html
```

Open `report.html`. First run downloads the ~1.1GB sentiment model from
Hugging Face; it's cached locally after that.

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
