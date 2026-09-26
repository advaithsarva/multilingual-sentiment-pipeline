# Results

## Sentiment accuracy

Command:

```bash
python scripts/measure_sentiment.py samples/eval_sentiment.tsv
```

20 hand-written sentences (8 English, 6 Hindi, 6 Spanish), balanced
positive/negative, unambiguous on purpose — this is a sanity check that the
model works in each language, **not a benchmark**. No Telugu rows: the model
wasn't fine-tuned on Telugu, so there's nothing honest to score it against
yet (see README's Honest limitations).

| Language | Correct / Total | Accuracy |
|---|---|---|
| en | 8/8 | 1.00 |
| es | 6/6 | 1.00 |
| hi | 6/6 | 1.00 |
| **overall** | **20/20** | **1.00** |

Perfect accuracy is expected here — the sentences were written to be
clear-cut, not adversarial. It confirms the pipeline correctly routes each
language through the same model and gets a sane label back; it does not
demonstrate accuracy on ambiguous or mixed-sentiment real-world text, which
would need a labelled dataset like an XLM-T or SemEval split, not 20
hand-written rows.

## Emotion accuracy

Command:

```bash
python scripts/measure_emotion.py samples/eval_emotion.tsv
```

18 hand-written sentences (6 English, 6 Hindi, 6 Spanish), 1-2 per label per
language across anger/joy/fear/sadness -- same "sanity check, not benchmark"
caveat as sentiment above. No Telugu rows, same reason.

| Language | Correct / Total | Accuracy |
|---|---|---|
| en | 6/6 | 1.00 |
| es | 6/6 | 1.00 |
| hi | 6/6 | 1.00 |
| **overall** | **18/18** | **1.00** |

`MilaNLProc/xlm-emo-t` (XLM-T fine-tuned for 4-way emotion) -- same base
architecture as the sentiment model, so it adds a second axis of signal
(anger/fear/joy/sadness) without a second model family or per-language
branching. ~1.5s/sentence on CPU, same ballpark as sentiment, and cached the
same way via `lru_cache` in `emotion.py`.

## Latency

Same sentiment run: 20 sentences in 29.0s on CPU (no GPU used) =
**~1.45s/sentence**. First call in a process pays a one-time model-load
cost; `sentiment.py` and `emotion.py` each cache their pipeline with
`lru_cache` so it's paid once per process, not once per sentence.

## NER spot-check

`tests/test_pipeline.py::test_english_ner_finds_a_person` asserts
`en_core_web_sm` tags "Barack Obama" as `PERSON` in
*"Barack Obama visited Berlin in 2016."* — passes. The demo report
(`samples/report.html`, generated from `samples/doc_en.txt` and
`samples/doc_es.txt`) additionally shows correct `GPE`/`LOC`/`DATE` tags on
real sentences; not scored quantitatively since there's no labelled NER set
here, only spot-checked by reading the output.

## Test suite

```
tests/test_segment.py   5 passed  (offline, offset-invariant checks)
tests/test_pipeline.py  5 passed  (real models: sentiment sign, emotion label, NER, summary, report rendering)
```
