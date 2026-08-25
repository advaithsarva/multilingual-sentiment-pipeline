"""Produces the sentiment accuracy numbers in RESULTS.md.

    python scripts/measure_sentiment.py samples/eval_sentiment.tsv
"""
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sentiment import score


def main():
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))

    per_lang = defaultdict(lambda: [0, 0])  # lang -> [correct, total]
    start = time.perf_counter()
    for row in rows:
        pred = score(row["text"])["label"]
        per_lang[row["lang"]][1] += 1
        if pred == row["label"]:
            per_lang[row["lang"]][0] += 1
    elapsed = time.perf_counter() - start

    total_correct = sum(c for c, _ in per_lang.values())
    total = sum(t for _, t in per_lang.values())

    for lang, (correct, n) in sorted(per_lang.items()):
        print(f"{lang}: {correct}/{n} = {correct/n:.2f}")
    print(f"overall: {total_correct}/{total} = {total_correct/total:.2f}")
    print(f"{len(rows)} sentences in {elapsed:.2f}s ({elapsed/len(rows)*1000:.0f}ms/sentence)")


if __name__ == "__main__":
    main()
