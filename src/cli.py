"""CLI: one or more text files in, one HTML report out.

    python -m src.cli samples/*.txt -o report.html
"""
import argparse
import json
from pathlib import Path

from src.pipeline import analyze
from src.report import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="text files to analyze")
    parser.add_argument("-o", "--output", type=Path, default=Path("report.html"))
    parser.add_argument("--json", type=Path, help="also dump raw results as JSON")
    args = parser.parse_args()

    docs = [analyze(f.read_text(encoding="utf-8"), label=f.name) for f in args.files]

    args.output.write_text(render(docs), encoding="utf-8")
    print(f"wrote {args.output} ({len(docs)} document(s))")

    if args.json:
        payload = [
            {
                "label": d.label,
                "language": d.language,
                "sentences": [
                    {"text": s.text, "start": s.start, "end": s.end,
                     "sentiment": s.sentiment, "entities": s.entities}
                    for s in d.sentences
                ],
            }
            for d in docs
        ]
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
