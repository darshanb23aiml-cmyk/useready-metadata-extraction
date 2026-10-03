"""Run extraction on a folder and (if labels exist) score it.

    python -m src.run --split train
    python -m src.run --split test
"""
import argparse
from pathlib import Path

import pandas as pd

from .evaluate import load_labels, mismatches, per_field_recall
import json

from . import extractor
from .extractor import QuotaExhausted, extract
from .loader import doc_id
from .schema import CSV_COLUMNS

ROOT = Path(__file__).resolve().parent.parent


def predict_one(f: Path) -> dict:
    """Extract one file, reusing a saved result if the prompt has not changed."""
    cache_dir = ROOT / "outputs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfile = cache_dir / f"{doc_id(f)}_{extractor.prompt_hash()}.json"
    if cfile.exists():
        print(f"Cached     {f.name}")
        return json.loads(cfile.read_text())["fields"]
    print(f"Extracting {f.name} ...")
    fields = extract(f).model_dump()
    cfile.write_text(json.dumps({"model": extractor.LAST_MODEL_USED or extractor.CLAUDE_MODEL,
                                 "fields": fields}, indent=2))
    return fields


def predict_folder(folder: Path, only=None) -> pd.DataFrame:
    """only: optional set of doc ids to process (saves API calls on unlabeled files)."""
    rows, skipped = {}, []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in (".docx", ".png"):
            continue
        if only is not None and doc_id(f) not in only:
            continue
        try:
            meta = predict_one(f)
        except QuotaExhausted as e:
            print(f"\nSTOPPED: {e}")
            skipped = [x.name for x in sorted(folder.iterdir())
                       if x.suffix.lower() in (".docx", ".png") and doc_id(x) not in rows
                       and (only is None or doc_id(x) in only)]
            break
        rows[doc_id(f)] = {CSV_COLUMNS[k]: v for k, v in meta.items()}
    if skipped:
        print(f"Not processed yet ({len(skipped)}): {skipped}")
        print("Scores below cover only the processed files.\n")
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "File Name"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    args = ap.parse_args()

    labels = load_labels(ROOT / "data" / "train.csv", ROOT / "data" / "test.csv")
    pred = predict_folder(ROOT / "data" / args.split, only=set(labels.index))
    if pred.empty:
        return
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    pred.to_csv(out / f"predictions_{args.split}.csv")

    table = per_field_recall(pred, labels)
    table.to_csv(out / f"recall_{args.split}.csv", index=False)
    print("\n", table.to_string(index=False))

    wrong = mismatches(pred, labels)
    if len(wrong):
        print("\nMismatches:\n", wrong.to_string(index=False))


if __name__ == "__main__":
    main()