"""Per-field evaluation.

Definitions (per field, over documents that have a reference label):
  Recall    = exact matches / documents                (the metric asked for in the assignment;
                                                        a correctly empty field counts as a match)
  Precision = correct non-empty answers / non-empty answers given
              (an empty answer is "no answer", so it lowers recall but not precision)
  F1        = 2 * Precision * Recall / (Precision + Recall)

"strict" = exact string match after trimming spaces.
"lenient" = also ignores letter case and repeated spaces.
"""
import pandas as pd

from .schema import CSV_COLUMNS


def _norm_strict(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _norm_lenient(v) -> str:
    return " ".join(_norm_strict(v).casefold().split())


def load_labels(*csv_paths) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str) for p in csv_paths]
    df = pd.concat(frames).drop_duplicates(subset="File Name")
    return df.set_index("File Name")


def _scores(correct: int, correct_nonempty: int, nonempty_preds: int, n: int):
    recall = correct / n if n else 0.0
    precision = correct_nonempty / nonempty_preds if nonempty_preds else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return round(recall, 3), round(precision, 3), round(f1, 3)


def _count(pred, labels, files, col, norm):
    correct = correct_nonempty = nonempty = 0
    for f in files:
        p, y = norm(pred.at[f, col]), norm(labels.at[f, col])
        if p != "":
            nonempty += 1
        if p == y:
            correct += 1
            if p != "":
                correct_nonempty += 1
    return correct, correct_nonempty, nonempty


def per_field_recall(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """pred: index = 'File Name', columns = CSV column names.
    Returns recall, precision and F1 per field (strict and lenient) plus an overall row."""
    files = [f for f in pred.index if f in labels.index]
    n = len(files)
    rows = []
    totals = {"strict": [0, 0, 0], "lenient": [0, 0, 0]}
    for col in CSV_COLUMNS.values():
        row = {"Field": col}
        for mode, norm in (("strict", _norm_strict), ("lenient", _norm_lenient)):
            c, cn, ne = _count(pred, labels, files, col, norm)
            r, p, f1 = _scores(c, cn, ne, n)
            row.update({f"Correct ({mode})": c, f"Recall ({mode})": r,
                        f"Precision ({mode})": p, f"F1 ({mode})": f1})
            for i, v in enumerate((c, cn, ne)):
                totals[mode][i] += v
        row["Docs"] = n
        rows.append(row)

    overall = {"Field": "ALL FIELDS (micro)"}
    for mode in ("strict", "lenient"):
        c, cn, ne = totals[mode]
        r, p, f1 = _scores(c, cn, ne, n * len(CSV_COLUMNS))
        overall.update({f"Correct ({mode})": c, f"Recall ({mode})": r,
                        f"Precision ({mode})": p, f"F1 ({mode})": f1})
    overall["Docs"] = n
    rows.append(overall)

    cols = ["Field", "Correct (strict)", "Recall (strict)", "Precision (strict)", "F1 (strict)",
            "Correct (lenient)", "Recall (lenient)", "Precision (lenient)", "F1 (lenient)", "Docs"]
    return pd.DataFrame(rows)[cols]


def mismatches(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """List every wrong field so the prompt can be improved."""
    out = []
    for f in pred.index:
        if f not in labels.index:
            continue
        for col in CSV_COLUMNS.values():
            if _norm_lenient(pred.at[f, col]) != _norm_lenient(labels.at[f, col]):
                out.append({"File": f, "Field": col,
                            "Expected": _norm_strict(labels.at[f, col]),
                            "Predicted": _norm_strict(pred.at[f, col])})
    return pd.DataFrame(out)