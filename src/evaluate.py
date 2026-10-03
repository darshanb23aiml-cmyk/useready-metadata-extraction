"""Per-field recall: exact matches / total documents."""
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


def per_field_recall(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """pred: index = 'File Name', columns = CSV column names.
    Returns a table with strict and lenient (case/space-insensitive) recall."""
    common = [f for f in pred.index if f in labels.index]
    rows = []
    for col in CSV_COLUMNS.values():
        strict = sum(_norm_strict(pred.at[f, col]) == _norm_strict(labels.at[f, col]) for f in common)
        lenient = sum(_norm_lenient(pred.at[f, col]) == _norm_lenient(labels.at[f, col]) for f in common)
        n = len(common)
        rows.append({"Field": col, "Correct (strict)": strict, "Recall (strict)": round(strict / n, 3),
                     "Correct (lenient)": lenient, "Recall (lenient)": round(lenient / n, 3), "Docs": n})
    return pd.DataFrame(rows)


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
