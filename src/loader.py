"""Load a .docx or .png into something the LLM can read.

DOCX -> plain text (paragraphs and tables, in document order)
PNG  -> list of base64 image slices (tall scans are cut into overlapping
        pieces so the text stays readable after the API downsizes images)
"""
import base64
import io
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

SLICE_ASPECT = 1.4      # slice height = width * 1.4
OVERLAP = 0.06          # 6% overlap so lines on the cut are not lost
MAX_WIDTH = 1600


def doc_id(path) -> str:
    """File name without extension; also drops the stray '.pdf' in 'x.pdf.docx'."""
    stem = Path(path).stem
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    return stem


def _docx_text(path) -> str:
    doc = Document(str(path))
    lines = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            text = Paragraph(child, doc).text.strip()
            if text:
                lines.append(text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                cells = []
                for c in row.cells:
                    t = c.text.strip()
                    if t and (not cells or cells[-1] != t):   # skip merged-cell repeats
                        cells.append(t)
                if cells:
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def _image_slices(path) -> list:
    img = Image.open(path).convert("RGB")
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
    w, h = img.size
    step_h = int(w * SLICE_ASPECT)
    overlap = int(step_h * OVERLAP)
    slices, top = [], 0
    while top < h:
        bottom = min(top + step_h, h)
        piece = img.crop((0, top, w, bottom))
        buf = io.BytesIO()
        piece.save(buf, format="PNG")
        slices.append(base64.standard_b64encode(buf.getvalue()).decode())
        if bottom >= h:
            break
        top = bottom - overlap
    return slices


def load_document(path) -> dict:
    """Return {'id', 'kind': 'text'|'image', 'text' or 'images'}."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".docx":
        return {"id": doc_id(path), "kind": "text", "text": _docx_text(path)}
    if ext in (".png", ".jpg", ".jpeg"):
        return {"id": doc_id(path), "kind": "image", "images": _image_slices(path)}
    raise ValueError(f"Unsupported file type: {ext} (use .docx or .png)")
