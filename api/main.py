"""RESTful web service.

Start:   uvicorn api.main:app --reload
Docs:    http://127.0.0.1:8000/docs
Use:     curl -X POST -F "file=@data/test/24158401-Rental-Agreement.png" http://127.0.0.1:8000/extract

Lookup order for every uploaded file (so the service keeps answering even when
the free LLM quota is used up):
  1. API cache      - same file content was already processed through this API
  2. Batch cache    - same content as a file in data/train or data/test that was
                      already processed by `python -m src.run`
  3. Live LLM call  - needs GEMINI_API_KEY (or ANTHROPIC_API_KEY) in .env
"""
import hashlib
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src import extractor
from src.extractor import QuotaExhausted
from src.loader import doc_id

ROOT = Path(__file__).resolve().parent.parent
API_CACHE = ROOT / "outputs" / "api_cache"
BATCH_CACHE = ROOT / "outputs" / "cache"
ALLOWED = {".docx", ".png", ".jpg", ".jpeg"}

app = FastAPI(title="Rental Agreement Metadata Extraction", version="1.0")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _known_files() -> dict:
    """content-hash -> doc id for the provided dataset files."""
    known = {}
    for split in ("train", "test"):
        folder = ROOT / "data" / split
        if folder.exists():
            for f in folder.iterdir():
                if f.suffix.lower() in ALLOWED:
                    known[_sha(f.read_bytes())] = doc_id(f)
    return known


def _read(path: Path):
    return json.loads(path.read_text())["fields"] if path.exists() else None


@app.get("/health")
def health():
    return {"status": "ok", "provider": extractor.PROVIDER}


@app.post("/extract")
def extract_metadata(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Upload .docx or .png")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")

    digest, ph = _sha(data), extractor.prompt_hash()
    API_CACHE.mkdir(parents=True, exist_ok=True)

    # 1) API cache   2) batch-run cache for the provided dataset files
    fields = _read(API_CACHE / f"{digest}_{ph}.json")
    source = "api-cache"
    if fields is None:
        known = _known_files().get(digest)
        if known:
            fields = _read(BATCH_CACHE / f"{known}_{ph}.json")
            source = "batch-cache"

    # 3) live model call
    if fields is None:
        source = "model"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(data)
            try:
                fields = extractor.extract(path).model_dump()
            except QuotaExhausted as e:
                raise HTTPException(503, f"Model quota exhausted, try again later. {e}",
                                    headers={"Retry-After": "3600"})
            except RuntimeError as e:             # e.g. API key missing
                raise HTTPException(503, f"Model unavailable: {e}")
            except ValueError as e:
                raise HTTPException(422, str(e))
        (API_CACHE / f"{digest}_{ph}.json").write_text(json.dumps({"fields": fields}, indent=2))

    return {"file_name": file.filename, "source": source, **fields}