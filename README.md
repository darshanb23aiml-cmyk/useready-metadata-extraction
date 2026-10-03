# Metadata Extraction from Rental Agreements

Extracts six fields from a rental agreement, whether it is a **`.docx`** file or a **scanned `.png`** image, regardless of template:

| Field | Meaning |
|---|---|
| Agreement Value | Monthly rent |
| Agreement Start Date | `DD.MM.YYYY` |
| Agreement End Date | `DD.MM.YYYY` |
| Renewal Notice (Days) | Notice period in days (empty if the document has none) |
| Party One | Owner / landlord |
| Party Two | Tenant |

No regex, keyword lists or hand-written rules are used. A pretrained LLM reads the document and fills a typed schema.

## Solution approach

1. **Loading (`src/loader.py`)**
   - `.docx`: paragraphs and tables are read in document order with `python-docx`.
   - `.png`: long scans are cut into a few overlapping slices so that the text stays readable when sent to the model. The slices are sent as images; no separate OCR step is needed.
2. **Extraction (`src/extractor.py`)** - one LLM call per document (Google Gemini by default, Anthropic Claude optional via `LLM_PROVIDER=claude`), temperature 0, with a JSON schema for the output (`src/schema.py`). The prompt describes the labelling conventions found in the training data (monthly rent rather than deposit, `DD.MM.YYYY` dates, end date = start + term - 1 day, empty renewal notice when absent, names without titles/addresses). These conventions are given to the model as instructions, not coded as rules.
3. **Why an LLM and not training a model?** Only 10 labelled documents are provided, which is far too few to train or fine-tune a reliable extractor. A pretrained model with a precise prompt generalises across templates.
4. **Evaluation (`src/evaluate.py`)** - per-field **recall** = exact matches / documents (the requested metric). Precision and F1 are reported as well. Strict (exact string) and lenient (ignores letter case and extra spaces) versions are saved in `outputs/recall_*.csv`.
5. **Saved results** - every processed document is stored in `outputs/cache/`, so the results can be reproduced **without an API key or quota** (see below).

## Project structure

```
src/        schema.py, loader.py, extractor.py, evaluate.py, run.py
api/        main.py  (FastAPI REST service)
notebooks/  run_pipeline.ipynb
data/       train/, test/, train.csv, test.csv
outputs/    predictions_test.csv, recall_*.csv, cache/
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Place the provided `data/` folder (train/, test/, train.csv, test.csv) in the project root. It is needed for the batch runs and for the API's lookup of saved results.

Copy `.env.example` to `.env` and add a key (only needed for documents that are not already in `outputs/cache/`):

```
GEMINI_API_KEY=your-key        # free key: https://aistudio.google.com
GEMINI_MODEL=gemini-3.8-flash
```

## Reproduce the predictions

```bash
python -m src.run --split test     # writes outputs/predictions_test.csv and outputs/recall_test.csv
python -m src.run --split train    # validation on the labelled training files
```

Results for the provided files are read from `outputs/cache/` (produced by the model run described below), so this works with **no API key**. A document that is not in the cache needs a key. Free-tier Gemini keys are limited (about 5 requests/minute, 20/day); the code spaces its calls to stay under the per-minute limit and stops cleanly if the daily quota is used up, keeping everything already processed.

The same steps are available as a notebook: `notebooks/run_pipeline.ipynb`.

## REST API

Start the service:

```bash
uvicorn api.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

```bash
curl -X POST -F "file=@data/test/24158401-Rental-Agreement.png" http://127.0.0.1:8000/extract
```

(on Windows PowerShell use `curl.exe`)

Example response:

```json
{
  "file_name": "24158401-Rental-Agreement.png",
  "source": "batch-cache",
  "agreement_value": "12000",
  "agreement_start_date": "01.04.2008",
  "agreement_end_date": "31.03.2009",
  "renewal_notice_days": "60",
  "party_one": "Hanumaiah",
  "party_two": "Vishal Bhardwaj"
}
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status |
| `POST /extract` | Upload a `.docx` or `.png`; returns the six fields |

`source` shows where the answer came from: `model` (live LLM call), `api-cache` or `batch-cache` (saved earlier result for identical file content). Errors: `415` unsupported file type, `400` empty file, `503` model unavailable or quota used up (retry later).

## Results

Model: `gemini-3.8-flash` (Google Gemini API), run on 3 Oct 2026.

**Metrics.** *Recall* = exact matches / documents (the metric requested in the assignment; a correctly empty field counts as a match). *Precision* = correct non-empty answers / non-empty answers given (an empty answer means "no answer", so it lowers recall but not precision). *F1* = harmonic mean of precision and recall. The tables below use exact string match; case/space-insensitive scores are in the CSV files.

**Test set (4 documents)** - predictions in `outputs/predictions_test.csv`, scores in `outputs/recall_test.csv`

| Field | Recall | Precision | F1 |
|---|---|---|---|
| Agreement Value | 1.00 | 1.00 | 1.00 |
| Agreement Start Date | 1.00 | 1.00 | 1.00 |
| Agreement End Date | 0.75 | 0.75 | 0.75 |
| Renewal Notice (Days) | 1.00 | 1.00 | 1.00 |
| Party One | 1.00 | 1.00 | 1.00 |
| Party Two | 0.50 | 0.50 | 0.50 |
| **Overall (all fields)** | **0.875** | **0.875** | **0.875** |

21 of 24 fields are correct. Precision equals recall here because the system gave an answer for every field.

**Validation on labelled training files** - *partial: 3 of 9 documents processed so far* (the free API quota ran out; see Limitations). Scores in `outputs/recall_train.csv`.

| Field | Recall | Precision | F1 |
|---|---|---|---|
| Agreement Value | 0.67 | 1.00 | 0.80 |
| Agreement Start Date | 0.67 | 1.00 | 0.80 |
| Agreement End Date | 0.00 | 0.00 | 0.00 |
| Renewal Notice (Days) | 1.00 | 1.00 | 1.00 |
| Party One | 0.33 | 0.33 | 0.33 |
| Party Two | 1.00 | 1.00 | 1.00 |
| **Overall (all fields)** | **0.61** | **0.71** | **0.66** |

Precision is higher than recall for Value and Start Date because one document (`44737744`) has an unreadable text layer, and the system returned empty values instead of guessing. With so few documents a single wrong field moves a score by 25 to 33 points, so these numbers are indicative only.

### Error analysis

Test set:
- `95980236` (End Date): the document says "11 (Eleven) months commencing from 1 April 2010". The system returned `28.02.2011`; the reference is `31.03.2011`.
- `228094620` (Party Two): the reference is `.B.Kishore` (a leftover dot from removing "Mr" in "Mr.B.Kishore"); the system returned `B.Kishore`.
- `156155545` (Party Two): the system returned `SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd.`; the reference drops "SRI" and the final period.

Training files processed so far:
- `18325926`, `36199312` (End Date): the references are not real calendar dates (`31.11.2009`, `31.04.2011`); the system returned real dates for the stated term.
- `18325926` (Party One): the reference keeps the title (`MR.K.Kuttan`); other references drop titles, so the labels are not consistent.
- `44737744`: this `.docx` is the output of OCR on a scan and its text is unreadable (the rent reads `9.99.7.9°`, the date `.?.??&`; the only embedded images are a stamp/signature). The system returns empty values rather than guessing, so Value, Start Date, End Date and the garbled Party One name do not match.

## Assumptions and limitations

- **Labels are not fully consistent.** Some reference end dates are nominal (`31.11.2009`, `31.04.2011`, `31.02.2011`), and titles are sometimes kept and sometimes removed. Exact matching against such labels cannot reach 100%, and I did not add hand-written rules to imitate them.
- **No tuning to the test labels.** The prompt was written from the conventions visible in the training data before the test run; it was not adjusted after seeing test results.
- **Unreadable inputs return empty values** ("never guess" is part of the prompt), which counts as a miss in recall.
- **No dataset preprocessing** was needed because no model is trained: documents go to a pretrained LLM, and scoring only trims whitespace.
- **Free API quota.** The free Gemini tier is limited (about 5 requests/minute and a small daily allowance). The validation run on the training files stopped after 3 documents. Re-running `python -m src.run --split train` after the quota resets processes the remaining files; finished documents are reused from `outputs/cache/`.
- **New documents need a working LLM key.** Documents already processed are served from the saved results; anything else needs `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY` with `LLM_PROVIDER=claude`) in `.env`. Without one, the API returns `503` with an explanatory message.
- `24158401-Rental-Agreement` is listed in `train.csv` but its document is in `test/`; it is scored once. `46239065-Standard-Rental-Agreement...docx` in `train/` has no label and is skipped.
