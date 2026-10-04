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

Requires Python 3.10 or newer (developed and tested on Python 3.13).

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

`outputs/cache/` holds the raw model output for each processed document (one JSON file per document), saved during the real model run described under Results. Re-running reads those files instead of calling the API again, so this works with **no API key**. Delete that folder to force fresh model calls (this needs a key). A document that is not in the cache needs a key. Free-tier Gemini keys are limited (about 5 requests/minute, 20/day); the code spaces its calls to stay under the per-minute limit and stops cleanly if the daily quota is used up, keeping everything already processed.

### Run the notebook (optional)

The same steps are available as a notebook that imports the code from `src/`:

```bash
jupyter lab
```

Open `notebooks/run_pipeline.ipynb`, select the Python environment where the requirements are installed as the kernel, and choose **Run > Run All Cells** (in VS Code: **Run All**). It works when started from the project root or from the `notebooks/` folder, and it uses the same saved results, so it needs no API key for the provided files. The last cell calls the REST API and only works while the server from the next section is running; otherwise it prints a short message.

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

**Validation on labelled training files** - *partial: 6 of 9 labelled documents processed* (the free API quota ran out; see Limitations). Scores use the labels exactly as provided. Files: `outputs/predictions_train.csv`, `outputs/recall_train.csv`.

| Field | Recall | Precision | F1 |
|---|---|---|---|
| Agreement Value | 0.67 | 0.80 | 0.73 |
| Agreement Start Date | 0.67 | 0.80 | 0.73 |
| Agreement End Date | 0.17 | 0.20 | 0.18 |
| Renewal Notice (Days) | 0.83 | 0.80 | 0.82 |
| Party One | 0.50 | 0.50 | 0.50 |
| Party Two | 0.83 | 0.83 | 0.83 |
| **Overall (all fields)** | **0.61** | **0.66** | **0.63** |

Precision is higher than recall for Value and Start Date because one document (`44737744`) has an unreadable text layer and the system returned empty values instead of guessing. For Renewal Notice, precision is lower than recall because one reference is empty and the system correctly left it empty, which counts for recall but not for precision. With so few documents a single wrong field moves a score by 17 to 33 points, so these numbers are indicative only. Most of the misses come from the reference labels, not the model (see below).

### Error analysis

Test set:
- `95980236` (End Date): the document says "11 (Eleven) months commencing from 1 April 2010". The system returned `28.02.2011`; the reference is `31.03.2011`.
- `228094620` (Party Two): the reference is `.B.Kishore` (a leftover dot from removing "Mr" in "Mr.B.Kishore"); the system returned `B.Kishore`.
- `156155545` (Party Two): the system returned `SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd.`; the reference drops "SRI" and the final period.

Training files:
- **Two labels appear to be swapped (`54770958` and `54945838`).** The scan of `54770958` states rent Rs. 5500, an agreement executed on 20 April 2011, 11 months, two months' notice, lessors Asha Ramesh & Ramesh K.N. and lessees Sadasivuni Deepti & Sadasivuni Kiran, which is what the system returned. The reference values for this file (8000, 01.04.2011 to 31.03.2012, 90 days, K. Parthasarathy / Veerabrahmam Bathini) match the text of the other scan, `54945838` (rent Rs. 8,000, 1 April 2011, twelve months, three months' notice, Prof. K. Parthasarathy and Veerabrahmam Bathini). Six of the misses come from this file. The labels were not altered; the scores above use them as provided. `54945838` has not been processed yet.
- Several reference end dates are not real calendar dates (`31.11.2009` and `31.02.2011`; `31.04.2011`), while the system returned real dates for the stated term (`18325926`, `47854715`, `36199312`).
- `18325926` (Party One): the reference keeps the title (`MR.K.Kuttan`); other references drop titles, so the labels are not consistent.
- `44737744`: this `.docx` is the output of OCR on a scan and its text is unreadable (the rent reads `9.99.7.9°`, the date `.?.??&`; the only embedded images are a stamp/signature). The system returns empty values rather than guessing, so Value, Start Date, End Date and the garbled Party One name do not match.
- `50070534` was extracted with all six fields matching.

## Assumptions and limitations

- **Labels are not fully consistent.** Some reference end dates are nominal (`31.11.2009`, `31.04.2011`, `31.02.2011`), titles are sometimes kept and sometimes removed, and the labels of two training files (`54770958`, `54945838`) appear to be swapped. Exact matching against such labels cannot reach 100%, and I did not add hand-written rules to imitate them.
- **No tuning to the test labels.** The prompt was written from the conventions visible in the training data before the test run; it was not adjusted after seeing test results.
- **Unreadable inputs return empty values** ("never guess" is part of the prompt), which counts as a miss in recall.
- **No dataset preprocessing** was needed because no model is trained: documents go to a pretrained LLM, and scoring only trims whitespace.
- **Free API quota.** The free Gemini tier is limited (about 5 requests/minute and a small daily allowance). The validation run on the training files stopped after 6 of the 9 labelled documents; `54945838`, `6683127` and `6683129` are not processed yet. Re-running `python -m src.run --split train` after the quota resets processes them; finished documents are reused from `outputs/cache/`.
- **New documents need a working LLM key.** Documents already processed are served from the saved results; anything else needs `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY` with `LLM_PROVIDER=claude`) in `.env`. Without one, the API returns `503` with an explanatory message.
- `24158401-Rental-Agreement` is listed in `train.csv` but its document is in `test/`; it is scored once. `46239065-Standard-Rental-Agreement...docx` in `train/` has no label and is skipped.
