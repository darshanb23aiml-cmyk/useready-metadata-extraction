"""LLM-based extraction (Gemini by default, Claude optional via LLM_PROVIDER=claude). No regex / static rules: the model reads the document
and fills a typed schema. Labelling conventions are given to it in the prompt."""
import base64
import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .loader import load_document
from .schema import AgreementMetadata

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()   # "gemini" or "claude"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")   # comma-separated list = fallback order
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You extract metadata from Indian/Philippine-style rental agreements.
The document may be a clean text or scanned page images, and the template varies.

Return the six fields by calling the tool. Follow these labelling conventions exactly,
they are how the reference answers were produced:

1. agreement_value: the MONTHLY RENT as digits only (e.g. 12000). Not the security
   deposit or advance, not the annual total, no commas or currency symbols.
2. agreement_start_date: the date the tenancy/lease starts, as DD.MM.YYYY
   (e.g. 01.04.2008). Prefer the date the lease is stated to commence or be
   effective from, over the date the paper was signed, when they differ.
3. agreement_end_date: the last day of the tenancy, DD.MM.YYYY. If the document
   gives only a duration (e.g. 11 months, 12 months), compute it as
   start date + duration - 1 day, written as day.month.year even if the
   result is a nominal date such as 31.03.2009.
   If the document states an explicit end date, use it.
4. renewal_notice_days: the number of days of notice for renewal/termination/vacating
   (e.g. 30, 60, 90). If given in months, convert to days (1 month = 30 days).
   If the document states no notice period, return null. Never guess.
5. party_one: the OWNER / landlord / lessor. party_two: the TENANT / lessee.
   Give the person's or company's name only: drop titles (Mr, Mrs, Sri, Smt, Dr,
   M/s), drop 'S/o', ages, occupations and addresses. Keep the spelling, initials
   and capitalisation as written in the document. If there are several owners or
   tenants, keep them together in the way the document joins them
   (e.g. 'A and B', 'A & B').
6. Any field you cannot find must be null.

Read the whole document before answering, including later clauses; the rent,
term and notice period are often in numbered clauses further down."""


# ---------------------------------------------------------------- Gemini ----
_gemini = None


def _gemini_client():
    global _gemini
    if _gemini is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key or key.startswith("paste"):
            raise RuntimeError("GEMINI_API_KEY missing. Put it in the .env file.")
        _gemini = genai.Client(api_key=key)
    return _gemini


def _gemini_parts(doc: dict) -> list:
    from google.genai import types
    if doc["kind"] == "text":
        return [f"Here is the agreement text:\n\n{doc['text']}\n\nExtract the metadata."]
    parts = [f"The agreement is a scan split into {len(doc['images'])} consecutive image "
             "pieces (top to bottom, slightly overlapping). Read all of them."]
    for img in doc["images"]:
        parts.append(types.Part.from_bytes(data=base64.b64decode(img), mime_type="image/png"))
    parts.append("Extract the metadata.")
    return parts


GEMINI_RPM = float(os.getenv("GEMINI_RPM", "4"))   # free tier allows 5/min; stay below it
_last_call = 0.0


def _throttle():
    """Space out API calls so the per-minute limit is never hit (rejected calls waste daily quota)."""
    global _last_call
    wait = 60.0 / GEMINI_RPM - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


class QuotaExhausted(RuntimeError):
    """Daily free-tier quota used up on every configured model."""


class ModelUnavailable(QuotaExhausted):
    """The model did not answer after retries (rate limit, overload, ...). Carries the real error text."""


LAST_MODEL_USED = None


def _extract_gemini(doc: dict) -> AgreementMetadata:
    global LAST_MODEL_USED
    from google.genai import types
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        response_mime_type="application/json",
        response_schema=AgreementMetadata,
    )
    models = [m.strip() for m in GEMINI_MODEL.split(",") if m.strip()]
    last_err = ""
    for model in models:
        if model in _exhausted:
            continue
        for attempt in range(4):
            try:
                _throttle()
                resp = _gemini_client().models.generate_content(
                    model=model, contents=_gemini_parts(doc), config=config)
                LAST_MODEL_USED = model
                if resp.parsed is not None:
                    return AgreementMetadata.model_validate(resp.parsed)
                return AgreementMetadata.model_validate_json(resp.text)
            except Exception as e:                # noqa: BLE001
                msg = str(e)
                low = msg.lower()
                last_err = msg
                if "429" in msg or "resource_exhausted" in low or "quota" in low:
                    if "perday" in low.replace(" ", "").replace("_", ""):
                        print(f"  [daily quota used up for model {model}]")
                        _exhausted.add(model)
                        break                      # try next model in the list
                    time.sleep(20 * (attempt + 1)) # per-minute limit: wait and retry
                    continue
                if "503" in msg or "unavailable" in low or "overloaded" in low:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
    if not all(m in _exhausted for m in models):
        raise ModelUnavailable(
            "The model did not answer after 4 attempts (temporary limit or server busy). "
            "Last error from Google: " + last_err[:600])
    raise QuotaExhausted(
        "Daily free quota is used up for: " + ", ".join(models) +
        ". Finished documents are cached; run the same command later (or add another "
        "model name to GEMINI_MODEL in .env) and it will continue where it stopped.")


_exhausted = set()


# ---------------------------------------------------------------- Claude ----
_claude = None


def _extract_claude(doc: dict) -> AgreementMetadata:
    global _claude
    if _claude is None:
        from anthropic import Anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key or key.startswith("paste"):
            raise RuntimeError("ANTHROPIC_API_KEY missing. Put it in the .env file.")
        _claude = Anthropic(api_key=key)
    if doc["kind"] == "text":
        content = [{"type": "text", "text": f"Here is the agreement text:\n\n{doc['text']}\n\nExtract the metadata."}]
    else:
        content = [{"type": "text", "text": f"The agreement is a scan split into {len(doc['images'])} consecutive image pieces. Read all of them."}]
        content += [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": i}} for i in doc["images"]]
        content.append({"type": "text", "text": "Extract the metadata."})
    tool = {"name": "record_metadata", "description": "Record the extracted agreement metadata.",
            "input_schema": AgreementMetadata.model_json_schema()}
    resp = _claude.messages.create(
        model=CLAUDE_MODEL, max_tokens=1000, temperature=0, system=SYSTEM_PROMPT,
        tools=[tool], tool_choice={"type": "tool", "name": "record_metadata"},
        messages=[{"role": "user", "content": content}])
    for block in resp.content:
        if block.type == "tool_use":
            return AgreementMetadata(**block.input)
    return AgreementMetadata()


# ----------------------------------------------------------------- public ---
def prompt_hash() -> str:
    """Changes whenever the prompt changes, so cached results are never stale."""
    return hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()[:8]


def extract_from_doc(doc: dict) -> AgreementMetadata:
    if PROVIDER == "claude":
        return _extract_claude(doc)
    return _extract_gemini(doc)


def extract(path) -> AgreementMetadata:
    """Extract metadata from a .docx or .png file path."""
    return extract_from_doc(load_document(path))
