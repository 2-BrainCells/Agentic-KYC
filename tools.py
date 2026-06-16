"""
tools.py — External tool calls used by the agents.

Three groups: LLM tools (vLLM text + JSON), document tools (Tesseract OCR +
field parsing), and sanctions tools (Qdrant vector search with deterministic
scoring, plus local-scan and LLM-as-judge fallbacks). Every function is
wrapped in try/except — a tool failure returns a safe default and logs it, it
never crashes the pipeline.

Setup once: pip install openai pytesseract pillow qdrant-client[fastembed]
            apt-get install -y tesseract-ocr tesseract-ocr-hin
"""

import atexit
import json
import re
import os
import pytesseract

from PIL                    import Image
from openai                 import OpenAI
from qdrant_client          import QdrantClient

from config import (
    VLLM_API_BASE, VLLM_API_KEY, MODEL_NAME,
    VLLM_TIMEOUT, VLLM_MAX_TOKENS,
    TEMP_FIRST_PASS, TEMP_REFINE_PASS,
    TEMP_AGENT_LOGIC, TEMP_EXPLANATION,
    QDRANT_LOCAL_PATH, QDRANT_COLLECTION_NAME, QDRANT_TOP_K,
    FUZZY_CLEAR_BELOW, FUZZY_AMBIGUOUS_HIGH,
    SANCTIONS_LIST_PATH,
)


# ── Clients (created once, reused) ──────────────────────────────────────────

# vLLM — OpenAI-compatible endpoint running Llama-3 on the MI300X.
llm_client = OpenAI(
    base_url = VLLM_API_BASE,
    api_key  = VLLM_API_KEY,
)


def _make_qdrant_client():
    """
    Open the embedded Qdrant store without ever crashing the app.

    Local mode takes an exclusive lock on QDRANT_LOCAL_PATH. If another process
    holds it, fall back to an in-memory client — sanctions screening still works
    because query_sanctions_db() degrades to the on-disk fuzzy scan. Only
    cross-session persistence of the exception cache is lost. Returns None only
    if even the in-memory client fails.
    """
    try:
        return QdrantClient(path=QDRANT_LOCAL_PATH)
    except Exception as e:
        print(f"[tools] ⚠ Embedded Qdrant at '{QDRANT_LOCAL_PATH}' is locked by "
              f"another process ({e}).\n"
              f"        Falling back to IN-MEMORY mode — sanctions screening "
              f"still works via the local fuzzy scan.\n"
              f"        To restore vector search: stop any other app.py / "
              f"notebook kernel holding that folder, then restart.")
        try:
            return QdrantClient(location=":memory:")
        except Exception as e2:
            print(f"[tools] ⚠ In-memory Qdrant also failed: {e2} — "
                  f"screening will use the on-disk fuzzy scan only.")
            return None


qdrant_client = _make_qdrant_client()

# Close the embedded client at exit to avoid a noisy shutdown traceback.
if qdrant_client is not None:
    atexit.register(qdrant_client.close)


# ── Group 1: LLM tools ──────────────────────────────────────────────────────

def call_text_llm(
    prompt:        str,
    temperature:   float = TEMP_AGENT_LOGIC,
    system_prompt: str   = "You are a precise KYC compliance assistant.",
) -> str:
    """
    Send a prompt to vLLM and return the model's text response.

    Returns "[LLM unavailable]" if the server is unreachable (demo-safe).
    """
    try:
        response = llm_client.chat.completions.create(
            model       = MODEL_NAME,
            temperature = temperature,
            max_tokens  = VLLM_MAX_TOKENS,
            timeout     = VLLM_TIMEOUT,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[tools] call_text_llm failed: {e}")
        return "[LLM unavailable]"


def call_llm_for_json(
    prompt:        str,
    temperature:   float = TEMP_AGENT_LOGIC,
    system_prompt: str   = "You are a precise KYC compliance assistant. "
                           "Always respond with valid JSON only. "
                           "No extra text, no markdown, no explanation outside the JSON.",
) -> dict:
    """
    Send a prompt to vLLM and parse the response as JSON.

    Strips ```json fences and any text before the first { or [. Returns {} on
    any parse failure.
    """
    raw = call_text_llm(prompt, temperature, system_prompt)

    # Strip markdown code fences, then any leading prose before the JSON.
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()
    json_start = min(
        raw.find("{") if "{" in raw else len(raw),
        raw.find("[") if "[" in raw else len(raw),
    )
    if json_start > 0:
        raw = raw[json_start:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[tools] JSON parse failed: {e}\nRaw response: {raw[:200]}")
        return {}


# ── Group 2: Document tools (Aadhaar OCR + field parsing) ───────────────────

def extract_text_from_image(image_path: str) -> str:
    """
    OCR an Aadhaar card image with Tesseract (English + Devanagari).

    Returns "" on failure. Requires tesseract-ocr and tesseract-ocr-hin.
    """
    try:
        img  = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang="eng+hin")
        return text.strip()

    except Exception as e:
        print(f"[tools] OCR failed for {image_path}: {e}")
        return ""


def parse_aadhaar_fields(
    ocr_text:    str,
    directive:   str   = "standard_first_pass",
    temperature: float = TEMP_FIRST_PASS,
) -> dict:
    """
    Ask Llama-3 to turn raw Aadhaar OCR text into structured identity fields.

    directive "standard_first_pass" extracts all fields; "refinement_pass"
    focuses on the father's name (QR 'care_of'), middle name, and place of
    birth. Returns a safe all-null structure if the LLM returns nothing.
    """
    if directive == "refinement_pass":
        focus = (
            "This is a REFINEMENT pass. The first extraction missed the father's "
            "name. Focus specifically on:\n"
            "1. The 'Care Of' (C/O or S/O or W/O or D/O) field — this contains "
            "the father's or husband's name. It often appears just above the address.\n"
            "2. Any Devanagari text that looks like a personal name.\n"
            "3. Place of birth if visible.\n"
            "Set 'extraction_pass' to 1."
        )
    else:
        focus = (
            "This is a STANDARD first pass. Extract all identity fields you can find.\n"
            "The text may include the Aadhaar BACK side: read the full ADDRESS and, "
            "if a 'Care Of' line is printed (C/O, S/O, W/O, D/O), capture the "
            "father's/husband's name into 'father_name'. If no such line is printed, "
            "leave 'father_name' null — do not guess.\n"
            "DOB on an Aadhaar is printed as DD/MM/YYYY — convert it to YYYY-MM-DD.\n"
            "Set 'extraction_pass' to 0."
        )

    prompt = f"""
You are reading text extracted from an Indian Aadhaar card via OCR.
The text may contain English and Hindi (Devanagari) characters mixed together.
OCR errors are common — use context to correct obvious mistakes.

{focus}

RAW OCR TEXT:
--------------
{ocr_text}
--------------

Return ONLY a JSON object with these exact keys.
Use null for any field you cannot find.
Do not add any text outside the JSON.

{{
  "full_name_english":    "NAME IN CAPS as printed on card",
  "full_name_devanagari": "नाम in Devanagari script, or null",
  "given_name":           "first name only",
  "surname":              "last name / family name",
  "father_name":          "father or husband name from Care Of field, or null",
  "care_of_type":         "S/O or W/O or D/O or C/O, or null",
  "dob":                  "YYYY-MM-DD format, or null",
  "gender":               "M or F or T, or null",
  "aadhaar_last4":        "last 4 digits of Aadhaar number, or null",
  "aadhaar_masked":       "XXXX XXXX 1234 format, or null",
  "address":              "full address as one string",
  "pin_code":             "6-digit PIN code, or null",
  "place_of_birth":       "city and state if visible, or null",
  "pan_number":           "PAN format AAAAA9999A if visible on PAN card, or null",
  "pan_name":             "name on PAN card if different from Aadhaar, or null",
  "qr_verified":          false,
  "uidai_status":         "ACTIVE",
  "extraction_pass":      0
}}
"""

    result = call_llm_for_json(prompt, temperature=temperature)

    if not result:
        return {
            "full_name_english": None, "full_name_devanagari": None,
            "given_name": None, "surname": None, "father_name": None,
            "dob": None, "gender": None, "aadhaar_last4": None,
            "address": None, "pin_code": None, "pan_number": None,
            "qr_verified": False, "uidai_status": "UNKNOWN",
            "extraction_pass": 0 if directive == "standard_first_pass" else 1,
        }

    return result


def compute_field_confidence(extracted: dict) -> dict:
    """
    Heuristic confidence (0.0–1.0) per extracted field: empty = 0.0, numeric/
    format-checkable fields = 0.97, text fields = 0.87. Drives the UI overlay
    colour (>= 0.75 green, otherwise amber/red).
    """
    confidence = {}
    high_confidence_fields = {
        "dob", "aadhaar_last4", "aadhaar_masked", "pin_code", "gender"
    }
    for field, value in extracted.items():
        if value is None or value == "":
            confidence[field] = 0.0
        elif field in high_confidence_fields:
            confidence[field] = 0.97
        else:
            confidence[field] = 0.87

    return confidence


def parse_salary_slip(image_path: str) -> dict:
    """
    OCR a salary slip and extract the pay figure so the financial agent can
    verify declared income against a document.

    Returns {monthly_income, annual_income, employer, raw_found}. Safe on every
    failure path (missing file, empty OCR, LLM down, bad number) — raw_found is
    then False and the financial agent simply skips verification.
    """
    result = {"monthly_income": None, "annual_income": None,
              "employer": None, "raw_found": False}
    try:
        if not image_path or not os.path.exists(image_path):
            return result

        text = extract_text_from_image(image_path)
        if not text.strip():
            return result

        parsed = call_llm_for_json(
            prompt=f"""You are reading an Indian salary slip / payslip extracted via OCR.
Indian payslips state a MONTHLY figure ('Net Pay', 'Net Salary', 'Gross Salary').
Numbers may use Indian formatting (e.g. 1,00,000). Return the plain number only.

RAW OCR TEXT:
--------------
{text}
--------------

Return ONLY this JSON (use null if a value is not present):
{{
  "monthly_net_income":   number or null,
  "monthly_gross_income": number or null,
  "employer_name":        "string or null"
}}""",
            temperature=TEMP_FIRST_PASS,
        )

        monthly = parsed.get("monthly_gross_income") or parsed.get("monthly_net_income")
        if monthly is not None:
            try:
                monthly = float(str(monthly).replace(",", "").strip())
                if monthly > 0:
                    result["monthly_income"] = monthly
                    result["annual_income"]  = round(monthly * 12, 2)
                    result["raw_found"]       = True
            except (TypeError, ValueError):
                pass
        result["employer"] = parsed.get("employer_name")
        return result

    except Exception as e:
        print(f"[tools] parse_salary_slip failed: {e}")
        return result


# ── Group 3: Sanctions tools ────────────────────────────────────────────────

def _sanctions_match_score(query_name: str, entry_name: str, aliases: list = None) -> float:
    """
    Deterministic 0.0–1.0 name-match score between a customer name and a
    watchlist entry (including aliases).

    A 50/50 blend of token overlap and difflib character similarity. Kept
    deterministic so the config thresholds (0.60/0.85) always mean the same
    thing on any machine. Alias matches are capped at 0.81 so even an exact AKA
    stays in the AMBIGUOUS band — this is what arms the self-correction loop.
    """
    import difflib

    ALIAS_SCORE_CAP = 0.81

    def _blend(a: str, b: str) -> float:
        a, b = a.lower().strip(), b.lower().strip()
        if not a or not b:
            return 0.0
        direct = difflib.SequenceMatcher(None, a, b).ratio()
        ta, tb = set(a.split()), set(b.split())
        overlap = len(ta & tb) / max(len(ta), len(tb)) if ta and tb else 0.0
        return 0.5 * overlap + 0.5 * direct

    primary = _blend(query_name, entry_name) if entry_name else 0.0
    alias_best = max(
        (_blend(query_name, a) for a in (aliases or []) if a),
        default=0.0,
    )
    return round(max(primary, min(alias_best, ALIAS_SCORE_CAP)), 3)


def _entry_to_match(entry: dict, score: float) -> dict:
    """Format a sanctions-list entry as the standard match dict agents expect."""
    return {
        "matched_name": entry.get("name", "Unknown"),
        "match_score":  round(score, 3),
        "list_source":  entry.get("list_source", "Unknown"),
        "aliases":      entry.get("aliases", []),
        "dob_range":    entry.get("dob_range", None),
        "father_name":  entry.get("father_name", None),
        "risk_level":   entry.get("risk_level", None),
        "case_ref":     entry.get("case_ref", None),
    }


def _load_sanctions_list() -> list:
    """Load the mock sanctions JSON from disk. Returns [] if missing/unreadable."""
    try:
        if not os.path.exists(SANCTIONS_LIST_PATH):
            print(f"[tools] Sanctions list not found at {SANCTIONS_LIST_PATH}")
            return []
        with open(SANCTIONS_LIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[tools] Failed to load sanctions list: {e}")
        return []


def local_sanctions_scan(name: str, name_variants: list = None) -> list:
    """
    Fallback: deterministic fuzzy scan of the sanctions JSON on disk. Needs no
    Qdrant and no LLM, so it is always available with the same answer. Returns
    the same format as query_sanctions_db().
    """
    sanctions = _load_sanctions_list()
    if not sanctions:
        return []

    queries = [q for q in [name] + (name_variants or []) if q]
    matches = []
    for entry in sanctions:
        score = max(
            (_sanctions_match_score(q, entry.get("name", ""), entry.get("aliases"))
             for q in queries),
            default=0.0,
        )
        if score >= FUZZY_CLEAR_BELOW:
            matches.append(_entry_to_match(entry, score))

    matches.sort(key=lambda m: m["match_score"], reverse=True)
    return matches[:QDRANT_TOP_K]


def query_sanctions_db(
    name:          str,
    name_variants: list = None,
    top_k:         int  = QDRANT_TOP_K,
) -> list:
    """
    Search the Qdrant sanctions DB for similar names.

    Qdrant retrieves candidates across spelling/script variants; the
    deterministic scorer assigns the calibrated match_score. Falls back to the
    local fuzzy scan, then to LLM-as-judge, if Qdrant is unavailable. Returns []
    on total failure.
    """
    try:
        query_text = name
        if name_variants:
            query_text = " | ".join([name] + name_variants)

        results = qdrant_client.query(
            collection_name = QDRANT_COLLECTION_NAME,
            query_text      = query_text,
            limit           = top_k,
        )

        queries = [q for q in [name] + (name_variants or []) if q]
        matches = []
        for r in results:
            # fastembed query() returns the stored entry in .metadata
            entry = r.metadata or {}
            score = max(
                (_sanctions_match_score(q, entry.get("name", ""), entry.get("aliases"))
                 for q in queries),
                default=0.0,
            )
            if score >= FUZZY_CLEAR_BELOW:
                match = _entry_to_match(entry, score)
                match["retrieval_score"] = round(r.score, 3)   # raw vector similarity
                matches.append(match)

        matches.sort(key=lambda m: m["match_score"], reverse=True)
        return matches

    except Exception as e:
        print(f"[tools] Qdrant query failed: {e} — falling back to local fuzzy scan")
        local = local_sanctions_scan(name, name_variants)
        if local or os.path.exists(SANCTIONS_LIST_PATH):
            return local
        return llm_sanctions_check(name, name_variants)


def llm_sanctions_check(
    name:          str,
    name_variants: list = None,
) -> list:
    """
    Fallback: LLM-as-judge sanctions check. Loads the JSON list and asks Llama-3
    to compare the customer name against each entry. Used only when Qdrant and
    the local scan are both unavailable. Returns the query_sanctions_db() format.
    """
    try:
        if not os.path.exists(SANCTIONS_LIST_PATH):
            print(f"[tools] Sanctions list not found at {SANCTIONS_LIST_PATH}")
            return []

        with open(SANCTIONS_LIST_PATH, "r", encoding="utf-8") as f:
            sanctions = json.load(f)

        sanctions_text = "\n".join([
            f"- {e.get('name')} | DOB: {e.get('dob_range','?')} | "
            f"Case: {e.get('case_ref','?')} | List: {e.get('list_source','?')}"
            for e in sanctions[:50]   # cap to stay within context window
        ])

        all_names = [name] + (name_variants or [])
        names_str = " / ".join(all_names)

        prompt = f"""
You are a KYC compliance officer performing a sanctions screening check.

CUSTOMER NAME(S) TO CHECK: {names_str}

SANCTIONS WATCHLIST (FIU-IND / ED / PMLA / UN):
{sanctions_text}

Task:
Compare the customer name against every entry on the watchlist.
For each entry that could possibly be the same person, return a match.

Scoring guide:
  1.0 = exact match (same name, same DOB range)
  0.85–0.99 = very likely match (same name, minor spelling difference)
  0.70–0.84 = possible match (similar name, needs more identity signals)
  below 0.70 = not a match (ignore these)

Return ONLY a JSON array. If no matches above 0.70, return an empty array [].

Format:
[
  {{
    "matched_name": "exact name from the watchlist",
    "match_score": 0.81,
    "list_source": "exact list source from the watchlist",
    "match_type": "EXACT / FUZZY_GIVEN_SURNAME / ALIAS / PHONETIC",
    "matched_fields": ["which parts matched: given_name, surname, etc."],
    "reason": "one sentence explaining why this is or isn't a match"
  }}
]
"""
        result = call_llm_for_json(prompt, temperature=TEMP_AGENT_LOGIC)

        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("matches", result.get("results", []))
        return []

    except Exception as e:
        print(f"[tools] LLM sanctions check failed: {e}")
        return []


def check_exception_cache(
    name: str,
    dob:  str,
) -> dict | None:
    """
    Look up the Human Exception Cache in Qdrant for a prior officer decision on
    a similar profile. Returns the cached decision dict if a >0.90 match exists,
    else None (also None if the cache collection does not exist yet).
    """
    try:
        results = qdrant_client.query(
            collection_name = "exception_cache",
            query_text      = f"{name} {dob}",
            limit           = 1,
        )
        if results and results[0].score > 0.90:
            return results[0].metadata
        return None
    except Exception:
        return None


# ── Setup helper — run ONCE to ingest the sanctions list into Qdrant ────────

def setup_sanctions_collection() -> bool:
    """
    Create the Qdrant collection and ingest the mock sanctions list. Run once
    before the demo (e.g. in 00_setup.ipynb). Returns True on success.

    Note: the collection is NOT pre-created — qdrant_client.add() creates it
    with the named-vector config fastembed expects.
    """
    try:
        if not os.path.exists(SANCTIONS_LIST_PATH):
            print(f"[setup] ⚠ Sanctions file not found: {SANCTIONS_LIST_PATH}")
            return False

        with open(SANCTIONS_LIST_PATH, "r", encoding="utf-8") as f:
            sanctions = json.load(f)

        # Idempotent — drop an existing collection so re-runs are safe.
        existing = [c.name for c in qdrant_client.get_collections().collections]
        if QDRANT_COLLECTION_NAME in existing:
            qdrant_client.delete_collection(QDRANT_COLLECTION_NAME)
            print(f"[setup] Deleted existing collection '{QDRANT_COLLECTION_NAME}'")

        # fastembed embeds the 'documents' text for semantic search.
        qdrant_client.add(
            collection_name = QDRANT_COLLECTION_NAME,
            documents       = [
                f"{e.get('name', '')} {' '.join(e.get('aliases', []))}"
                for e in sanctions
            ],
            metadata        = sanctions,
            ids             = list(range(len(sanctions))),
        )

        print(f"[setup] ✅ Ingested {len(sanctions)} sanctions entries "
              f"into '{QDRANT_COLLECTION_NAME}'")
        return True

    except Exception as e:
        print(f"[setup] ❌ Failed to set up sanctions collection: {e}")
        return False


# ── Sanity check — run this file directly to test all connections ──────────

if __name__ == "__main__":
    print("=" * 60)
    print("tools.py — Connection Sanity Check")
    print("=" * 60)

    # 1. vLLM
    print("\n1. Testing vLLM connection...")
    try:
        response = call_text_llm(
            prompt      = "Reply with exactly three words: KYC system ready",
            temperature = 0.0,
        )
        if "[LLM unavailable]" in response:
            print("   ⚠ vLLM not reachable — start the Docker container first")
        else:
            print(f"   ✅ vLLM responded: '{response}'")
    except Exception as e:
        print(f"   ❌ vLLM error: {e}")

    # 2. JSON extraction
    print("\n2. Testing JSON extraction from LLM...")
    result = call_llm_for_json(
        prompt = 'Return this exact JSON: {"status": "ok", "system": "KYC"}'
    )
    if result.get("status") == "ok":
        print(f"   ✅ JSON extraction working: {result}")
    else:
        print(f"   ⚠ JSON extraction returned: {result}")

    # 3. Tesseract OCR
    print("\n3. Testing Tesseract OCR...")
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"   ✅ Tesseract installed — version {version}")
        langs = pytesseract.get_languages()
        has_hindi = "hin" in langs
        print(f"   {'✅' if has_hindi else '⚠'} Hindi (Devanagari) language pack: "
              f"{'installed' if has_hindi else 'NOT installed — run: sudo apt-get install tesseract-ocr-hin'}")
    except Exception as e:
        print(f"   ❌ Tesseract not installed: {e}")
        print("      Fix: sudo apt-get install tesseract-ocr tesseract-ocr-hin")

    # 4. Qdrant
    print("\n4. Testing Qdrant connection...")
    try:
        info = qdrant_client.get_collections()
        names = [c.name for c in info.collections]
        print(f"   ✅ Qdrant reachable — collections: {names or '(none yet)'}")
        if QDRANT_COLLECTION_NAME not in names:
            print(f"   ⚠ '{QDRANT_COLLECTION_NAME}' not found — "
                  f"run setup_sanctions_collection() in 00_setup.ipynb")
    except Exception as e:
        print(f"   ❌ Qdrant not reachable: {e}")
        print(f"      Embedded mode — check that '{QDRANT_LOCAL_PATH}' is writable "
              f"and no other process (app.py / a notebook kernel) has it open")

    print("\n" + "=" * 60)
    print("Run the fixes above, then re-run this file until all ✅")
    print("=" * 60)
