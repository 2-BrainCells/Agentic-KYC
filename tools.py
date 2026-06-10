"""
tools.py — External Tool Calls
================================
Next file after config.py. Every agent imports from here.

Install dependencies first (run once in your notebook terminal):
    pip install openai pytesseract pillow qdrant-client[fastembed]
    sudo apt-get install -y tesseract-ocr tesseract-ocr-hin

    tesseract-ocr     → reads English text from Aadhaar card
    tesseract-ocr-hin → reads Devanagari (Hindi) text from Aadhaar card

Three tool groups:
  1. LLM Tools     → calls vLLM for text generation and JSON extraction
  2. Document Tools → OCR on Aadhaar card + structured field parsing
  3. Sanctions Tools → Qdrant semantic search + LLM-as-judge fallback

All functions have try/except — they NEVER crash the demo.
On failure they return a safe default and log the error.
"""

import json
import re
import os
import pytesseract

from PIL                    import Image
from openai                 import OpenAI
from qdrant_client          import QdrantClient
from qdrant_client.models   import Distance, VectorParams, PointStruct

from config import (
    VLLM_API_BASE, VLLM_API_KEY, MODEL_NAME,
    VLLM_TIMEOUT, VLLM_MAX_TOKENS,
    TEMP_FIRST_PASS, TEMP_REFINE_PASS,
    TEMP_AGENT_LOGIC, TEMP_EXPLANATION,
    QDRANT_URL, QDRANT_COLLECTION_NAME, QDRANT_TOP_K,
    FUZZY_CLEAR_BELOW, FUZZY_AMBIGUOUS_HIGH,
    SANCTIONS_LIST_PATH,
)


# ═══════════════════════════════════════════════════════════════════════════
# CLIENTS  —  created once, reused across all calls
# ═══════════════════════════════════════════════════════════════════════════

# vLLM — OpenAI-compatible endpoint running Llama-3 on your MI300X
llm_client = OpenAI(
    base_url = VLLM_API_BASE,
    api_key  = VLLM_API_KEY,
)

# Qdrant — stores your mock sanctions list as embeddings
qdrant_client = QdrantClient(url=QDRANT_URL)


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — LLM TOOLS
# ═══════════════════════════════════════════════════════════════════════════

def call_text_llm(
    prompt:        str,
    temperature:   float = TEMP_AGENT_LOGIC,
    system_prompt: str   = "You are a precise KYC compliance assistant.",
) -> str:
    """
    Sends a prompt to vLLM and returns the model's text response.
    Used by every agent for reasoning, decisions, and explanations.

    Args:
        prompt:        What you want the model to do.
        temperature:   0.1 = very precise.  0.7 = more creative.
        system_prompt: Gives the model its role context.

    Returns:
        The model's response as a plain string.
        Returns "[LLM unavailable]" if the server is down — demo safe.

    Example:
        response = call_text_llm(
            prompt      = "Is 'Priya Sharma' a common Indian name? Answer yes or no.",
            temperature = 0.1,
        )
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
    Sends a prompt to vLLM and parses the response as JSON.
    Used when an agent needs structured data back (not free text).

    Handles three common failure modes:
      - Model wraps JSON in ```json ... ``` fences → strips them
      - Model adds text before/after the JSON → extracts the JSON block
      - Model returns invalid JSON → returns {}

    Returns:
        Parsed dict, or {} on any failure.

    Example:
        result = call_llm_for_json(
            prompt = "Extract name and DOB from this text as JSON: ..."
        )
        name = result.get("full_name", "UNKNOWN")
    """
    raw = call_text_llm(prompt, temperature, system_prompt)

    # Strip markdown code fences if present: ```json ... ```
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()

    # If there is text before the JSON, find the first { or [
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


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — DOCUMENT TOOLS  (Aadhaar OCR + structured field parsing)
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_image(image_path: str) -> str:
    """
    Runs OCR on the Aadhaar card image using Tesseract.
    Reads both English and Hindi (Devanagari) text.

    Args:
        image_path: Path to the Aadhaar card image file.

    Returns:
        Raw text extracted from the image.
        Returns "" on failure — agents handle this gracefully.

    Requires:
        sudo apt-get install tesseract-ocr tesseract-ocr-hin
    """
    try:
        img  = Image.open(image_path)
        # "eng+hin" tells Tesseract to read English AND Devanagari
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
    Sends the raw OCR text to Llama-3 and asks it to extract
    Aadhaar identity fields as structured JSON.

    Two modes (controlled by `directive`):
      "standard_first_pass" — extract all standard fields
      "refinement_pass"     — focus on father's name (care_of field),
                              middle name, place of birth

    Args:
        ocr_text:    Raw text from extract_text_from_image().
        directive:   Which fields to prioritise.
        temperature: Use TEMP_FIRST_PASS normally, TEMP_REFINE_PASS on retry.

    Returns:
        Dict of extracted identity fields. Missing fields are null.

    Example return value:
        {
          "full_name_english":    "PRIYA SHARMA",
          "full_name_devanagari": "प्रिया शर्मा",
          "given_name":           "PRIYA",
          "surname":              "SHARMA",
          "father_name":          null,
          "dob":                  "1992-09-08",
          "gender":               "F",
          "aadhaar_last4":        "7823",
          "address":              "B-204 Green Park Bengaluru Karnataka",
          "pin_code":             "560034",
          "pan_number":           null,
          "extraction_pass":      0
        }
    """
    # Choose the right instruction based on which pass this is
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

    # If LLM returned empty dict, return a safe structure
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
    Assigns a confidence score (0.0 – 1.0) to each extracted field.
    Simple heuristic: None / empty = 0.0, populated = 0.85 – 0.99.

    Used by the Streamlit UI to colour-code bounding boxes:
      score >= 0.75 → green box
      score  < 0.75 → amber/red box (flagged for human review)
    """
    confidence = {}
    high_confidence_fields = {
        "dob", "aadhaar_last4", "aadhaar_masked", "pin_code", "gender"
    }
    for field, value in extracted.items():
        if value is None or value == "":
            confidence[field] = 0.0
        elif field in high_confidence_fields:
            confidence[field] = 0.97   # these are numeric / format-checkable
        else:
            confidence[field] = 0.87   # text fields — moderate confidence

    return confidence


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — SANCTIONS TOOLS
# ═══════════════════════════════════════════════════════════════════════════

def _sanctions_match_score(query_name: str, entry_name: str, aliases: list = None) -> float:
    """
    Deterministic 0.0–1.0 name-match score between a customer name and a
    watchlist entry (including its aliases).

    Blend of two signals (50/50):
      - token overlap: shared name components / total components
        ("Vikram Malhotra" vs "Vikram Suresh Malhotra" → 2 of 3 tokens)
      - difflib character similarity on the full strings

    Why deterministic instead of embedding cosine: the screening thresholds in
    config.py (0.60 / 0.85) are calibrated against THIS scale. Embedding scores
    drift with model choice; this never does — same result on a laptop and on
    the MI300X. What to do with this: call it with the entry's name and aliases,
    take the best score across query variants.

    Alias matches are capped at 0.81: an AKA on a watchlist is reported
    intelligence, not a verified legal name, so even an exact alias match
    stays in the AMBIGUOUS band and needs more identity signals (father's
    name, DOB) before it can confirm. This is what arms the self-correction
    loop — "Vikram Malhotra" hits the alias of "Vikram Suresh Malhotra" at
    exactly 0.81, not 1.0.
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
    """Formats a sanctions-list entry as a standard match dict for the agents."""
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
    """Loads the mock sanctions JSON from disk. Returns [] if missing."""
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
    FALLBACK: deterministic fuzzy scan of the sanctions JSON on disk.
    Needs no Qdrant and no LLM — always available, always the same answer.
    Used when Qdrant is down (e.g. running the pipeline on a laptop).
    Returns the same format as query_sanctions_db().
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
    Searches the Qdrant sanctions database for names similar to the input.
    Uses semantic/vector search — finds fuzzy matches across spelling variants.

    Args:
        name:          The customer's full name.
        name_variants: Extra forms of the name (Devanagari, inverted, etc.)
        top_k:         How many nearest matches to return.

    Returns:
        List of matches, each with name, score, and list source.
        Returns [] on any failure (Qdrant down, collection missing, etc.)

    Example return:
        [
          {
            "matched_name": "Vikram Suresh Malhotra",
            "match_score":  0.81,
            "list_source":  "ED — PMLA Case No. 2019/DEL/0147",
            "aliases":      ["V.S. Malhotra", "Vikram Malhotra"],
            "dob_range":    "1966–1970"
          }
        ]
    """
    try:
        # Build a combined query from all name variants
        query_text = name
        if name_variants:
            query_text = " | ".join([name] + name_variants)

        # Qdrant with fastembed — handles the embedding automatically.
        # Vector search RETRIEVES the candidates (spelling/script variants);
        # the deterministic scorer then assigns the calibrated match_score,
        # so the 0.60/0.85 thresholds in config.py always mean the same thing.
        results = qdrant_client.query(
            collection_name = QDRANT_COLLECTION_NAME,
            query_text      = query_text,
            limit           = top_k,
        )

        queries = [q for q in [name] + (name_variants or []) if q]
        matches = []
        for r in results:
            entry = r.payload or {}
            score = max(
                (_sanctions_match_score(q, entry.get("name", ""), entry.get("aliases"))
                 for q in queries),
                default=0.0,
            )
            if score >= FUZZY_CLEAR_BELOW:          # only return meaningful matches
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
        # Last resort — no Qdrant AND no local file: ask the LLM directly
        return llm_sanctions_check(name, name_variants)


def llm_sanctions_check(
    name:          str,
    name_variants: list = None,
) -> list:
    """
    FALLBACK: uses Llama-3 directly to check if a name appears on
    the sanctions list (loaded from your JSON file).

    Used when Qdrant is unavailable or not yet set up.
    This is the LLM-as-judge approach — no embeddings needed.

    Returns same format as query_sanctions_db().
    """
    try:
        # Load the mock sanctions list from disk
        if not os.path.exists(SANCTIONS_LIST_PATH):
            print(f"[tools] Sanctions list not found at {SANCTIONS_LIST_PATH}")
            return []

        with open(SANCTIONS_LIST_PATH, "r", encoding="utf-8") as f:
            sanctions = json.load(f)

        # Format as a readable list for the LLM
        sanctions_text = "\n".join([
            f"- {e.get('name')} | DOB: {e.get('dob_range','?')} | "
            f"Case: {e.get('case_ref','?')} | List: {e.get('list_source','?')}"
            for e in sanctions[:50]   # cap at 50 to stay within context window
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

        # Result could be a list or a dict with a list inside
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
    Checks the Human Exception Cache in Qdrant.
    If a compliance officer has already reviewed a similar case,
    return their decision so the system can learn from it.

    Returns:
        {"decision": "APPROVE_WITH_EDD", "rationale": "..."} if found.
        None if no matching prior case.
    """
    try:
        results = qdrant_client.query(
            collection_name = "exception_cache",
            query_text      = f"{name} {dob}",
            limit           = 1,
        )
        if results and results[0].score > 0.90:
            return results[0].payload
        return None
    except Exception:
        # Cache not set up yet — that is fine, return None
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SETUP HELPER  —  run this ONCE to ingest your sanctions list into Qdrant
# ═══════════════════════════════════════════════════════════════════════════

def setup_sanctions_collection() -> bool:
    """
    Creates the Qdrant collection and ingests your mock sanctions list.
    Run this ONCE before starting the demo (add a cell in 00_setup.ipynb).

    Returns True on success, False on failure.

    Usage in 00_setup.ipynb:
        from tools import setup_sanctions_collection
        setup_sanctions_collection()
    """
    try:
        # Load your mock sanctions JSON
        if not os.path.exists(SANCTIONS_LIST_PATH):
            print(f"[setup] ⚠ Sanctions file not found: {SANCTIONS_LIST_PATH}")
            return False

        with open(SANCTIONS_LIST_PATH, "r", encoding="utf-8") as f:
            sanctions = json.load(f)

        # Delete + recreate the collection (idempotent — safe to re-run)
        existing = [c.name for c in qdrant_client.get_collections().collections]
        if QDRANT_COLLECTION_NAME in existing:
            qdrant_client.delete_collection(QDRANT_COLLECTION_NAME)
            print(f"[setup] Deleted existing collection '{QDRANT_COLLECTION_NAME}'")

        # Create with fastembed — it handles vector size automatically
        qdrant_client.create_collection(
            collection_name = QDRANT_COLLECTION_NAME,
            vectors_config  = VectorParams(size=384, distance=Distance.COSINE),
        )

        # Ingest each sanctions entry
        # fastembed will embed the 'document' text for semantic search
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


# ═══════════════════════════════════════════════════════════════════════════
# SANITY CHECK  —  run this file directly to test all connections
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("tools.py — Connection Sanity Check")
    print("=" * 60)

    # 1. Test vLLM
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

    # 2. Test JSON extraction
    print("\n2. Testing JSON extraction from LLM...")
    result = call_llm_for_json(
        prompt = 'Return this exact JSON: {"status": "ok", "system": "KYC"}'
    )
    if result.get("status") == "ok":
        print(f"   ✅ JSON extraction working: {result}")
    else:
        print(f"   ⚠ JSON extraction returned: {result}")

    # 3. Test Tesseract OCR
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

    # 4. Test Qdrant
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
        print("      Fix: docker run -p 6333:6333 qdrant/qdrant")

    print("\n" + "=" * 60)
    print("Run the fixes above, then re-run this file until all ✅")
    print("=" * 60)