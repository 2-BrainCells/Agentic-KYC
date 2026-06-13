"""
config.py — Central Configuration
==================================
All tunable settings live here.
No logic — only values.

Every other file imports from this one:
    from config import VLLM_URL, RISK_REVIEW_THRESHOLD, ...

When something needs to change (a port, a threshold, a model name),
you change it HERE — not scattered across five different files.
"""

# ── vLLM Inference Server ──────────────────────────────────────────────────
# Your Llama-3 text model, served by vLLM on the MI300X.
# The /v1 path makes it OpenAI-compatible — agents call it like ChatGPT.

VLLM_URL        = "http://localhost:8000"
VLLM_API_BASE   = f"{VLLM_URL}/v1"
VLLM_API_KEY    = "not-needed"           # vLLM doesn't require a key locally
MODEL_NAME      = "meta-llama/Meta-Llama-3-8B-Instruct"
VLLM_TIMEOUT    = 60                     # seconds — increase if model is slow
VLLM_MAX_TOKENS = 1024                   # max tokens per agent response

# Temperature settings (do not change these without good reason)
TEMP_FIRST_PASS   = 0.2   # conservative — for standard extraction
TEMP_REFINE_PASS  = 0.7   # higher — surfaces hard-to-read fields on retry
TEMP_AGENT_LOGIC  = 0.1   # near-zero — for compliance/financial decisions
TEMP_EXPLANATION  = 0.4   # some creativity — for NL explanation paragraph


# ── Qdrant Vector Database ─────────────────────────────────────────────────
# Stores the mock sanctions list as embeddings for semantic search.
# Embedded local mode — data lives in a folder on disk, no Docker server.
# NOTE: only ONE process can open this folder at a time (it takes a file
# lock). Don't run setup / bulk_acceptance_test.py while app.py is running.

QDRANT_LOCAL_PATH      = "./qdrant_local_db" # embedded storage folder
QDRANT_URL             = "http://localhost:6333"  # only used if you switch back to server mode
QDRANT_COLLECTION_NAME = "india_sanctions"   # created by your setup script
QDRANT_TOP_K           = 5                   # return top 5 nearest matches


# ── Telemetry ─────────────────────────────────────────────────────────────
# vLLM exposes a Prometheus metrics feed automatically.
# telemetry.py polls this every 2 seconds.

VLLM_METRICS_URL    = f"{VLLM_URL}/metrics"
TELEMETRY_REFRESH_S = 2      # how often the dashboard refreshes (seconds)


# ── Compliance Screening Thresholds ───────────────────────────────────────
# These control when a sanctions match triggers different actions.
# Tweak during testing — do NOT change on demo day.

FUZZY_CLEAR_BELOW     = 0.60   # score < 0.60  → definitely not a match → CLEAR
FUZZY_AMBIGUOUS_LOW   = 0.60   # 0.60 – 0.85   → ambiguous → self-correction
FUZZY_AMBIGUOUS_HIGH  = 0.85   # score > 0.85  → confirmed hit → REJECT
REFINEMENT_MAX_TRIES  = 1      # how many times the loop can retry before giving up


# ── Risk Score Thresholds ─────────────────────────────────────────────────
# risk_score is 0.0 – 1.0.  These map it to a decision band.

RISK_AUTO_APPROVE_BELOW = 0.30   # score < 0.30  → AUTO APPROVE
RISK_REVIEW_BELOW       = 0.65   # 0.30 – 0.65   → ROUTE TO HUMAN (REVIEW)
                                  # score > 0.65  → ROUTE TO HUMAN (HIGH RISK)


# ── Risk Score Weights ────────────────────────────────────────────────────
# Must add up to 1.0 — each agent contributes this fraction of the final score.

WEIGHT_ID_VERIFICATION  = 0.30
WEIGHT_COMPLIANCE       = 0.40
WEIGHT_NETWORK_RISK     = 0.20
WEIGHT_FINANCIAL        = 0.10


# ── Indian Income Bands (INR per annum) ───────────────────────────────────
# Used by the financial agent to check income plausibility vs occupation.
# These are approximate 2024 market bands — adjust as needed.

INCOME_BANDS = {
    "entry":        (300_000,   600_000),    # ₹3L – ₹6L
    "middle":       (600_000,  1_500_000),   # ₹6L – ₹15L
    "upper_middle": (1_500_000, 5_000_000),  # ₹15L – ₹50L
    "hni":          (5_000_000, float("inf"))  # ₹50L+  (High Net Worth)
}

# Occupations that are plausible for each band
# Financial agent uses this to flag income/occupation mismatches
OCCUPATION_INCOME_MAP = {
    "software engineer":        "middle",
    "senior software engineer": "upper_middle",
    "doctor":                   "upper_middle",
    "lawyer":                   "upper_middle",
    "government employee":      "entry",
    "teacher":                  "entry",
    "business owner":           "upper_middle",   # needs GST/ITR to verify
    "independent consultant":   "upper_middle",   # needs contracts to verify
    "student":                  "entry",
    "retired":                  "entry",
    "other":                    "entry",
}

# If declared income is in HNI band, these documents MUST be present
# Financial agent flags if they are missing
HNI_REQUIRED_DOCS = [
    "gst_registration_number",
    "income_tax_return",
    "ca_certified_accounts",
]


# ── Document & Data Paths ─────────────────────────────────────────────────
# Where files live on the AMD cloud notebook filesystem.

UPLOAD_DIR          = "./uploads"             # customer-uploaded Aadhaar/salary slip
MOCK_DATA_DIR       = "./mock_data"           # your 20 test dossiers
SANCTIONS_LIST_PATH = "./mock_data/sanctions_list.json"
BULK_DOSSIERS_PATH  = "./mock_data/bulk_customers.json"

# Human review inbox — cases routed to a human (ROUTE_TO_HUMAN) are written
# here so a compliance officer can action them later. JSON on disk so the
# queue survives app restarts (Streamlit session state does not).
REVIEW_QUEUE_PATH   = "./review_queue/queue.json"

# Salary-slip verification: how far the declared annual income may differ from
# the figure parsed off an uploaded salary slip before it counts as a
# discrepancy. 0.25 = ±25%.
SALARY_MATCH_TOLERANCE = 0.25


# ── Aadhaar / PAN Rules ───────────────────────────────────────────────────

AADHAAR_LAST4_LENGTH  = 4       # Aadhaar always displayed as XXXX XXXX 1234
PAN_FORMAT            = r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$"  # e.g. BKXPS9876M
PAN_AADHAAR_LINK_REQUIRED = True   # mandatory per RBI 2023 — flag if missing


# ── Indian Regulatory References ─────────────────────────────────────────
# Used in the explanation agent's output to cite specific regulations.
# These strings appear in the NL explanation the LLM generates.

REG_RBI_KYC         = "RBI KYC Master Direction (2016, updated 2023)"
REG_PMLA            = "Prevention of Money Laundering Act 2002 (PMLA)"
REG_FEMA            = "Foreign Exchange Management Act 1999 (FEMA)"
REG_PAN_AADHAAR     = "RBI Circular RBI/2023-24/69 — PAN-Aadhaar Linkage"

WATCHLISTS = [
    "FIU-IND (Financial Intelligence Unit — India)",
    "ED (Enforcement Directorate) — PMLA Cases",
    "RBI Wilful Defaulters List",
    "UN Security Council Consolidated Sanctions List",
    "INTERPOL Red Notices",
]


# ── Quick check — run this file to confirm config loads cleanly ───────────
if __name__ == "__main__":
    print("✅ config.py loaded successfully\n")
    print(f"  vLLM          : {VLLM_API_BASE}")
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Qdrant        : {QDRANT_LOCAL_PATH} / {QDRANT_COLLECTION_NAME}")
    print(f"  Thresholds    : clear<{FUZZY_CLEAR_BELOW} | "
          f"ambiguous {FUZZY_AMBIGUOUS_LOW}–{FUZZY_AMBIGUOUS_HIGH} | "
          f"hit>{FUZZY_AMBIGUOUS_HIGH}")
    print(f"  Risk bands    : approve<{RISK_AUTO_APPROVE_BELOW} | "
          f"review<{RISK_REVIEW_BELOW} | high>{RISK_REVIEW_BELOW}")
    print(f"  Watchlists    : {len(WATCHLISTS)} lists configured")
    print(f"  Income bands  : {list(INCOME_BANDS.keys())}")