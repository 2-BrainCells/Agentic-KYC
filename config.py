"""
config.py — Central configuration.

All tunable settings (endpoints, thresholds, weights, income bands, regulatory
strings) live here — values only, no logic. Every other module imports from
this file, so a change here propagates everywhere. VLLM_URL is the only
environment-specific value.
"""

# ── vLLM inference server (OpenAI-compatible Llama-3 on the MI300X) ─────────
VLLM_URL        = "http://localhost:8000"
VLLM_API_BASE   = f"{VLLM_URL}/v1"
VLLM_API_KEY    = "not-needed"           # vLLM needs no key locally
MODEL_NAME      = "meta-llama/Meta-Llama-3-8B-Instruct"
VLLM_TIMEOUT    = 60                     # seconds per request
VLLM_MAX_TOKENS = 1024                   # max tokens per agent response

# Per-task temperatures (do not change without good reason)
TEMP_FIRST_PASS   = 0.0   # greedy — deterministic standard extraction (same image → same fields → same score)
TEMP_REFINE_PASS  = 0.7   # higher — surfaces hard-to-read fields on retry
TEMP_AGENT_LOGIC  = 0.1   # near-zero — compliance/financial decisions
TEMP_EXPLANATION  = 0.4   # some creativity — NL explanation paragraph


# ── Qdrant vector database (embedded local mode) ───────────────────────────
# Stores the mock sanctions list as embeddings. The folder takes an exclusive
# file lock — only one process may open it at a time. Don't run setup or
# bulk_acceptance_test.py while app.py is running.
QDRANT_LOCAL_PATH      = "./qdrant_local_db"
QDRANT_URL             = "http://localhost:6333"  # only for server mode
QDRANT_COLLECTION_NAME = "india_sanctions"
QDRANT_TOP_K           = 5                         # nearest matches to return


# ── Telemetry ──────────────────────────────────────────────────────────────
VLLM_METRICS_URL    = f"{VLLM_URL}/metrics"
TELEMETRY_REFRESH_S = 2      # dashboard refresh interval (seconds)


# ── GPU memory footprint (the platform is deliberately CAPPED to a slice) ────
# vLLM otherwise grabs ~90% of the card for its KV-cache pool, which makes it
# look like the demo "needs" a 192 GB MI300X. We pin it to a small fraction so
# the whole platform fits on a far smaller/cheaper GPU — and so the telemetry
# tab tells that efficiency story instead of showing a near-full card.
#
# Put these on the `vllm serve` command (see CLAUDE.md §11):
#   --gpu-memory-utilization {GPU_MEMORY_UTILIZATION} --max-model-len {MAX_MODEL_LEN}
GPU_TOTAL_VRAM_GB      = 192     # physical MI300X HBM3 (the card we run on)
GPU_MEMORY_UTILIZATION = 0.20    # vLLM may use AT MOST this fraction of the card
MAX_MODEL_LEN          = 8192    # context cap — keeps the KV-cache pool small
# Allocated budget vLLM is pinned to. Llama-3-8B weights (~16 GB bf16) + a
# bounded KV-cache pool live inside this; the rest of the card stays free.
VRAM_BUDGET_GB         = round(GPU_TOTAL_VRAM_GB * GPU_MEMORY_UTILIZATION)  # ≈ 38 GB


# ── Compliance screening thresholds ────────────────────────────────────────
# score < 0.60 → CLEAR · 0.60–0.85 → AMBIGUOUS (self-correction) · > 0.85 → hit
FUZZY_CLEAR_BELOW     = 0.60
FUZZY_AMBIGUOUS_LOW   = 0.60
FUZZY_AMBIGUOUS_HIGH  = 0.85
REFINEMENT_MAX_TRIES  = 1      # max self-correction retries


# ── Risk-score decision thresholds (risk_score is 0.0–1.0) ─────────────────
RISK_AUTO_APPROVE_BELOW = 0.30   # < 0.30        → AUTO APPROVE
RISK_REVIEW_BELOW       = 0.65   # 0.30–0.65     → ROUTE TO HUMAN (review)
                                  # > 0.65        → ROUTE TO HUMAN (high risk)


# ── Risk-score weights (must sum to 1.0) ───────────────────────────────────
WEIGHT_ID_VERIFICATION  = 0.30
WEIGHT_COMPLIANCE       = 0.40
WEIGHT_NETWORK_RISK     = 0.20
WEIGHT_FINANCIAL        = 0.10


# ── ID-verification sub-checks ──────────────────────────────────────────────
# The ID factor is GRADED (not pass/fail-binary) so small OCR/extraction noise
# moves the risk score by a little, not by a 0.24 cliff. These sub-weights are
# how much each failed check adds to the ID risk contribution (capped at 1.0).
ID_PENALTY_NAME      = 0.40   # name similarity below NAME_MATCH_THRESHOLD
ID_PENALTY_DOB       = 0.30   # DOB does not match
ID_PENALTY_PIN       = 0.15   # PIN code does not match
ID_PENALTY_ADDRESS   = 0.15   # address does not match (hard gate, see below)

# Hard gates: any of these failing blocks auto-approval. A failed identity
# check means the document does not match the declared data → the case is
# auto-rejected and the customer is asked to re-apply with correct documents.
NAME_MATCH_THRESHOLD    = 0.80   # extracted vs declared name similarity to pass
ADDRESS_MATCH_THRESHOLD = 0.70   # extracted vs declared address similarity to pass


# ── Indian income bands (INR per annum) ────────────────────────────────────
# Used by the financial agent to check income plausibility vs occupation.
INCOME_BANDS = {
    "entry":        (300_000,   600_000),      # ₹3L – ₹6L
    "middle":       (600_000,  1_500_000),     # ₹6L – ₹15L
    "upper_middle": (1_500_000, 5_000_000),    # ₹15L – ₹50L
    "hni":          (5_000_000, float("inf")), # ₹50L+ (High Net Worth)
}

# Plausible income band for each occupation (mismatches get flagged)
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

# Documents an HNI-band declaration must include (financial agent flags if missing)
HNI_REQUIRED_DOCS = [
    "gst_registration_number",
    "income_tax_return",
    "ca_certified_accounts",
]


# ── Data paths ─────────────────────────────────────────────────────────────
UPLOAD_DIR          = "./uploads"
MOCK_DATA_DIR       = "./mock_data"
SANCTIONS_LIST_PATH = "./mock_data/sanctions_list.json"
BULK_DOSSIERS_PATH  = "./mock_data/bulk_customers.json"

# Human review inbox — ROUTE_TO_HUMAN cases are persisted here as JSON so the
# queue survives app restarts (Streamlit session state does not).
REVIEW_QUEUE_PATH   = "./review_queue/queue.json"

# Applicant registry — a small on-disk SQLite DB of every accepted/rejected/held
# application, keyed by a unique application number. Powers the re-application
# guard: an already-accepted applicant is told so; a recently-rejected one must
# wait out the cooldown before retrying.
APPLICANT_DB_PATH         = "./applicant_registry/applications.db"
REAPPLY_COOLDOWN_MINUTES  = 10                  # rejected applicant must wait this long
DOCUMENT_SUBMISSION_EMAIL = "example@example.com"  # where HOLD applicants mail documents
APPLICATION_NUMBER_PREFIX = "KYC-APP"           # prefix for generated application numbers

# Max allowed gap between declared income and a parsed salary slip before it
# counts as a discrepancy. 0.25 = ±25%.
SALARY_MATCH_TOLERANCE = 0.25


# ── Aadhaar / PAN rules ────────────────────────────────────────────────────
AADHAAR_LAST4_LENGTH      = 4                          # displayed as XXXX XXXX 1234
PAN_FORMAT                = r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$"   # e.g. BKXPS9876M
PAN_AADHAAR_LINK_REQUIRED = True                       # mandatory per RBI 2023


# ── Indian regulatory references (cited in the LLM explanation) ────────────
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


# ── Sanity check — run this file to confirm config loads cleanly ───────────
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
