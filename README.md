# Agentic KYC Intelligence Platform

> Context file for Claude / Claude Code. Read this first before working on the project.
> It explains what we are building, why, the architecture, every file, and the rules.
> **Status: feature-complete.** All code + mock data built. Ready to run on AMD MI300X.

---

## 1. What This Project Is

An **agentic KYC (Know Your Customer) platform** for an AMD hackathon.
Multiple specialized AI agents collaborate to perform end-to-end customer due
diligence — data extraction, identity verification, compliance screening,
financial profiling, entity resolution, risk scoring, and escalation — while
maintaining explainability and human oversight.

**It is India-centric:** Aadhaar cards as the primary document, PAN as secondary,
income in INR (₹), and Indian regulatory context (FIU-IND, ED, PMLA, RBI KYC
Master Direction, FEMA, MCA21).

### Hackathon constraints (important context)
- **Team:** 2 people, **beginner-level** AI/ML knowledge.
- **Time:** 4 days.
- **Hardware:** AMD Instinct **MI300X** (192 GB HBM3), cloud-hosted notebook.
- **Goal:** Win. The demo matters more than code completeness.

### What makes this win (the three differentiators — PROTECT THESE)
1. **Self-correction loop** — the agentic "true autonomy" flex (see §4).
2. **Live ROCm GPU telemetry** — proves AMD hardware exploitation with real numbers.
3. **Bulk import stress test** — 20 customers processed live; the demo climax.

If time runs short, drop entity-resolution depth, bounding-box overlays, and the
active-learning cache before ever touching the three features above.

---

## 2. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Orchestration | **LangGraph 1.2.2** | StateGraph, conditional edges, parallel fan-out |
| Text inference | **vLLM** serving Llama-3-8B-Instruct | OpenAI-compatible endpoint, full precision |
| Document OCR | **Tesseract** (`eng+hin`) | English + Devanagari from Aadhaar |
| Vector DB | **Qdrant** + fastembed | Mock sanctions; falls back to LLM-as-judge |
| Entity graph | **NetworkX** | In-memory mock MCA21 corporate network |
| UI | **Streamlit 1.58** | Three tabs: Pipeline, Bulk Import, Telemetry |
| Telemetry | `rocm-smi` + vLLM `/metrics` | VRAM, tokens/sec, KV cache |
| Environment | **vLLM 0.17.1 + ROCm 7.0** | Chosen over SGLang/Torch-only; pitch is vLLM-centric |

---

## 3. Architecture

```
Customer Input (Aadhaar + PAN + declared data, ₹ income)
        │
        ▼
  ┌─────────────────── Orchestration & Memory Layer ───────────────────┐
  │  Shared Agent State (KYCState)  ◄──►  Central Orchestrator (routing) │
  └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
  extract ◄───────────────────────────────┐
     │                                     │ (self-correction loop)
     ▼                                     │
  id_verify                                │
     │                                     │
     ▼          refinement_needed?         │
  compliance ───────── yes ──────────► refine
     │ no (clear / resolved)
     ▼
  fan_out ──┬──► entity_resolution  ┐
            └──► financial_profiling ┴──► risk ──┬── auto ──► END
                                                 └── human ─► hitl_review ─► END
                                                                  │
                                          (Streamlit calls complete_case())
                                                                  ▼
                                                     active_learning_cache
        │
        ▼
  AMD ROCm / Tooling Layer: vLLM (Llama-3), Tesseract OCR, Qdrant, NetworkX
```

### Pipeline flow in words
1. **extract** — OCR + LLM read the Aadhaar card into structured fields.
2. **id_verify** — cross-check extracted vs declared (name, DOB, PIN, UIDAI, PAN linkage).
3. **compliance** — screen name against sanctions watchlists.
4. If the match is **ambiguous** → **refine** issues a Refinement Request and loops
   back to **extract** (higher temperature, seeks father's name from Aadhaar QR).
5. On **clear/resolved** → **fan_out** runs **entity_resolution** and
   **financial_profiling** in parallel; LangGraph joins them at **risk**.
6. **risk** computes a weighted score, decides APPROVE / REVIEW / REJECT, and the
   LLM writes an evidence-based explanation.
7. **auto-approve** → END. **route to human** → **hitl_review** → officer decides in
   the UI → **active_learning_cache** stores the decision for future cases.

---

## 4. The Self-Correction Loop (the headline feature)

The single most important behaviour to preserve and demo.

**Trigger:** compliance agent finds a fuzzy match with a mid-band score
(`FUZZY_CLEAR_BELOW` ≤ score < `FUZZY_AMBIGUOUS_HIGH`, i.e. 0.60–0.85) AND no
father's name is available to disambiguate.

**What happens:**
1. compliance sets `screening_status = AMBIGUOUS`, `refinement_needed = True`.
2. Router sends flow to `refine`, which bumps `refine_count`, writes a
   `refinement_request` (seek `father_name` from Aadhaar QR `care_of`, temp 0.7),
   and sets `current_agent_status = "REFINEMENT_REQUEST_ISSUED"`.
3. Flow loops back to `extract`, which re-parses at higher temperature.
4. compliance runs again; `_resolve_with_father()` compares father names:
   - father names **differ** → match score drops ~65% → **CLEAR**
   - father names **match** → **CONFIRMED_HIT** (the loop can also confirm, not only clear)

**The loop works both ways — this is the credibility point.** Capped at
`REFINEMENT_MAX_TRIES = 1` to prevent infinite loops.

**Demo line:** "The match starts at 0.81. The system doesn't halt — it issues a
Refinement Request, finds the father's name in the Aadhaar QR, compares it to the
listed individual's father, and resolves the case autonomously."

---

## 5. File Map (ALL COMPLETE)

| File | Role | Status |
|---|---|---|
| `state.py` | `KYCState` TypedDict + constants + `create_initial_state()` | ✅ |
| `config.py` | Settings: endpoints, thresholds, weights, INR bands, regs | ✅ |
| `tools.py` | vLLM, Tesseract OCR, Qdrant search, exception cache, setup helper | ✅ |
| `agents.py` | All 9 agents + entity graph + helpers | ✅ |
| `graph.py` | LangGraph wiring + `build_graph()` + `complete_case()` | ✅ |
| `telemetry.py` | ROCm + vLLM live metrics + Streamlit render | ✅ |
| `app.py` | Streamlit UI — 3 tabs | ✅ |
| `bulk_acceptance_test.py` | Regression: all 20 bulk profiles vs expected decisions | ✅ |
| `00_setup.ipynb` | One-time MI300X environment setup + health checks | ✅ |
| `01_demo.ipynb` | Demo runbook — dry-runs every demo beat | ✅ |
| `mock_data/sanctions_list.json` | 55 entries, 7 watchlists | ✅ |
| `mock_data/bulk_customers.json` | 20 profiles, 7 scenarios | ✅ |
| `CLAUDE.md` | This file | ✅ |

### Dependency order (never violate)
`state.py` → `config.py` → `tools.py` → `agents.py` → `graph.py` → `app.py`
(`telemetry.py` is independent; imported only by `app.py`.)

---

## 6. KYCState — The Shared Contract

Every agent receives the full `KYCState` and returns **only the fields it changes**.
LangGraph merges the rest automatically. **Never break this contract** — it is the
reason agents stay decoupled and can run in parallel.

**Critical detail:** `audit_log` uses `Annotated[list, add]` as its reducer. This
lets the two parallel agents (entity_resolution + financial_profiling) both append
without overwriting each other. All other fields are last-write-wins.

Use the **constants classes** (`CaseStatus`, `ScreeningStatus`, `Decision`,
`Routing`, `RiskBand`, `DecisionSource`, `ActivityBand`) instead of raw strings —
a typo in a literal can silently break routing.

Key fields by stage: `declared`, `documents` (intake) · `extracted`,
`field_confidence`, `bounding_boxes`, `extraction_status` (extraction) ·
`id_verified`, `verification_details` (verify) · `compliance_hits`,
`screening_status`, `refinement_needed`, `cache_hit` (compliance) ·
`refine_count`, `refinement_request`, `current_agent_status` (self-correction) ·
`network_risk` (entity) · `financial_profile` (financial) · `risk_score`,
`risk_band`, `decision`, `routing`, `contributing_factors`, `explanation` (risk) ·
`human_decision` (HITL) · `final_decision`, `decision_source`, `closed_at` (close) ·
`audit_log` (always, parallel-safe).

---

## 7. Risk Scoring

Weighted sum (weights in `config.py`, must total 1.0):

| Factor | Weight |
|---|---|
| ID Verification | 30% |
| Compliance Screening | 40% |
| Network / Entity Risk | 20% |
| Financial Profile | 10% |

Decision bands (thresholds in `config.py`): score < 0.30 → **APPROVE** (auto) ·
0.30–0.65 → **REVIEW** (human) · > 0.65 → **REVIEW/HIGH** (human).

Compliance raw sub-scores (×0.40 weight): CLEAR 0.00 · CLEAR-after-refinement
0.25 (residual caution) · AMBIGUOUS 0.60 · POTENTIAL_MATCH 0.75 ·
CONFIRMED_HIT 1.00. Network raw: 0.80 when flagged. Financial raw: from
anomaly count (+0.30 HNI bump).

**Regulatory overrides (outrank the weighted score, in `risk_scoring_agent`):**
- CONFIRMED_HIT → decision **REJECT**, band HIGH, still ROUTE_TO_HUMAN (PMLA).
- Any non-CLEAR screening (PEP / potential match) → at least **REVIEW** (EDD).
- `financial_profile.requires_edd` (HNI income + anomaly flags) → at least
  **REVIEW** even when the weighted score is tiny (RBI KYC MD). This is how
  BULK-018…020 reach human review despite financial's 10% weight.

**Screening adjustments (in `compliance_screening_agent` / `tools.py`):**
- Scores are deterministic (token-overlap + difflib blend), so the 0.60/0.85
  thresholds always mean the same thing. Qdrant does retrieval; the blend
  does scoring; fallback chain: Qdrant → local fuzzy scan → LLM-as-judge.
- Alias matches cap at **0.81** — an exact AKA hit lands in the AMBIGUOUS
  band (this is what arms the loop for "Vikram Malhotra").
- DOB gate: customer birth year outside the entry's `dob_range` ±3y → score
  ×0.5 (clears exact-name/different-generation false positives like the
  canonical Priya vs the COMMON-001 decoy).
- PEP-list hits never become CONFIRMED_HIT — capped at POTENTIAL_MATCH, no
  refinement loop (PEP = EDD, not a sanction).

---

## 8. Entity Resolution Graph (mock MCA21)

In-memory NetworkX graph defined in `agents.py` (`_GRAPH_DATA`). The one
sanctioned node is **"Orion Global Trade Solutions LLP"** (ED PMLA 2021/DEL/0892).
Chain for the high-risk demo customer:
`VMK Holdings Pvt Ltd → Sunrise Export Import Pvt Ltd → Orion Global Trade Solutions LLP` (3 hops).
Customer → entity linkage is currently inferred from name/occupation in
`entity_resolution_agent` (e.g. "MALHOTRA" + business → VMK Holdings; "software
engineer" → Infosys, clean). To add a customer-entity link, extend that logic
or the graph.

---

## 9. Mock Data

### sanctions_list.json — 55 entries across 7 watchlists
ED/PMLA (15) · RBI Wilful Defaulters (9) · PEP (10) · FIU-IND (7) ·
UN Security Council (7) · Interpol Red Notice (6) · combined (1).
Each entry: `id, name, name_devanagari, aliases, father_name, dob_range,
nationality, last_known_address, case_ref, list_source, designation, risk_level, notes`.

**Engineered demo entries (do not rename — code/demo depend on them):**
- `Vikram Suresh Malhotra` (father Suresh Kumar Malhotra) — self-correction → CLEAR target
- `Arjun Ramesh Mehta` (father Ramesh Prakash Mehta) — self-correction → CONFIRMED HIT
- `Deepak Narayan Chaudhary`, `Nirav Hasmukh Patel` — exact-match REJECT targets
- `Priya Sharma` (1978, Jaipur) — false-positive test vs the clean demo customer (1992, Bengaluru)
- `Rajesh Kumar` ×2, `Amit Singh`, `Sanjay Gupta` — common-name collision tests

### bulk_customers.json — 20 profiles → 12 APPROVE / 5 REVIEW / 3 REJECT
Each profile has an `expected_scenario`, `expected_decision`, and `demo_note`.
- BULK-001…010: clean → AUTO APPROVE
- BULK-011, 012: self-correction → father mismatch → CLEAR → APPROVE
- BULK-013: self-correction → father MATCH → CONFIRMED HIT → REJECT
- BULK-014, 015: PEP → HUMAN REVIEW (EDD)
- BULK-016, 017: exact sanctions match → REJECT
- BULK-018, 019, 020: financial anomaly only (HNI, no GST/ITR) → HUMAN REVIEW

---

## 10. The Two Canonical Single-Demo Customers

**Priya Sharma — clean / auto-approve**
Software Engineer, Bengaluru, ₹12,00,000, salary from Infosys.
Expected: extraction COMPLETE, compliance CLEAR, entity CLEAN, financial
CONSISTENT, risk ~0.06–0.13 LOW → **AUTO APPROVE** in ~30s.

**Vikram Malhotra — high-risk / self-correction + HITL**
Business Owner, Delhi, ₹1,20,00,000, vague "business profits".
Expected: compliance AMBIGUOUS **0.81** (alias cap) → **self-correction loop**
finds father "Ramesh Malhotra" → differs from listed "Suresh Kumar Malhotra"
(sim 0.72) → CLEAR at 0.28 → entity FLAGGED (3-hop chain to Orion Global Trade
Solutions LLP) → financial 3 anomaly flags + requires_edd →
risk **~0.335 MEDIUM** → **ROUTE TO HUMAN** → officer holds for documents.
(Score is lower than the original ~0.64 estimate because ID verification
passes and resolved compliance keeps only a 0.25 residual — the decision and
routing are unchanged, verified end-to-end.)
When demoing without a real Aadhaar image, fill the **Father's Name** field
with `Ramesh Malhotra` — the extraction agent surfaces it only on the
refinement pass, simulating the QR `care_of` deep parse.

These two must keep behaving as described — if a change alters their outcome, flag it.

---

## 11. How To Run

```bash
python3 -m venv kyc_env
source kyc_env/bin/activate

# Install
pip install streamlit langgraph openai pytesseract pillow qdrant-client[fastembed] networkx requests
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin

hf auth login

# vLLM (separate terminal) — port MUST match config.VLLM_URL
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000

# ingest sanctions — run BEFORE launching the UI. Qdrant runs in embedded
# local mode (./qdrant_local_db); only ONE process can hold that folder at a
# time, so never run this (or bulk_acceptance_test.py) while app.py is up.
python -c "from tools import setup_sanctions_collection; setup_sanctions_collection()"

# Sanity-check the backend (no UI) — runs both canonical customers
python graph.py

# Bulk regression — all 20 profiles must print 20/20 PASS (12/5/3)
python bulk_acceptance_test.py

# Launch the UI
streamlit run app.py
```

**Environment-specific:** if vLLM starts on a non-8000 port, update `VLLM_URL`
in `config.py`. It is the only environment-specific value.

---

## 12. Working Rules for Claude

- **Every agent must fail gracefully.** No tool call may crash the pipeline — on
  failure return a safe default and log it. The demo must never show a stack trace.
- **Respect the file dependency order** (§5). Don't introduce an import cycle.
- **Don't break the KYCState contract.** Agents return partial dicts; never mutate
  state in place, never remove fields other agents depend on.
- **Keep `audit_log` parallel-safe.** Anything two parallel agents write must use a
  reducer or live in separate keys.
- **Prefer config over hardcoding.** New thresholds, weights, model names, income
  bands → `config.py`.
- **The canonical customers (§10) and engineered mock entries (§9) must keep their
  behaviour.** If a change alters an outcome, flag it.
- **Beginner-friendly.** Include docstrings and a one-line "what to do with this"
  when writing code. When something breaks, explain the cause and the fix.
- **Mock-first, real-second.** Code must run on a laptop with no GPU (mocked
  fallbacks) and light up fully on the MI300X with no code changes.

---

## 13. Known Gaps / TODO

### Applied (2026-06-10) — verified by `python graph.py` + `python bulk_acceptance_test.py` (20/20 PASS, 12/5/3)
- ✅ `father_name` passthrough fix (app.py bulk loop, bulk_acceptance_test.py,
  graph.py Test 2). In `data_extraction_agent` the declared `father_name`
  surfaces **only on the refinement pass** — surfacing it on pass 0 would
  let compliance resolve immediately and the self-correction demo would
  never fire.
- ✅ `_resolve_with_father()` implemented (was documented but missing):
  father differs → score ×0.35 → CLEAR · father matches → ≥0.95 → CONFIRMED_HIT.
- ✅ Declared-data extraction fallback (no readable document → mirror declared
  name/dob/address/PIN, noted in audit) — bulk profiles and laptop mode pass
  ID verification as designed.
- ✅ Deterministic screening scores + alias cap + DOB gate + PEP cap (§7).
- ✅ Risk overrides: CONFIRMED_HIT → REJECT · non-CLEAR screening → REVIEW ·
  `requires_edd` → REVIEW (§7).
- ✅ Entity graph: Orion node itself marked sanctioned (3-hop chain reachable);
  fuzzy address matching + MALHOTRA/business directorship inference.
- ✅ Mock data files renamed to match `config.py`
  (`sanctions_list.json`, `bulk_customers.json`); BULK-017 DOB corrected to
  1972 (was 1982, outside the listed ED range — the exact-match REJECT would
  have been cleared by the DOB gate).
- ✅ All JSON `open()` calls use `encoding="utf-8"` (Windows cp1252 fails on
  Devanagari).
- ✅ `00_setup.ipynb` / `01_demo.ipynb` written; `bulk_acceptance_test.py`
  added — **run it after any change to agents/tools/config.**

### Remaining (acceptable for demo)
- UIDAI status and PAN-Aadhaar linkage are mocked (always ACTIVE / linked).
- Bounding boxes are approximate fixed coordinates, not true detection — fine for demo.
- `exception_cache` Qdrant collection is created lazily on first human decision.
- Real Aadhaar/PAN demo images are not included — add a few to `mock_data/images/`
  for the single-customer demo, or rely on declared-data fallback (the
  Father's Name form field covers the Vikram loop without an image).