# Agentic KYC Intelligence Platform — Technical Manual

A developer/operator reference for the agentic KYC platform: what each file does,
what each agent decides, the exact factors that drive every decision, and how the
risk score is computed. For project intent, demo strategy, and the engineered mock
data, see `CLAUDE.md`.

> **India-centric.** Aadhaar is the primary document, PAN secondary, income is in
> INR (₹), and the regulatory frame is FIU-IND, ED/PMLA, RBI KYC Master Direction,
> FEMA, and MCA21.

---

## 1. System at a glance

```
Customer (Aadhaar + declared form, ₹ income)
        │
        ▼
  intake → extract → id_verify → compliance ──ambiguous?──► refine ─┐
                                     │ clear/resolved              │ (loop, max 1)
                                     ▼                             │
                                  fan_out                         │
                              ┌──────┴───────┐                    │
                              ▼              ▼                    │
                     entity_resolution  financial_profiling       │
                              └──────┬───────┘                    │
                                     ▼                            │
                                   risk ──auto──► END             │
                                     │                            │
                                   human ──► hitl_review ──► END  │
                                                  │               │
                                  (UI complete_case → cache)      │
        extract ◄────────────────────────────────────────────────┘
```

The pipeline is a **LangGraph `StateGraph`**. Each node is one agent function.
Agents are **stateless and decoupled** — every agent receives the full `KYCState`
and returns only the keys it changes; LangGraph merges the rest. A single vLLM
server (Llama-3-8B-Instruct) backs the four agents that need an LLM; the rest are
deterministic Python.

---

## 2. File-by-file reference

Dependency order (never violate — it prevents import cycles):
`state.py → config.py → tools.py → agents.py → graph.py → app.py`.
`telemetry.py` and `review_queue.py` are leaf modules imported only by `app.py`.

| File | Responsibility |
|---|---|
| **`state.py`** | Defines `KYCState` (the shared contract) and the constant classes (`CaseStatus`, `ScreeningStatus`, `Decision`, `Routing`, `RiskBand`, `DecisionSource`, `ActivityBand`). `create_initial_state()` builds the starting state from the submitted form. `audit_log` uses an `Annotated[list, add]` reducer so the two parallel agents can both append. |
| **`config.py`** | Every tunable value: vLLM endpoint/model/temperatures, Qdrant paths, screening thresholds, risk weights and bands, INR income bands, occupation→band map, regulatory citation strings, watchlist names, file paths. Values only, no logic. `VLLM_URL` is the only environment-specific setting. |
| **`tools.py`** | External I/O, all wrapped in try/except so a failure returns a safe default and never crashes the pipeline. Three groups: **LLM** (`call_text_llm`, `call_llm_for_json`), **document** (`extract_text_from_image`, `parse_aadhaar_fields`, `compute_field_confidence`, `parse_salary_slip`), **sanctions** (`query_sanctions_db` with the deterministic `_sanctions_match_score`, plus `local_sanctions_scan` and `llm_sanctions_check` fallbacks, and `check_exception_cache`). `setup_sanctions_collection()` ingests the list into Qdrant once. |
| **`agents.py`** | The 10 agent functions plus helpers and the in-memory MCA21 corporate graph (`CORPORATE_GRAPH`). This is where all decision logic lives. |
| **`graph.py`** | Wires agents into the compiled graph (`build_graph()`), defines the two routers (`route_after_compliance`, `route_after_risk`) and the `fan_out_node`, and provides `complete_case()` for the human-decision hand-off. Running it directly executes a two-customer end-to-end sanity check. |
| **`telemetry.py`** | Live AMD telemetry. `get_vram()` (rocm-smi) + `get_vllm_stats()` (vLLM `/metrics`) → `get_snapshot()`; `render_telemetry_tab()` draws the Streamlit dashboard and self-refreshes every 2 s. |
| **`review_queue.py`** | Pure JSON-file storage for the officer inbox (`enqueue_case`, `list_cases`, `get_case`, `close_case`, `clear_queue`, `counts`). Survives app restarts. Imports nothing from the pipeline, so it can't create a cycle. |
| **`app.py`** | Streamlit UI with four tabs: KYC Pipeline, Bulk Import, Review Queue, ROCm Telemetry. Builds the graph once (`@st.cache_resource`), runs cases, renders evidence, and routes human cases into `review_queue`. |
| **`bulk_acceptance_test.py`** | Regression test: runs all 20 bulk profiles and asserts each decision matches `expected_decision`. Target: **20/20 PASS (12 APPROVE / 5 REVIEW / 3 REJECT)**. Run after any change to agents/tools/config. |
| **`mock_data/sanctions_list.json`** | 55 watchlist entries across 7 lists. |
| **`mock_data/bulk_customers.json`** | 20 customer profiles with `expected_decision`. |

---

## 3. The agents — what each one decides

Every agent writes one or more `audit_log` lines. "Factors" below are the exact
inputs that change the agent's output.

### 3.1 `intake_agent` — *no LLM*
Confirms receipt and sets `case_status = RECEIVED`. No decision; it only opens the
audit trail.

### 3.2 `data_extraction_agent` — *LLM*
OCRs the Aadhaar (Tesseract `eng+hin`), then asks Llama-3 to structure the text
into identity fields.

- **Factors:** presence/readability of the Aadhaar front (and back) image;
  `refine_count` (pass 0 vs refinement pass 1); declared form data (fallback).
- **Behaviour:** runs at low temperature on pass 0, higher temperature on the
  refinement pass. If no readable document exists, it **mirrors the declared form**
  so the rest of the pipeline still runs real logic, and notes this in the audit.
- **Critical timing:** the declared `father_name` is surfaced **only on the
  refinement pass** — this simulates the deep QR `care_of` parse and is what gives
  the self-correction loop something to discover. Surfacing it on pass 0 would let
  compliance resolve immediately and the headline loop would never fire.
- **Outputs:** `extracted`, `field_confidence`, `bounding_boxes`,
  `low_confidence_fields`, `extraction_status` (COMPLETE if name + DOB +
  aadhaar_last4 all present, else PARTIAL).

### 3.3 `id_verification_agent` — *no LLM*
Cross-checks extracted Aadhaar fields against declared data.

- **Factors & pass conditions:**
  - **Name** — fuzzy similarity (`_name_similarity`) ≥ **0.80** to pass; < 0.60 raises a flag.
  - **DOB** — must match exactly (after `/`→`-` normalisation).
  - **PIN** — must match when both present (else treated as pass).
  - **UIDAI status** — must be `ACTIVE` (mocked ACTIVE in the demo).
  - **PAN-Aadhaar linkage** — mocked `True`.
- **Decision:** `id_verified = name_match AND dob_match AND uidai_active AND no flags`.
- **Outputs:** `id_verified`, `verification_details` (per-check breakdown + reasons).

### 3.4 `compliance_screening_agent` — *LLM only as last-resort fallback*
The screening brain. Screens the name against the watchlists and decides the
`screening_status`.

- **Step 1 — exception cache:** a prior officer decision for this name+DOB
  short-circuits to `CLEAR`.
- **Step 2 — retrieval:** `query_sanctions_db` searches Qdrant across name variants
  (English, Devanagari, inverted order, declared). Fallback chain:
  **Qdrant → local fuzzy scan → LLM-as-judge.**
- **Step 3 — scoring & adjustment** (deterministic, so the 0.60/0.85 thresholds are
  stable):
  - Base score is a 50/50 blend of token overlap and difflib similarity.
  - **Alias cap 0.81** — an exact match to an *alias* (AKA) can never exceed 0.81,
    so it lands in the AMBIGUOUS band, not a confirmed hit. (This arms the loop.)
  - **DOB gate** — if the customer's birth year is outside the entry's `dob_range`
    ±3 years, the score is **halved** (clears same-name/different-generation false
    positives).
  - **Father's-name resolution** (`_resolve_with_father`, only when a father's name
    is present, i.e. on the refinement pass): father **differs** → score ×0.35
    (usually → CLEAR); father **matches** (similarity ≥ 0.80) → score ≥ 0.95
    (→ CONFIRMED_HIT).
- **Step 4 — status from best adjusted score:**

  | Best adjusted score | `screening_status` |
  |---|---|
  | < 0.60 | `CLEAR` |
  | 0.60 – 0.85 | `AMBIGUOUS` (non-PEP) — arms the loop when no father's name and retries remain |
  | > 0.85 | `CONFIRMED_HIT` (non-PEP) |
  | any PEP-list hit | capped at `POTENTIAL_MATCH` (EDD, never auto-reject, no loop) |

- **Loop trigger:** `status == AMBIGUOUS` **AND** no father's name yet **AND**
  `refine_count < REFINEMENT_MAX_TRIES (1)` → sets `refinement_needed = True`.
- **Outputs:** `compliance_hits`, `screening_status`, `refinement_needed`, `cache_hit`.

### 3.5 `refine_agent` — *no LLM*
Pure control signal for the self-correction loop. Increments `refine_count`, writes
the `refinement_request` (seek `father_name` from the QR `care_of` at temp 0.7), and
sets `current_agent_status = "REFINEMENT_REQUEST_ISSUED"`. The graph then loops back
to `extract`. Capped at **1** retry to prevent infinite loops.

### 3.6 `entity_resolution_agent` — *no LLM* (runs parallel to financial)
Traverses `CORPORATE_GRAPH` (mock MCA21, NetworkX) for an indirect link to a
sanctioned entity within **3 hops**.

- **Factors:** declared address, PIN, extracted PAN, occupation. Entry points are
  matched by exact node, then fuzzy address containment, then a
  `MALHOTRA + business/consultant` directorship inference.
- **The one sanctioned node:** *Orion Global Trade Solutions LLP* (ED PMLA
  2021/DEL/0892). The high-risk chain is
  `address → VMK Holdings → Sunrise Export Import → Orion` (3 hops).
- **Decision:** `indirect_risk_flag = True` if any path ≤ 3 hops reaches a
  sanctioned node.
- **Outputs:** `network_risk` (`linked_entities`, `shortest_path_to_sanctioned`,
  `indirect_risk_flag`, `link_explanation`).

### 3.7 `financial_profiling_agent` — *LLM only if a salary slip is uploaded* (runs parallel to entity)
Assesses income plausibility and source-of-funds quality.

- **Factors & anomaly flags raised:**
  - **Income vs occupation** — HNI-band income for an entry/middle occupation.
  - **Vague source of funds** for upper-middle/HNI income (terms like "business",
    "profits", "various" in a short phrase).
  - **HNI income without GST** for business owner / independent consultant.
  - **Overseas-investment purpose + vague funds** (FEMA-pattern flag).
  - **Salary-slip discrepancy** — if a slip is uploaded and its annualised figure
    differs from declared income by more than ±25% (`SALARY_MATCH_TOLERANCE`).
- **`requires_edd`** = `True` when income is HNI-band **and** there is ≥1 anomaly.
  This forces human review even when the weighted score is tiny (see §4).
- **Outputs:** `financial_profile` (`expected_activity_band`, `plausibility`,
  `anomaly_flags`, `requires_edd`, `income_verification`,
  `financial_risk_contribution`).

### 3.8 `risk_scoring_agent` — *LLM (for the explanation only)*
Aggregates everything, decides, then has Llama-3 write the narrative. See §4 for the
full arithmetic and overrides.

### 3.9 `hitl_review_agent` — *no LLM*
Stub that marks `case_status = IN_REVIEW` so the graph completes cleanly. The real
human decision happens in the UI via `complete_case()`.

### 3.10 `active_learning_cache_agent` — *no LLM*
After the officer decides, embeds the decision into the Qdrant `exception_cache`
(best-effort — failure is non-critical) and closes the case
(`final_decision`, `decision_source = HUMAN`, `closed_at`, `case_status = CLOSED`).

---

## 4. How the risk score is calculated

`risk_scoring_agent` computes a single `risk_score ∈ [0.0, 1.0]` as a **weighted sum
of four per-factor raw scores**, maps it to a band/decision, then applies
**regulatory overrides** that can outrank the arithmetic.

### 4.1 Weights (must sum to 1.0 — `config.py`)

| Factor | Weight |
|---|---|
| ID Verification | 0.30 |
| Compliance Screening | 0.40 |
| Network / Entity Risk | 0.20 |
| Financial Profile | 0.10 |

### 4.2 Per-factor raw scores (each ∈ [0, 1])

- **ID** — `0.00` if `id_verified` else `0.80`.
- **Compliance** — by `screening_status`:

  | Status | Raw |
  |---|---|
  | CLEAR (never flagged) | 0.00 |
  | CLEAR after the loop (`refine_count > 0`) | **0.25** (residual caution) |
  | AMBIGUOUS | 0.60 |
  | POTENTIAL_MATCH (incl. PEP) | 0.75 |
  | CONFIRMED_HIT | 1.00 |

- **Network** — `0.80` if `indirect_risk_flag` else `0.00`.
- **Financial** — `financial_risk_contribution` = `0.15 × anomaly_count`, plus
  `+0.30` if income is HNI-band with any anomaly (capped at 1.0).

### 4.3 Weighted sum → band → decision

```
risk_score = 0.30·ID + 0.40·Compliance + 0.20·Network + 0.10·Financial   (capped at 1.0)
```

| `risk_score` | Band | Decision | Routing |
|---|---|---|---|
| < 0.30 | LOW | APPROVE | AUTO_APPROVE |
| 0.30 – 0.65 | MEDIUM | REVIEW | ROUTE_TO_HUMAN |
| > 0.65 | HIGH | REVIEW | ROUTE_TO_HUMAN |

### 4.4 Regulatory overrides (applied after the arithmetic)

These exist because some outcomes are legally mandated regardless of the weighted
number. Evaluated in order:

1. **`screening_status == CONFIRMED_HIT`** → decision **REJECT**, band HIGH, still
   ROUTE_TO_HUMAN. *(PMLA — cannot onboard a sanctioned individual.)*
2. **Decision would be APPROVE but `screening_status != CLEAR`** (PEP / potential
   match) → at least **REVIEW**, ROUTE_TO_HUMAN. *(Enhanced Due Diligence.)*
3. **Decision would be APPROVE but `financial_profile.requires_edd`** → at least
   **REVIEW**, ROUTE_TO_HUMAN. *(RBI KYC Master Direction — this is how an HNI
   profile with anomalies reaches a human despite Financial's 10% weight.)*

### 4.5 Worked examples (the two canonical customers)

**Priya Sharma (clean).** ID passes (0.00), CLEAR (0.00), no network link (0.00),
no anomalies (0.00) → `risk_score ≈ 0.06–0.13` LOW → **AUTO APPROVE**.

**Vikram Malhotra (high-risk + loop).** Alias hit at 0.81 → AMBIGUOUS → loop finds
father "Ramesh" ≠ listed "Suresh Kumar" → score ×0.35 → CLEAR-after-loop (compliance
raw **0.25**). ID passes (0.00). Network FLAGGED via the 3-hop Orion chain (raw
0.80). Financial: 3 anomalies + HNI → `requires_edd`. Weighted:
`0.40·0.25 + 0.20·0.80 + 0.10·(~0.75) ≈ 0.335` MEDIUM → **ROUTE TO HUMAN**
(override #3 also independently forces review).

---

## 5. The self-correction loop (headline behaviour)

1. Compliance returns `AMBIGUOUS` (mid-band score, no father's name yet) and sets
   `refinement_needed = True`.
2. `route_after_compliance` → `refine_agent` bumps `refine_count` and issues the
   refinement request.
3. Graph loops back to `extract`, which re-parses at higher temperature and now
   surfaces the father's name (QR `care_of`).
4. Compliance runs again; `_resolve_with_father` compares father names:
   - **differ** → score ×0.35 → **CLEAR** (exonerated),
   - **match** → score ≥ 0.95 → **CONFIRMED_HIT** (convicted).
5. Capped at `REFINEMENT_MAX_TRIES = 1`.

The loop **works both ways** — it can clear a false positive *or* confirm a real
hit. That two-directional behaviour is the credibility point.

---

## 6. The shared state contract (`KYCState`)

- Every agent returns a **partial dict**; never mutate state in place, never remove a
  key another agent depends on.
- `audit_log` is the only reduced field (`Annotated[list, add]`) — it is the one key
  the two parallel agents may both write. Everything else is last-write-wins, so the
  parallel branches must write **disjoint** keys (`network_risk` vs
  `financial_profile`), which they do.
- Always use the **constant classes**, never raw string literals — a typo in a
  status string silently breaks routing.

---

## 7. Failure & degradation model

Designed to run identically on a laptop (everything mocked) and on the MI300X (fully
lit) with no code changes:

| Dependency down | Behaviour |
|---|---|
| **vLLM** | `call_text_llm` returns `"[LLM unavailable]"`; extraction falls back to declared data; risk explanation shows a placeholder. Decisions still compute (the scoring math is deterministic). |
| **Qdrant locked/down** | `_make_qdrant_client` falls back to in-memory; `query_sanctions_db` degrades to the on-disk fuzzy scan, then to LLM-as-judge. Screening still works. |
| **Tesseract / no image** | OCR returns `""`; extraction mirrors declared data and notes it in the audit. |
| **rocm-smi absent** | Telemetry shows VRAM 0/192 with an error note; the rest of the app is unaffected. |

No tool failure raises out of an agent — the demo never shows a stack trace.

---
