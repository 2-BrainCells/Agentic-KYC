"""
app.py — Streamlit demo interface.

Four tabs: KYC Pipeline (submit + run + decide one customer), Bulk Import
(stress-test 20 customers), Review Queue (persistent officer inbox for every
human-routed case), and ROCm Telemetry (live MI300X + vLLM metrics).

    streamlit run app.py
"""

import os
import json
import time
from datetime import datetime, date

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

from graph        import build_graph, complete_case
from state        import create_initial_state, Decision, Routing, CaseStatus
from telemetry    import render_telemetry_tab
from config       import BULK_DOSSIERS_PATH, DOCUMENT_SUBMISSION_EMAIL, REAPPLY_COOLDOWN_MINUTES
from review_queue import (
    enqueue_case, list_cases, get_case, close_case, clear_queue, counts,
    STATUS_PENDING, STATUS_CLOSED,
)
from applicant_registry import (
    generate_application_number, precheck, record_decision,
)


# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Agentic KYC Platform — AMD MI300X",
    page_icon  = "🔐",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# Custom CSS: decision badges, self-correction banner, AMD badge, audit lines.
st.markdown("""
<style>
  .badge-approve { background:#10B981; color:white; padding:6px 18px;
    border-radius:20px; font-weight:700; font-size:1.1rem; }
  .badge-review  { background:#F59E0B; color:white; padding:6px 18px;
    border-radius:20px; font-weight:700; font-size:1.1rem; }
  .badge-reject  { background:#EF4444; color:white; padding:6px 18px;
    border-radius:20px; font-weight:700; font-size:1.1rem; }
  .refine-banner { background:linear-gradient(90deg,#1e1b4b,#312e81);
    border-left:4px solid #E84040; border-radius:6px;
    padding:12px 16px; margin:12px 0; }
  .amd-badge { background:#1a1a2e; border-left:4px solid #E84040;
    border-radius:6px; padding:10px 16px; margin-bottom:8px; }
  .intro-card { background:#0f172a; border:1px solid #334155;
    border-radius:8px; padding:14px 18px; margin-bottom:16px; }
  .audit-line { font-family:monospace; font-size:0.82rem;
    color:#9CA3AF; margin:2px 0; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources (built once, reused across reruns) ─────────────────────

@st.cache_resource
def get_graph():
    """Build and compile the LangGraph pipeline once at startup."""
    return build_graph()


# ── Session state (persists between reruns) ─────────────────────────────────

def init_state():
    """Seed session_state keys used across tabs on first run."""
    defaults = {
        "case_result":      None,   # final state from app.invoke()
        "aadhaar_img_path": None,   # path to uploaded Aadhaar front image
        "queued_id":        None,   # customer_id if this case went to the queue
        "bulk_results":     [],     # list of bulk import results
        "bulk_queued":      0,      # how many bulk cases were sent to the queue
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helper functions ─────────────────────────────────────────────────────────

def save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to ./uploads and return its path."""
    os.makedirs("./uploads", exist_ok=True)
    path = f"./uploads/{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def draw_aadhaar_overlay(image_path, bounding_boxes, low_confidence_fields,
                         field_confidence):
    """Draw coloured field boxes on the Aadhaar image (green/amber by confidence)."""
    try:
        img  = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for field, box in bounding_boxes.items():
            conf = field_confidence.get(field, 0.0)
            if conf == 0.0:
                continue
            if field in low_confidence_fields:
                color, width = (255, 165, 0), 3
            elif conf >= 0.90:
                color, width = (16, 185, 129), 2
            else:
                color, width = (245, 158, 11), 2
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
        return img
    except Exception:
        return None


def decision_badge(decision: str) -> str:
    """Return an HTML decision badge for the given decision string."""
    cls = {
        Decision.APPROVE:            "badge-approve",
        Decision.REVIEW:             "badge-review",
        Decision.HOLD_FOR_DOCUMENTS: "badge-review",
        Decision.REJECT:             "badge-reject",
    }.get(decision, "badge-review")
    return f'<span class="{cls}">{decision.replace("_", " ")}</span>'


def show_self_correction(audit_log: list) -> None:
    """Show the self-correction banner (and resolved note) if the loop ran."""
    loop_lines  = [l for l in audit_log if "SELF-CORRECTION" in l.upper()
                   or "REFINEMENT REQUEST" in l.upper()]
    clear_lines = [l for l in audit_log
                   if "REFINEMENT" in l.upper() and "CLEAR" in l.upper()]
    if loop_lines:
        st.markdown(f"""
        <div class="refine-banner">
          <strong>🔄 Self-Correction Loop Triggered</strong><br>
          <span style="color:#A5B4FC;font-size:0.88rem">{loop_lines[0]}</span>
        </div>
        """, unsafe_allow_html=True)
    if clear_lines:
        st.success("✅ Autonomously resolved — compliance ambiguity cleared by agent")


def hold_instruction(app_no: str) -> str:
    """Next-step message for an applicant whose case was held for documents."""
    return (
        f"📨 **Next step — submit your documents.** Please email your supporting "
        f"documents (e.g. ITR, GST certificate, bank statements) to "
        f"**{DOCUMENT_SUBMISSION_EMAIL}**, quoting your application number "
        f"**`{app_no or 'N/A'}`** in the subject line. Our compliance team will "
        f"review them and proceed with your case manually."
    )


def build_analysis_report(state: dict, app_no: str) -> str:
    """
    Build a plain-text / Markdown KYC analysis report for an ACCEPTED applicant,
    carrying all declared personal information, the decision, the risk breakdown
    and the screening outcome. Offered to the officer/customer as a download.
    """
    declared = state.get("declared", {}) or {}
    decision = state.get("final_decision") or state.get("decision", "")
    score    = state.get("risk_score", 0.0)
    band     = state.get("risk_band", "?")
    hd       = state.get("human_decision") or {}
    when     = state.get("closed_at") or state.get("received_at") or \
               datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST")

    lines = [
        "=" * 64,
        "        KYC ANALYSIS REPORT — Agentic KYC Platform",
        "=" * 64,
        "",
        f"Application Number : {app_no or 'N/A'}",
        f"Customer ID        : {state.get('customer_id', 'N/A')}",
        f"Generated          : {when}",
        f"Final Decision     : {decision}",
        f"Risk Score / Band  : {score:.3f} ({band})",
        "",
        "-" * 64,
        "APPLICANT DETAILS",
        "-" * 64,
        f"Full Name          : {declared.get('name', '')}",
        f"Date of Birth      : {declared.get('dob', '')}",
        f"Nationality        : {declared.get('nationality', '')}",
        f"Father's/Husband's : {declared.get('father_name', '—')}",
        f"Residential Address: {declared.get('address', '')}",
        f"PIN Code           : {declared.get('pin_code', '')}",
        f"Occupation         : {declared.get('occupation', '')}",
        f"Declared Income    : ₹{float(declared.get('income', 0)):,.0f} p.a.",
        f"Source of Funds    : {declared.get('source_of_funds', '')}",
        f"Account Purpose    : {declared.get('account_purpose', '')}",
        "",
        "-" * 64,
        "DUE-DILIGENCE OUTCOME",
        "-" * 64,
        f"ID Verification    : {'PASSED' if state.get('id_verified') else 'FAILED'}",
        f"Screening Status   : {state.get('screening_status', 'N/A')}",
        f"Self-Correction    : {state.get('refine_count', 0)} loop(s)",
    ]

    network = state.get("network_risk", {}) or {}
    lines.append(
        f"Network / Entity   : "
        f"{'FLAGGED — ' + network.get('link_explanation', '') if network.get('indirect_risk_flag') else 'CLEAN'}"
    )
    fin = state.get("financial_profile", {}) or {}
    lines.append(f"Financial Anomalies: {len(fin.get('anomaly_flags', []))} flag(s)")

    factors = state.get("contributing_factors", []) or []
    if factors:
        lines += ["", "Contributing Factors:"]
        for f in factors:
            lines.append(
                f"  - {f.get('factor', '')} ({f.get('weight', 0)*100:.0f}%): "
                f"contribution {f.get('score_contribution', 0):.3f}"
            )

    if hd.get("officer_id"):
        lines += [
            "",
            "-" * 64,
            "OFFICER REVIEW",
            "-" * 64,
            f"Officer            : {hd.get('officer_id', '')}",
            f"Decision           : {hd.get('decision', '')}",
            f"Rationale          : {hd.get('rationale', '') or '—'}",
            f"Reviewed At        : {hd.get('reviewed_at', '')}",
        ]

    explanation = state.get("explanation", "")
    if explanation and "[LLM unavailable]" not in explanation:
        lines += ["", "-" * 64, "DECISION NARRATIVE", "-" * 64, explanation]

    lines += ["", "=" * 64,
              "This report was generated automatically and contains "
              "confidential personal data.", "=" * 64]
    return "\n".join(lines)


# How long to pause between each agent's checklist appearing during a live run.
# Purely cosmetic — makes the "decisions forming" effect watchable in the demo.
STREAM_STEP_DELAY = 0.6

CHECK_ICON = {"pass": "✅", "fail": "❌", "warn": "⚠️", "info": "•"}


def render_checklists(checklists: list) -> None:
    """
    Render each agent's step-wise checklist as a bordered card with ✓/✗/⚠ lines.
    Shared by the live stream during a run and the static results/queue views.
    """
    if not checklists:
        return
    for cl in checklists:
        with st.container(border=True):
            st.markdown(
                f"**{cl.get('icon','')} {cl.get('agent','')}** "
                f"<span style='color:#9CA3AF'>— {cl.get('summary','')}</span>",
                unsafe_allow_html=True,
            )
            for s in cl.get("steps", []):
                icon   = CHECK_ICON.get(s.get("status", "info"), "•")
                detail = s.get("detail", "")
                st.markdown(
                    f"<div class='audit-line'>{icon} {s.get('label','')}"
                    + (f" — <span style='color:#CBD5E1'>{detail}</span>" if detail else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )


def show_contributing_factors(factors: list) -> None:
    """Render the per-factor risk breakdown as a table."""
    if not factors:
        return
    rows = []
    for f in factors:
        ev = f.get("evidence", "")
        rows.append({
            "Factor":       f.get("factor", ""),
            "Weight":       f"{f.get('weight', 0)*100:.0f}%",
            "Evidence":     (ev[:120] + "...") if len(ev) > 120 else ev,
            "Contribution": f"{f.get('score_contribution', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_case_evidence(result: dict, show_overlay: bool = False) -> None:
    """
    Render the full decision view for one case state — shared by the Pipeline
    results panel and the Review Queue detail panel.
    """
    audit_log = result.get("audit_log", [])
    decision  = result.get("decision",  "REVIEW")
    score     = result.get("risk_score", 0.0)
    band      = result.get("risk_band",  "?")
    refines   = result.get("refine_count", 0)

    st.markdown(f"<div style='margin:8px 0'>{decision_badge(decision)}</div>",
                unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("Risk Score", f"{score:.2f}")
    r2.metric("Risk Band",  band)
    r3.metric("Self-Correction Loops", refines)
    st.progress(min(float(score), 1.0), text=f"Risk: {score:.0%}")

    st.divider()
    show_self_correction(audit_log)

    st.markdown("**Decision Explanation**")
    explanation = result.get("explanation", "")
    if explanation and "[LLM unavailable]" not in explanation:
        st.info(explanation)
    else:
        st.info("vLLM explanation unavailable — connect to MI300X for the full "
                "narrative. (The decision and evidence below are still valid.)")

    # Salary-slip verification, if a slip was processed
    iv = (result.get("financial_profile") or {}).get("income_verification")
    if iv:
        if iv.get("status") == "VERIFIED":
            st.success(f"💸 Income proof: {iv.get('note')}")
        else:
            st.warning(f"💸 Income proof: {iv.get('note')}")

    checklists = result.get("agent_checklists", [])
    if checklists:
        with st.expander("🧠 Agent Decision Checklists (step-by-step)", expanded=True):
            render_checklists(checklists)

    st.markdown("**Contributing Factors**")
    show_contributing_factors(result.get("contributing_factors", []))

    if show_overlay:
        img_path = st.session_state.aadhaar_img_path
        if img_path and os.path.exists(img_path):
            st.markdown("**Document Analysis (Aadhaar front)**")
            overlaid = draw_aadhaar_overlay(
                img_path,
                result.get("bounding_boxes",        {}),
                result.get("low_confidence_fields", []),
                result.get("field_confidence",      {}),
            )
            if overlaid:
                st.image(overlaid,
                         caption="🟢 High confidence  🟡 Medium / Low confidence",
                         width=380)

    with st.expander("📋 Full Audit Trail", expanded=False):
        for line in audit_log:
            st.markdown(f'<p class="audit-line">→ {line}</p>',
                        unsafe_allow_html=True)


# ── Header + intro ───────────────────────────────────────────────────────────

st.markdown("""
<div class="amd-badge">
  <span style="color:#E84040;font-weight:700;font-size:1.05rem">
    AMD Instinct MI300X
  </span>
  <span style="color:#9CA3AF"> &nbsp;|&nbsp; </span>
  <span style="color:#F0F4F8">Agentic KYC Intelligence Platform</span>
  <span style="color:#9CA3AF"> &nbsp;|&nbsp; </span>
  <span style="color:#F0F4F8">vLLM 0.17.1 + ROCm 7.0 &nbsp;|&nbsp; LangGraph + Llama-3-8B</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="intro-card">
  <span style="font-size:1.0rem;color:#F0F4F8;font-weight:600">
    Automated customer due diligence, India-first.
  </span><br>
  <span style="color:#CBD5E1;font-size:0.9rem">
  Submit a customer's <b>Aadhaar card</b> (the primary identity document under
  the RBI KYC Master Direction) and declared details. A team of specialised AI
  agents then runs the case end-to-end:
  &nbsp;<b>① Extract</b> identity fields from the Aadhaar →
  <b>② Verify</b> them against what was declared →
  <b>③ Screen</b> the name against FIU-IND / ED / PMLA / UN watchlists →
  <b>④ Profile</b> corporate links and income →
  <b>⑤ Score</b> the risk and either auto-approve or route to a human officer.
  Clean cases clear in seconds; anything ambiguous is sent to the
  <b>Review Queue</b> for a compliance officer to decide.
  </span>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────

q = counts()
tab_kyc, tab_bulk, tab_queue, tab_telemetry = st.tabs([
    "🔐 KYC Pipeline",
    "📦 Bulk Import Stress Test",
    f"🧑‍⚖️ Review Queue ({q['pending']})",
    "⚡ ROCm Telemetry",
])


# ── Tab 1: KYC pipeline ──────────────────────────────────────────────────────

with tab_kyc:

    col_form, col_results = st.columns([1, 1.4], gap="large")

    # Left: upload + declared-data form
    with col_form:
        st.subheader("Customer Submission")
        st.caption("Fields marked **\\***  are required.")

        st.markdown("**Documents**")
        aadhaar_file = st.file_uploader(
            "Aadhaar Card — Front *", type=["jpg", "jpeg", "png"],
            help="Front face: photo, name, DOB, Aadhaar number. OCR reads these.",
        )
        aadhaar_back_file = st.file_uploader(
            "Aadhaar Card — Back (address / QR side)", type=["jpg", "jpeg", "png"],
            help="Back side carries the full address and the QR 'care_of' "
                 "(father's/husband's name) used by the self-correction loop.",
        )
        salary_slip_file = st.file_uploader(
            "Salary Slip / Income Proof (optional)", type=["jpg", "jpeg", "png"],
            help="If provided, the financial agent verifies declared income "
                 "against the figure on the slip.",
        )

        st.markdown("**Declared Information**")
        c1, c2 = st.columns(2)
        name        = c1.text_input("Full Name *")
        dob_date    = c2.date_input(
            "Date of Birth *  (DD-MM-YYYY)",
            value=None,                       # start empty — officer must pick
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            format="DD-MM-YYYY",              # display as on the Aadhaar card
            help="As printed on the Aadhaar card. Stored internally as an ISO date.",
        )
        address     = st.text_input(
            "Residential Address *",
            help="Required. Used for ID verification (address match) and the "
                 "corporate-network / entity-resolution check.",
        )
        c3, c4      = st.columns(2)
        pin_code    = c3.text_input("PIN Code *")
        nationality = c4.text_input("Nationality", value="Indian")
        father_name = st.text_input(
            "Father's / Husband's Name (optional)",
            help="Used by the self-correction loop to disambiguate watchlist "
                 "matches when the Aadhaar QR cannot be parsed.",
        )
        occupation  = st.selectbox("Occupation", [
            "Software Engineer", "Senior Software Engineer",
            "Doctor", "Lawyer", "Government Employee", "Teacher",
            "Business Owner", "Independent Consultant", "Retired", "Other",
        ])
        income      = st.number_input(
            "Annual Income (₹ INR)", min_value=0,
            value=1_200_000, step=100_000,
            help="Declared annual income in Indian Rupees",
        )
        st.caption(f"Selected band: ₹{income:,.0f} p.a.")

        sof     = st.text_input("Source of Funds")
        purpose = st.text_input("Account Purpose")

        run_btn = st.button("🚀 Run KYC", type="primary", use_container_width=True)

    # Right: results panel
    with col_results:
        st.subheader("KYC Decision")

        if run_btn:
            # date_input returns a date object (or None); store canonical ISO.
            dob = dob_date.isoformat() if dob_date else ""
            missing = [
                label for label, val in [
                    ("Full Name", name), ("Date of Birth", dob),
                    ("Residential Address", address), ("PIN Code", pin_code),
                ] if not str(val).strip()
            ]
            if missing:
                st.error("Please fill in the required field(s): " + ", ".join(missing) + ".")
                st.stop()

            # Re-application guard — has this person (name + DOB) been onboarded
            # or recently rejected? Checked BEFORE any processing.
            pre = precheck(name, dob)
            if pre[0] == "ALREADY_ACCEPTED":
                prior = pre[1] or {}
                st.session_state.case_result = None
                st.info(
                    f"✅ **Applicant already registered and accepted.** "
                    f"{name} (DOB {dob}) was onboarded under application number "
                    f"`{prior.get('application_number', 'N/A')}`. No re-processing needed."
                )
                st.stop()
            elif pre[0] == "REJECTED_COOLDOWN":
                minutes_left = pre[2] if len(pre) > 2 else REAPPLY_COOLDOWN_MINUTES
                st.session_state.case_result = None
                st.error(
                    f"⛔ **A recent application for {name} was rejected.** "
                    f"Please try again after {minutes_left} minute(s)."
                )
                st.stop()

            aadhaar_path     = save_upload(aadhaar_file)     if aadhaar_file     else ""
            aadhaar_back_path = save_upload(aadhaar_back_file) if aadhaar_back_file else ""
            salary_slip_path = save_upload(salary_slip_file) if salary_slip_file else ""

            st.session_state.aadhaar_img_path = aadhaar_path or None

            customer_id    = f"CUST-IN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            application_no = generate_application_number()
            st.caption(f"📄 Application number: `{application_no}`")
            initial = create_initial_state(
                customer_id        = customer_id,
                application_number = application_no,
                name               = name,
                dob                = dob,
                nationality        = nationality,
                address            = address,
                pin_code           = pin_code,
                occupation         = occupation.lower(),
                income             = float(income),
                source_of_funds    = sof,
                account_purpose    = purpose,
                aadhaar_path       = aadhaar_path,
                aadhaar_back_path  = aadhaar_back_path,
                salary_slip_path   = salary_slip_path,
                received_at        = datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
            )
            if father_name:
                initial["declared"]["father_name"] = father_name

            st.markdown("**🧠 Agents working — decisions forming live**")
            checklist_ph = st.empty()
            result = initial
            seen   = 0
            with st.spinner("🔍 Agents processing — Extraction → ID Verify → "
                            "Compliance → Risk..."):
                # Stream state snapshots so each agent's checklist appears as it
                # finishes (stream_mode='values' yields the full state per step).
                for snapshot in get_graph().stream(initial, stream_mode="values"):
                    result = snapshot
                    checklists = snapshot.get("agent_checklists", [])
                    if len(checklists) > seen:
                        with checklist_ph.container():
                            render_checklists(checklists)
                        seen = len(checklists)
                        time.sleep(STREAM_STEP_DELAY)
            # Clear the live view — the persistent copy is shown in the results.
            checklist_ph.empty()

            st.session_state.case_result = result

            # Route anything that needs a human into the persistent inbox.
            # Terminal (auto) decisions are recorded in the applicant registry
            # now; human-routed cases are recorded when the officer decides.
            routing = result.get("routing")
            if routing == Routing.ROUTE_TO_HUMAN:
                st.session_state.queued_id = enqueue_case(result)
            else:
                st.session_state.queued_id = None
                if routing in (Routing.AUTO_APPROVE, Routing.AUTO_REJECT):
                    record_decision(
                        application_number = result.get("application_number", ""),
                        customer_id        = result.get("customer_id", ""),
                        name               = (result.get("declared") or {}).get("name", ""),
                        dob                = (result.get("declared") or {}).get("dob", ""),
                        decision           = result.get("decision", ""),
                        risk_score         = result.get("risk_score", 0.0),
                        payload            = result.get("declared") or {},
                    )

        # Render results if available
        result = st.session_state.case_result
        if result:
            # If this case was routed to a human and an officer has since decided
            # it, reflect that final decision here instead of the stale banner.
            queued_id  = st.session_state.get("queued_id")
            closed_rec = None
            if queued_id:
                rec = get_case(queued_id)
                if rec and rec.get("status") == STATUS_CLOSED:
                    closed_rec = rec
                    result = dict(rec.get("state", result))
                    if result.get("final_decision"):
                        result["decision"] = result["final_decision"]

            render_case_evidence(result, show_overlay=True)

            st.divider()
            if closed_rec:
                hd     = result.get("human_decision") or {}
                final  = result.get("final_decision", result.get("decision", ""))
                cname  = (result.get("declared") or {}).get("name", "the applicant")
                when   = result.get("closed_at") or hd.get("reviewed_at", "")
                st.markdown(
                    f"<div style='margin:6px 0'>Final decision: "
                    f"{decision_badge(final)}</div>", unsafe_allow_html=True,
                )
                st.success(
                    f"🧑‍⚖️ **Review complete for {cname}** — officer "
                    f"`{hd.get('officer_id','') or 'unknown'}` recorded a "
                    f"**{final.replace('_',' ')}** decision and the case "
                    f"(`{queued_id}`) is now closed"
                    + (f" · {when}" if when else "") + "."
                )
                if hd.get("rationale"):
                    st.caption(f"📝 Officer rationale: {hd['rationale']}")

                app_no = result.get("application_number", "")
                if final == Decision.HOLD_FOR_DOCUMENTS:
                    st.info(hold_instruction(app_no))
                elif final == Decision.APPROVE:
                    st.download_button(
                        "📥 Download analysis report",
                        data      = build_analysis_report(result, app_no),
                        file_name = f"KYC_Report_{app_no or queued_id}.md",
                        mime      = "text/markdown",
                        key       = f"rep_closed_{queued_id}",
                    )
            elif (result.get("decision") == Decision.REJECT
                  and result.get("routing") == Routing.AUTO_REJECT):
                st.error(
                    "⛔ **Application rejected — identity could not be verified.** "
                    "The submitted Aadhaar does not match the declared details, so "
                    "we cannot establish the applicant's identity. Please re-apply "
                    "with a document whose name, date of birth, PIN and address "
                    "match the information entered."
                )
                for f in (result.get("verification_details") or {}).get("authenticity_flags", []):
                    st.caption(f"• {f}")
            elif result.get("routing") == Routing.ROUTE_TO_HUMAN:
                st.warning(
                    "⚠ **Human review required** — this case was added to the "
                    f"**Review Queue** tab as `{st.session_state.queued_id}`. "
                    "A compliance officer will approve, reject, or hold it there."
                )
            else:
                st.success("✅ **Auto-approved** — no human review needed.")
                app_no = result.get("application_number", "")
                st.download_button(
                    "📥 Download analysis report",
                    data      = build_analysis_report(result, app_no),
                    file_name = f"KYC_Report_{app_no or result.get('customer_id','case')}.md",
                    mime      = "text/markdown",
                    key       = f"rep_auto_{result.get('customer_id','')}",
                )
        else:
            st.info("Upload the Aadhaar front, fill in the declared data, and "
                    "click **Run KYC** to begin.")


# ── Tab 2: bulk import stress test ──────────────────────────────────────────

with tab_bulk:
    st.subheader("📦 Bulk Import Stress Test")
    st.markdown(
        "Simulates 20 simultaneous customer onboarding requests. "
        "Watch the pipeline auto-approve clean profiles and route "
        "high-risk cases to the human Review Queue in real time."
    )

    if not os.path.exists(BULK_DOSSIERS_PATH):
        st.warning(f"Mock data not found at `{BULK_DOSSIERS_PATH}`.")
    else:
        with open(BULK_DOSSIERS_PATH, encoding="utf-8") as f:
            customers = json.load(f)

        st.info(f"**{len(customers)} customer profiles loaded** — "
                f"ready to process on AMD MI300X")

        if st.button("🚀 Import All Customers", type="primary", use_container_width=True):
            st.session_state.bulk_results = []
            st.session_state.bulk_queued  = 0
            app_graph = get_graph()

            table_placeholder   = st.empty()
            summary_placeholder = st.empty()

            approved = review = rejected = processed = queued = 0

            for customer in customers:
                try:
                    application_no = generate_application_number()
                    initial = create_initial_state(
                        customer_id        = customer.get("customer_id", f"BULK-{processed:03d}"),
                        application_number = application_no,
                        name               = customer.get("name", ""),
                        dob                = customer.get("dob", ""),
                        nationality        = customer.get("nationality", "Indian"),
                        address            = customer.get("address", ""),
                        pin_code           = customer.get("pin_code", ""),
                        occupation         = customer.get("occupation", "other"),
                        income             = float(customer.get("income", 0)),
                        source_of_funds    = customer.get("source_of_funds", ""),
                        account_purpose    = customer.get("account_purpose", ""),
                        aadhaar_path       = customer.get("aadhaar_path", ""),
                        received_at        = datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
                    )
                    # Father's-name passthrough — lets the self-correction loop
                    # resolve bulk profiles that have no Aadhaar image.
                    if customer.get("father_name"):
                        initial["declared"]["father_name"] = customer["father_name"]

                    final   = app_graph.invoke(initial)
                    dec     = final.get("decision",  "REVIEW")
                    score   = final.get("risk_score", 0.0)
                    refines = final.get("refine_count", 0)

                    if dec == Decision.APPROVE:
                        approved += 1
                        status_icon = "✅ Auto-Approved"
                    elif dec == Decision.REJECT:
                        rejected += 1
                        status_icon = "⛔ Rejected"
                    else:
                        review += 1
                        status_icon = "⚠️ Human Review"

                    # Send every human-routed case to the persistent queue.
                    if final.get("routing") == Routing.ROUTE_TO_HUMAN:
                        if enqueue_case(final):
                            queued += 1

                    st.session_state.bulk_results.append({
                        "#":            processed + 1,
                        "App No.":      application_no,
                        "Name":         customer.get("name", ""),
                        "Occupation":   customer.get("occupation", ""),
                        "Income ₹":     f"₹{float(customer.get('income',0)):,.0f}",
                        "Score":        f"{score:.2f}",
                        "Self-Correct": "🔄 Yes" if refines > 0 else "—",
                        "Decision":     status_icon,
                    })

                except Exception as e:
                    st.session_state.bulk_results.append({
                        "#":            processed + 1,
                        "App No.":      locals().get("application_no", ""),
                        "Name":         customer.get("name", ""),
                        "Occupation":   "",
                        "Income ₹":     "",
                        "Score":        "—",
                        "Self-Correct": "—",
                        "Decision":     f"❌ Error: {str(e)[:40]}",
                    })

                processed += 1

                df = pd.DataFrame(st.session_state.bulk_results)
                table_placeholder.dataframe(df, use_container_width=True, hide_index=True)

                with summary_placeholder.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Processed",       processed)
                    m2.metric("✅ Auto-Approved", approved)
                    m3.metric("⚠️ Human Review",  review)
                    m4.metric("⛔ Rejected",      rejected)
                    m5.metric("Progress",        f"{processed}/{len(customers)}")

            st.session_state.bulk_queued = queued
            st.balloons()
            st.success(
                f"**Bulk import complete** — {approved} auto-approved · "
                f"{review} routed to human review · {rejected} rejected · "
                f"processed {processed} cases on AMD MI300X."
            )
            st.info(f"🧑‍⚖️ {queued} case(s) added to the **Review Queue** tab "
                    f"for officer action.")

        elif st.session_state.bulk_results:
            st.dataframe(
                pd.DataFrame(st.session_state.bulk_results),
                use_container_width=True, hide_index=True,
            )


# ── Tab 3: review queue (persistent officer inbox) ──────────────────────────

with tab_queue:
    st.subheader("🧑‍⚖️ Compliance Review Queue")
    st.caption("Every case the pipeline routes to a human — from a single run "
               "or the bulk import — lands here. Persisted to disk, so the "
               "inbox survives restarts.")

    # Post-decision flash (survives the st.rerun that refreshes the pending list).
    flash = st.session_state.pop("queue_flash", None)
    if flash:
        if flash["kind"] == "hold":
            st.info(hold_instruction(flash.get("app_no", "")))
        elif flash["kind"] == "approve":
            st.success(f"✅ **{flash.get('name','Applicant')} approved** and recorded "
                       f"in the applicant registry.")
            if flash.get("report"):
                st.download_button(
                    "📥 Download analysis report",
                    data      = flash["report"],
                    file_name = f"KYC_Report_{flash.get('app_no','case')}.md",
                    mime      = "text/markdown",
                    key       = "rep_queue_flash",
                )
        elif flash["kind"] == "reject":
            st.error(f"⛔ **{flash.get('name','Applicant')} rejected** — recorded in the "
                     f"registry; re-application is blocked for "
                     f"{REAPPLY_COOLDOWN_MINUTES} minutes.")

    pending = list_cases(STATUS_PENDING)
    closed  = list_cases(STATUS_CLOSED)

    m1, m2, m3 = st.columns(3)
    m1.metric("⏳ Awaiting Review", len(pending))
    m2.metric("✅ Decided",          len(closed))
    m3.metric("📂 Total",            len(pending) + len(closed))

    if not pending:
        st.info("No cases awaiting review. Cases routed to human review will "
                "appear here automatically.")
    else:
        st.markdown(f"**{len(pending)} case(s) awaiting review** — one tab each.")
        # One tab per pending case; clicking a tab shows its full detail + the
        # officer decision form.
        tab_labels = [
            f"{r['summary'].get('name','?')} · "
            f"{r['summary'].get('decision','')} · "
            f"risk {float(r['summary'].get('risk_score',0)):.2f}"
            for r in pending
        ]
        case_tabs = st.tabs(tab_labels)

        def _finalize(cid, state, officer_id, decision, rationale, override):
            """Close one case: write officer decision, persist, record, flash."""
            if not officer_id.strip():
                st.warning("Enter an Officer ID before recording a decision.")
                return
            closed_state = complete_case(state, officer_id, decision, rationale, override)
            close_case(cid, closed_state)
            declared = closed_state.get("declared") or {}
            app_no   = closed_state.get("application_number", "")
            record_decision(
                application_number = app_no,
                customer_id        = cid,
                name               = declared.get("name", ""),
                dob                = declared.get("dob", ""),
                decision           = decision,
                risk_score         = closed_state.get("risk_score", 0.0),
                payload            = declared,
            )
            if decision == Decision.HOLD_FOR_DOCUMENTS:
                st.session_state.queue_flash = {"kind": "hold", "app_no": app_no}
            elif decision == Decision.APPROVE:
                st.session_state.queue_flash = {
                    "kind": "approve", "app_no": app_no,
                    "name": declared.get("name", ""),
                    "report": build_analysis_report(closed_state, app_no),
                }
            else:
                st.session_state.queue_flash = {
                    "kind": "reject", "app_no": app_no,
                    "name": declared.get("name", ""),
                }
            st.rerun()

        for case_tab, r in zip(case_tabs, pending):
            with case_tab:
                cid   = r["customer_id"]
                rec   = get_case(cid)
                state = rec["state"]

                render_case_evidence(state, show_overlay=False)

                st.divider()
                st.markdown("#### Officer Decision")
                app_no = state.get("application_number", "")
                st.caption(f"📄 Application: `{app_no or 'N/A'}` · Case: `{cid}`")
                officer_id = st.text_input("Officer ID *", value="OFFICER-KYC-001",
                                           key=f"off_{cid}")
                rationale  = st.text_area("Decision Rationale", height=90,
                                          key=f"rat_{cid}")
                sys_decision = state.get("decision", Decision.REVIEW)

                b1, b2, b3 = st.columns(3)
                if b1.button("✅ Approve", use_container_width=True, key=f"ap_{cid}"):
                    _finalize(cid, state, officer_id, Decision.APPROVE, rationale,
                              sys_decision != Decision.APPROVE)
                if b2.button("🚫 Reject", use_container_width=True, key=f"rj_{cid}"):
                    _finalize(cid, state, officer_id, Decision.REJECT, rationale,
                              sys_decision != Decision.REJECT)
                if b3.button("📁 Hold for Documents", use_container_width=True, key=f"hd_{cid}"):
                    _finalize(cid, state, officer_id, Decision.HOLD_FOR_DOCUMENTS,
                              rationale, False)

    # Decided-cases history
    if closed:
        with st.expander(f"✅ Decided cases ({len(closed)})", expanded=False):
            rows = []
            for r in closed:
                hd = (r.get("state", {}) or {}).get("human_decision", {}) or {}
                rows.append({
                    "Case":     r["customer_id"],
                    "Name":     r["summary"].get("name", ""),
                    "Final":    r.get("final_decision", ""),
                    "Officer":  hd.get("officer_id", ""),
                    "Closed":   r.get("closed_at", ""),
                    "Rationale": (hd.get("rationale", "") or "")[:80],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if pending or closed:
        with st.expander("⚙ Queue admin", expanded=False):
            st.caption("Reset the inbox between demo runs.")
            if st.button("🗑 Clear entire queue"):
                clear_queue()
                st.rerun()


# ── Tab 4: ROCm telemetry ───────────────────────────────────────────────────

with tab_telemetry:
    render_telemetry_tab()
