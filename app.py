"""
app.py — Streamlit Demo Interface
===================================
The face of the platform. Everything the judge sees.

Three tabs:
  1. KYC Pipeline  — upload Aadhaar, run the pipeline, see the decision
  2. Bulk Import   — stress-test with 20 customers simultaneously
  3. ROCm Telemetry — live AMD MI300X GPU + vLLM metrics

Run with:
    streamlit run app.py

Then open the URL shown in the terminal (usually http://localhost:8501)
"""

import os
import json
import time
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from graph      import build_graph, complete_case
from state      import create_initial_state, Decision, Routing, CaseStatus
from telemetry  import render_telemetry_tab
from config     import BULK_DOSSIERS_PATH


# ════════════════════════════════════════════════════════════════
# PAGE SETUP
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Agentic KYC Platform — AMD MI300X",
    page_icon  = "🔐",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  /* Decision badges */
  .badge-approve { background:#10B981; color:white; padding:6px 18px;
    border-radius:20px; font-weight:700; font-size:1.1rem; }
  .badge-review  { background:#F59E0B; color:white; padding:6px 18px;
    border-radius:20px; font-weight:700; font-size:1.1rem; }
  .badge-reject  { background:#EF4444; color:white; padding:6px 18px;
    border-radius:20px; font-weight:700; font-size:1.1rem; }

  /* Self-correction banner */
  .refine-banner { background:linear-gradient(90deg,#1e1b4b,#312e81);
    border-left:4px solid #E84040; border-radius:6px;
    padding:12px 16px; margin:12px 0; }

  /* AMD hardware badge */
  .amd-badge { background:#1a1a2e; border-left:4px solid #E84040;
    border-radius:6px; padding:10px 16px; margin-bottom:16px; }

  /* Audit log — monospace */
  .audit-line { font-family:monospace; font-size:0.82rem;
    color:#9CA3AF; margin:2px 0; }

  /* Agent status dots */
  .agent-dot-done    { color:#10B981; font-size:1.2rem; }
  .agent-dot-flagged { color:#EF4444; font-size:1.2rem; }
  .agent-dot-pending { color:#6B7280; font-size:1.2rem; }
  .agent-dot-running { color:#F59E0B; font-size:1.2rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# CACHED RESOURCES (built once, reused across all reruns)
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def get_graph():
    """Build and compile the LangGraph pipeline once at startup."""
    return build_graph()


# ════════════════════════════════════════════════════════════════
# SESSION STATE  (persists between Streamlit reruns)
# ════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "case_result":       None,   # final state from app.invoke()
        "aadhaar_img_path":  None,   # path to uploaded Aadhaar image
        "case_closed":       False,  # whether HITL has been completed
        "closed_result":     None,   # result of complete_case()
        "bulk_results":      [],     # list of bulk import results
        "bulk_running":      False,  # is bulk import in progress
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to disk and return the path."""
    os.makedirs("./uploads", exist_ok=True)
    path = f"./uploads/{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def draw_aadhaar_overlay(
    image_path:            str,
    bounding_boxes:        dict,
    low_confidence_fields: list,
    field_confidence:      dict,
) -> Image.Image:
    """
    Draws coloured bounding boxes on the Aadhaar card image.
      Green  = high confidence field (≥ 0.90)
      Amber  = medium confidence     (0.75 – 0.90)
      Red    = low confidence / found only on refinement pass
    """
    try:
        img  = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for field, box in bounding_boxes.items():
            conf = field_confidence.get(field, 0.0)
            if conf == 0.0:
                continue                            # field not extracted — skip
            if field in low_confidence_fields:
                color, width = (255, 165, 0), 3    # amber
            elif conf >= 0.90:
                color, width = (16, 185, 129), 2   # green
            else:
                color, width = (245, 158, 11), 2   # yellow

            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

        return img
    except Exception:
        return None


def decision_badge(decision: str) -> str:
    cls = {
        Decision.APPROVE:            "badge-approve",
        Decision.REVIEW:             "badge-review",
        Decision.HOLD_FOR_DOCUMENTS: "badge-review",
        Decision.REJECT:             "badge-reject",
    }.get(decision, "badge-review")
    return f'<span class="{cls}">{decision.replace("_", " ")}</span>'


def agent_status_row(audit_log: list) -> None:
    """
    Reads the audit log and renders a coloured status dot for each agent.
    Green = ran clean. Red = flagged. Grey = did not run.
    """
    agents = {
        "Extraction":       ("extract",    "📄"),
        "ID Verify":        ("ID Verify",  "🪪"),
        "Compliance":       ("Compliance", "🔍"),
        "Self-Correct":     ("SELF-CORR",  "🔄"),
        "Entity":           ("Entity",     "🕸"),
        "Financial":        ("Financial",  "💰"),
        "Risk Score":       ("Risk",       "⚖"),
        "HITL":             ("HITL",       "👤"),
        "Cache":            ("Cache",      "💾"),
    }
    log_text = " ".join(audit_log).upper()
    cols = st.columns(len(agents))
    for i, (label, (keyword, icon)) in enumerate(agents.items()):
        ran     = keyword.upper() in log_text
        flagged = ran and any(
            w in log_text for w in
            ["FAIL", "FLAGGED", "AMBIGUOUS", "INCONSISTENT"]
            if keyword.upper() in log_text
        )
        dot = "🟢" if (ran and not flagged) else ("🔴" if flagged else "⚫")
        cols[i].markdown(
            f"<div style='text-align:center;font-size:0.75rem'>"
            f"{dot}<br>{icon} {label}</div>",
            unsafe_allow_html=True,
        )


def show_self_correction(audit_log: list) -> None:
    """Shows the self-correction banner if the loop triggered."""
    loop_lines = [l for l in audit_log if "SELF-CORRECTION" in l.upper()]
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


def show_contributing_factors(factors: list) -> None:
    """Renders the contributing factors as a clean table."""
    if not factors:
        return
    rows = []
    for f in factors:
        rows.append({
            "Factor":       f.get("factor", ""),
            "Weight":       f"{f.get('weight', 0)*100:.0f}%",
            "Evidence":     f.get("evidence", "")[:120] + "..."
                            if len(f.get("evidence","")) > 120
                            else f.get("evidence",""),
            "Contribution": f"{f.get('score_contribution', 0):.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab_kyc, tab_bulk, tab_telemetry = st.tabs([
    "🔐 KYC Pipeline",
    "📦 Bulk Import Stress Test",
    "⚡ ROCm Telemetry",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — KYC PIPELINE
# ════════════════════════════════════════════════════════════════

with tab_kyc:

    col_form, col_results = st.columns([1, 1.4], gap="large")

    # ── LEFT: Upload + Declared Data Form ─────────────────────
    with col_form:
        st.subheader("Customer Submission")

        aadhaar_file = st.file_uploader(
            "Aadhaar Card (front face)", type=["jpg", "jpeg", "png"],
            help="Upload the Aadhaar card image. OCR will extract all fields.",
        )
        pan_file = st.file_uploader(
            "PAN Card", type=["jpg", "jpeg", "png"],
        )

        st.markdown("**Declared Information**")

        c1, c2 = st.columns(2)
        name        = c1.text_input("Full Name",           placeholder="Priya Sharma")
        dob         = c2.text_input("Date of Birth",       placeholder="YYYY-MM-DD")
        address     = st.text_input("Residential Address", placeholder="B-204, Green Park, Bengaluru")
        c3, c4      = st.columns(2)
        pin_code    = c3.text_input("PIN Code",            placeholder="560034")
        nationality = c4.text_input("Nationality",         value="Indian")
        father_name = st.text_input(
            "Father's / Husband's Name (optional)",
            placeholder="As on Aadhaar C/O field",
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

        sof     = st.text_input(
            "Source of Funds",
            placeholder="Monthly salary from Infosys Limited",
        )
        purpose = st.text_input(
            "Account Purpose",
            placeholder="Savings and mutual fund investments",
        )

        run_btn = st.button(
            "🚀 Run KYC", type="primary", use_container_width=True,
        )

    # ── RIGHT: Results Panel ───────────────────────────────────
    with col_results:
        st.subheader("KYC Decision")

        if run_btn:
            # Validate minimum inputs
            if not name or not dob:
                st.error("Please enter at least the customer's name and date of birth.")
                st.stop()

            # Save uploaded files
            aadhaar_path = save_upload(aadhaar_file) if aadhaar_file else "/uploads/no_image.jpg"
            pan_path     = save_upload(pan_file)     if pan_file     else "/uploads/no_pan.jpg"

            if aadhaar_file:
                st.session_state.aadhaar_img_path = aadhaar_path

            # Build initial state
            customer_id = f"CUST-IN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            initial = create_initial_state(
                customer_id     = customer_id,
                name            = name,
                dob             = dob,
                nationality     = nationality,
                address         = address,
                pin_code        = pin_code,
                occupation      = occupation.lower(),
                income          = float(income),
                source_of_funds = sof,
                account_purpose = purpose,
                aadhaar_path    = aadhaar_path,
                pan_path        = pan_path,
                received_at     = datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
            )
            if father_name:
                initial["declared"]["father_name"] = father_name

            # Run the pipeline
            with st.spinner("🔍 Agents processing — Extraction → ID Verify → Compliance → Risk..."):
                app_graph = get_graph()
                result    = app_graph.invoke(initial)

            st.session_state.case_result = result
            st.session_state.case_closed = False

        # ── Render results if available ────────────────────────
        result = st.session_state.case_result
        if result:
            audit_log = result.get("audit_log", [])
            decision  = result.get("decision",  "REVIEW")
            score     = result.get("risk_score", 0.0)
            band      = result.get("risk_band",  "?")
            routing   = result.get("routing",    Routing.ROUTE_TO_HUMAN)
            refines   = result.get("refine_count", 0)

            # Decision badge + score
            st.markdown(
                f"<div style='margin:8px 0'>{decision_badge(decision)}</div>",
                unsafe_allow_html=True,
            )
            r1, r2, r3 = st.columns(3)
            r1.metric("Risk Score", f"{score:.2f}")
            r2.metric("Risk Band",  band)
            r3.metric("Self-Correction Loops", refines)

            score_color = (
                "normal" if score < 0.30 else
                "inverse" if score > 0.65 else "off"
            )
            st.progress(float(score), text=f"Risk: {score:.0%}")

            st.divider()

            # Self-correction banner
            show_self_correction(audit_log)

            # Agent status row
            st.markdown("**Agent Status**")
            agent_status_row(audit_log)

            st.divider()

            # Explanation
            st.markdown("**Decision Explanation**")
            explanation = result.get("explanation", "")
            if explanation and "[LLM unavailable]" not in explanation:
                st.info(explanation)
            else:
                st.info(
                    "vLLM explanation unavailable — "
                    "connect to MI300X for full narrative output."
                )

            # Contributing factors
            st.markdown("**Contributing Factors**")
            show_contributing_factors(result.get("contributing_factors", []))

            # Aadhaar overlay
            img_path = st.session_state.aadhaar_img_path
            if img_path and os.path.exists(img_path):
                st.markdown("**Document Analysis**")
                overlaid = draw_aadhaar_overlay(
                    img_path,
                    result.get("bounding_boxes",        {}),
                    result.get("low_confidence_fields", []),
                    result.get("field_confidence",      {}),
                )
                if overlaid:
                    st.image(overlaid, caption="🟢 High confidence  🟡 Medium  🟠 Low confidence", width=380)

            # Audit trail
            with st.expander("📋 Full Audit Trail", expanded=False):
                for line in audit_log:
                    st.markdown(f'<p class="audit-line">→ {line}</p>', unsafe_allow_html=True)

            st.divider()

            # ── HITL Review Panel ──────────────────────────────
            if routing == Routing.ROUTE_TO_HUMAN and not st.session_state.case_closed:
                st.markdown("""
                <div style='background:#1c1917;border:1px solid #F59E0B;
                     border-radius:8px;padding:16px;margin:8px 0'>
                <strong style='color:#F59E0B'>⚠ Human Review Required</strong><br>
                <span style='color:#D1D5DB;font-size:0.9rem'>
                This case has been routed to the compliance officer queue.
                Review the evidence above and make your decision below.
                </span></div>
                """, unsafe_allow_html=True)

                officer_id = st.text_input("Officer ID", value="OFFICER-KYC-001")
                rationale  = st.text_area(
                    "Decision Rationale",
                    placeholder="Enter your reasoning for the decision...",
                    height=100,
                )
                h1, h2, h3 = st.columns(3)

                if h1.button("✅ Approve", use_container_width=True):
                    closed = complete_case(
                        result, officer_id,
                        Decision.APPROVE, rationale,
                        override=(decision != Decision.APPROVE),
                    )
                    st.session_state.closed_result = closed
                    st.session_state.case_closed   = True
                    st.rerun()

                if h2.button("🚫 Reject", use_container_width=True):
                    closed = complete_case(
                        result, officer_id,
                        Decision.REJECT, rationale,
                        override=(decision != Decision.REJECT),
                    )
                    st.session_state.closed_result = closed
                    st.session_state.case_closed   = True
                    st.rerun()

                if h3.button("📁 Hold for Documents", use_container_width=True):
                    closed = complete_case(
                        result, officer_id,
                        Decision.HOLD_FOR_DOCUMENTS, rationale, False,
                    )
                    st.session_state.closed_result = closed
                    st.session_state.case_closed   = True
                    st.rerun()

            # Closed case confirmation
            if st.session_state.case_closed and st.session_state.closed_result:
                cr = st.session_state.closed_result
                st.success(
                    f"✅ Case closed — **{cr.get('final_decision')}** "
                    f"by {cr.get('human_decision', {}).get('officer_id', '?')} "
                    f"at {cr.get('closed_at', '?')} | "
                    f"Decision cached: {cr.get('cache_update', {}).get('status', '?')}"
                )

        else:
            st.info("Upload an Aadhaar card, fill in the declared data, and click **Run KYC** to begin.")


# ════════════════════════════════════════════════════════════════
# TAB 2 — BULK IMPORT STRESS TEST
# ════════════════════════════════════════════════════════════════

with tab_bulk:
    st.subheader("📦 Bulk Import Stress Test")
    st.markdown(
        "Simulates 20 simultaneous customer onboarding requests. "
        "Watch the pipeline auto-approve clean profiles and route "
        "high-risk cases to the human review queue in real time."
    )

    # Check mock data exists
    if not os.path.exists(BULK_DOSSIERS_PATH):
        st.warning(
            f"Mock data not found at `{BULK_DOSSIERS_PATH}`. "
            "Ask Claude to generate `bulk_customers.json` and place it in `./mock_data/`."
        )
    else:
        with open(BULK_DOSSIERS_PATH, encoding="utf-8") as f:
            customers = json.load(f)

        st.info(f"**{len(customers)} customer profiles loaded** — "
                f"ready to process on AMD MI300X")

        if st.button("🚀 Import All Customers", type="primary", use_container_width=True):
            st.session_state.bulk_results  = []
            st.session_state.bulk_running  = True
            app_graph = get_graph()

            # Live results table
            table_placeholder = st.empty()
            summary_placeholder = st.empty()

            approved  = 0
            review    = 0
            rejected  = 0
            processed = 0

            for customer in customers:
                try:
                    initial = create_initial_state(
                        customer_id     = customer.get("customer_id", f"BULK-{processed:03d}"),
                        name            = customer.get("name", ""),
                        dob             = customer.get("dob", ""),
                        nationality     = customer.get("nationality", "Indian"),
                        address         = customer.get("address", ""),
                        pin_code        = customer.get("pin_code", ""),
                        occupation      = customer.get("occupation", "other"),
                        income          = float(customer.get("income", 0)),
                        source_of_funds = customer.get("source_of_funds", ""),
                        account_purpose = customer.get("account_purpose", ""),
                        aadhaar_path    = customer.get("aadhaar_path", "/uploads/mock.jpg"),
                        pan_path        = customer.get("pan_path",    "/uploads/mock_pan.jpg"),
                        received_at     = datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
                    )
                    # Father's name passthrough — lets the self-correction
                    # loop resolve bulk profiles that have no Aadhaar image
                    # (the refinement pass picks it up as the QR 'care_of').
                    if customer.get("father_name"):
                        initial["declared"]["father_name"] = customer["father_name"]

                    final = app_graph.invoke(initial)

                    dec   = final.get("decision",  "REVIEW")
                    score = final.get("risk_score", 0.0)
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

                    st.session_state.bulk_results.append({
                        "#":          processed + 1,
                        "Name":       customer.get("name", ""),
                        "Occupation": customer.get("occupation", ""),
                        "Income ₹":   f"₹{float(customer.get('income',0)):,.0f}",
                        "Score":      f"{score:.2f}",
                        "Self-Correct": "🔄 Yes" if refines > 0 else "—",
                        "Decision":   status_icon,
                    })

                except Exception as e:
                    st.session_state.bulk_results.append({
                        "#":          processed + 1,
                        "Name":       customer.get("name",""),
                        "Occupation": "",
                        "Income ₹":   "",
                        "Score":      "—",
                        "Self-Correct": "—",
                        "Decision":   f"❌ Error: {str(e)[:40]}",
                    })

                processed += 1

                # Update live table
                df = pd.DataFrame(st.session_state.bulk_results)
                table_placeholder.dataframe(df, use_container_width=True, hide_index=True)

                # Update summary metrics
                with summary_placeholder.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Processed",      processed)
                    m2.metric("✅ Auto-Approved", approved)
                    m3.metric("⚠️ Human Review", review)
                    m4.metric("⛔ Rejected",     rejected)
                    m5.metric("Progress",       f"{processed}/{len(customers)}")

            # Final summary
            st.balloons()
            st.success(
                f"**Bulk import complete** — "
                f"{approved} auto-approved · "
                f"{review} routed to human review · "
                f"{rejected} rejected · "
                f"processed {processed} cases on AMD MI300X"
            )
            st.session_state.bulk_running = False

        # Show previous results if available
        elif st.session_state.bulk_results:
            st.dataframe(
                pd.DataFrame(st.session_state.bulk_results),
                use_container_width=True, hide_index=True,
            )
            approved = sum(
                1 for r in st.session_state.bulk_results
                if "Approved" in r.get("Decision","")
            )
            review   = sum(
                1 for r in st.session_state.bulk_results
                if "Review" in r.get("Decision","")
            )
            rejected = sum(
                1 for r in st.session_state.bulk_results
                if "Rejected" in r.get("Decision","")
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("✅ Auto-Approved", approved)
            c2.metric("⚠️ Human Review", review)
            c3.metric("⛔ Rejected",     rejected)
            c4.metric("Total Processed", len(st.session_state.bulk_results))


# ════════════════════════════════════════════════════════════════
# TAB 3 — ROCM TELEMETRY
# ════════════════════════════════════════════════════════════════

with tab_telemetry:
    render_telemetry_tab()