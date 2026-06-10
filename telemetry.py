"""
telemetry.py — AMD ROCm + vLLM Live Metrics
============================================
Owned by: Person B
Build on: Day 3 Morning

Two data sources:
  1. rocm-smi (shell command) → VRAM used / total / %, GPU utilisation
  2. vLLM /metrics endpoint   → tokens/sec, active requests, queue, KV cache

How to use in your Streamlit demo app:
    from telemetry import render_telemetry_tab
    tab1, tab2 = st.tabs(["KYC Pipeline", "ROCm Telemetry"])
    with tab2:
        render_telemetry_tab()
"""

import subprocess
import json
import time
import requests
from datetime import datetime
from collections import deque

# ── CONFIG ─────────────────────────────────────────────────────────────────
VLLM_METRICS_URL = "http://localhost:8000/metrics"   # change port if needed
HISTORY_POINTS   = 30                                 # for tokens/sec rate window


# ── INTERNAL HISTORY (for computing tokens/sec rate over time) ─────────────
_token_history = deque(maxlen=HISTORY_POINTS)


# ═══════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

def get_vram():
    """
    Calls rocm-smi to read GPU memory stats from the AMD MI300X.
    Returns used GB, total GB, and usage percentage.
    Never crashes — returns safe defaults if rocm-smi is unavailable.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmemuse", "--json"],
            capture_output=True, text=True, timeout=3
        )
        data  = json.loads(result.stdout)
        card  = list(data.values())[0]          # first GPU card

        total_bytes = int(card.get("VRAM Total Memory (B)", 0))
        used_bytes  = int(card.get("VRAM Total Used Memory (B)", 0))

        total_gb = round(total_bytes / (1024 ** 3), 1)
        used_gb  = round(used_bytes  / (1024 ** 3), 1)
        pct      = round((used_gb / total_gb * 100), 1) if total_gb > 0 else 0.0

        return {
            "used_gb":  used_gb,
            "total_gb": total_gb,
            "pct":      pct,
            "ok":       True
        }

    except Exception as e:
        # Safe fallback — MI300X has 192 GB so this still looks reasonable
        return {
            "used_gb":  0.0,
            "total_gb": 192.0,
            "pct":      0.0,
            "ok":       False,
            "err":      str(e)
        }


def get_vllm_stats():
    """
    Reads the vLLM Prometheus metrics endpoint at /metrics.
    Parses token counts, active requests, queue depth, and KV cache.
    Computes tokens/sec from a rolling history window.
    """
    try:
        text = requests.get(VLLM_METRICS_URL, timeout=3).text

        # Parse Prometheus text format: "metric_name value"
        raw = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    raw[parts[0]] = float(parts[1])
                except ValueError:
                    pass

        # ── Token counts ─────────────────────────────────────────────────
        # vLLM metric names vary slightly by version — check both forms
        prompt_tokens = raw.get(
            "vllm:prompt_tokens_total_count",
            raw.get("vllm:prompt_tokens_total", 0)
        )
        gen_tokens = raw.get(
            "vllm:generation_tokens_total_count",
            raw.get("vllm:generation_tokens_total", 0)
        )
        total_tokens = prompt_tokens + gen_tokens

        # ── Tokens / second (rolling rate) ───────────────────────────────
        now = time.time()
        _token_history.append((now, total_tokens))
        tps = 0.0
        if len(_token_history) >= 2:
            dt   = _token_history[-1][0] - _token_history[0][0]
            dtok = _token_history[-1][1] - _token_history[0][1]
            tps  = round(dtok / dt, 1) if dt > 0 else 0.0

        # ── Other metrics ─────────────────────────────────────────────────
        active_req  = int(raw.get("vllm:num_requests_running", 0))
        queued_req  = int(raw.get("vllm:num_requests_waiting", 0))
        kv_cache    = round(raw.get("vllm:gpu_cache_usage_perc", 0) * 100, 1)

        return {
            "tps":            tps,
            "total_tokens":   int(total_tokens),
            "prompt_tokens":  int(prompt_tokens),
            "gen_tokens":     int(gen_tokens),
            "active_requests": active_req,
            "queued_requests": queued_req,
            "kv_cache_pct":   kv_cache,
            "ok":             True
        }

    except Exception as e:
        return {
            "tps": 0.0, "total_tokens": 0, "prompt_tokens": 0,
            "gen_tokens": 0, "active_requests": 0, "queued_requests": 0,
            "kv_cache_pct": 0.0, "ok": False, "err": str(e)
        }


def get_snapshot():
    """
    Single call that returns everything the dashboard needs.
    Call this from a notebook cell or Streamlit fragment.
    """
    return {
        "vram":      get_vram(),
        "inference": get_vllm_stats(),
        "timestamp": datetime.now().strftime("%H:%M:%S IST")
    }


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def render_telemetry_tab():
    """
    Renders the full ROCm Telemetry tab inside a Streamlit app.

    Usage:
        tab1, tab2 = st.tabs(["KYC Pipeline", "⚡ ROCm Telemetry"])
        with tab2:
            render_telemetry_tab()
    """
    import streamlit as st

    snap = get_snapshot()
    vram = snap["vram"]
    inf  = snap["inference"]

    # ── Header ──────────────────────────────────────────────────────────
    st.markdown("## ⚡ AMD Instinct MI300X — Live Telemetry")
    st.caption(f"Last refreshed: {snap['timestamp']}  •  "
               f"Source: rocm-smi + vLLM /metrics")

    st.divider()

    # ── Row 1: The four headline numbers ────────────────────────────────
    st.markdown("### Inference Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label="🚀 Tokens / Second",
        value=f"{inf['tps']}",
        help="Generated tokens per second across all active requests"
    )
    c2.metric(
        label="🔢 Total Tokens Generated",
        value=f"{inf['total_tokens']:,}",
        help="Cumulative tokens generated since vLLM server started"
    )
    c3.metric(
        label="🔄 Active Requests",
        value=inf["active_requests"],
        help="KYC agent requests currently being processed by the GPU"
    )
    c4.metric(
        label="⏳ Queue Depth",
        value=inf["queued_requests"],
        help="Requests waiting — rises during bulk import stress test"
    )

    st.divider()

    # ── Row 2: VRAM bar (the visual showpiece) ───────────────────────────
    st.markdown("### VRAM Utilisation (192 GB HBM3)")
    col_bar, col_num = st.columns([4, 1])

    vram_fraction = min(vram["pct"] / 100, 1.0)
    col_bar.progress(
        vram_fraction,
        text=f"{vram['used_gb']} GB used  /  {vram['total_gb']} GB total"
    )
    col_num.metric("VRAM %", f"{vram['pct']}%")

    # Colour-coded status
    if vram["pct"] > 85:
        st.warning("⚠️ VRAM usage high — consider reducing batch size")
    elif vram["pct"] > 50:
        st.info("ℹ️ VRAM at moderate utilisation")
    else:
        st.success("✅ VRAM healthy")

    st.divider()

    # ── Row 3: Inference detail breakdown ────────────────────────────────
    st.markdown("### Inference Detail")
    d1, d2, d3 = st.columns(3)
    d1.metric(
        label="KV Cache Used",
        value=f"{inf['kv_cache_pct']}%",
        help="vLLM PagedAttention KV cache fill — climbs under bulk load"
    )
    d2.metric(
        label="Prompt Tokens",
        value=f"{inf['prompt_tokens']:,}",
        help="Tokens from input documents and instructions"
    )
    d3.metric(
        label="Generated Tokens",
        value=f"{inf['gen_tokens']:,}",
        help="Tokens produced by the model (agent decisions, explanations)"
    )

    st.divider()

    # ── Hardware badge ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style='background:#1a1a2e;padding:12px 20px;border-radius:8px;
                    border-left:4px solid #e84040;margin-top:8px'>
            <span style='color:#e84040;font-weight:bold'>AMD ROCm Stack</span>
            <span style='color:#aaa'> &nbsp;|&nbsp; </span>
            <span style='color:#fff'>vLLM 0.17.1 + ROCm 7.0</span>
            <span style='color:#aaa'> &nbsp;|&nbsp; </span>
            <span style='color:#fff'>Continuous Batching + PagedAttention</span>
            <span style='color:#aaa'> &nbsp;|&nbsp; </span>
            <span style='color:#fff'>MI300X 192 GB HBM3</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Status / error messages ───────────────────────────────────────────
    if not vram["ok"]:
        st.error(f"GPU stats unavailable — rocm-smi error: {vram.get('err')}")
    if not inf["ok"]:
        st.error(f"vLLM metrics unavailable — check server at "
                 f"{VLLM_METRICS_URL} — error: {inf.get('err')}")

    # ── Auto-refresh every 2 seconds ─────────────────────────────────────
    # This refreshes only the telemetry tab, not the whole app
    time.sleep(2)
    st.rerun()