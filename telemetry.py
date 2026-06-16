"""
telemetry.py — AMD ROCm + vLLM live metrics.

Two sources: rocm-smi (VRAM used/total/%) and the vLLM /metrics Prometheus feed
(tokens/sec, active/queued requests, KV cache). Both collectors return safe
defaults on failure, so the dashboard never crashes.

    from telemetry import render_telemetry_tab
    with st.tabs(["KYC Pipeline", "ROCm Telemetry"])[1]:
        render_telemetry_tab()
"""

import subprocess
import json
import time
import requests
from datetime import datetime
from collections import deque

VLLM_METRICS_URL = "http://localhost:8000/metrics"   # change port if needed
HISTORY_POINTS   = 30                                 # tokens/sec rate window

# Rolling (timestamp, cumulative-tokens) history for the tokens/sec rate.
_token_history = deque(maxlen=HISTORY_POINTS)


# ── Data collection ──────────────────────────────────────────────────────────

def get_vram():
    """
    Read GPU memory from rocm-smi. Returns used/total GB, percentage, and ok.
    Falls back to 0 GB / 192 GB on any failure (e.g. no AMD GPU on a laptop).
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=3
        )
        data  = json.loads(result.stdout)
        card  = list(data.values())[0]          # first GPU card

        total_bytes = int(card.get("VRAM Total Memory (B)", 0))
        used_bytes  = int(card.get("VRAM Total Used Memory (B)", 0))

        total_gb = round(total_bytes / (1024 ** 3), 1)
        used_gb  = round(used_bytes  / (1024 ** 3), 1)
        pct      = round((used_gb / total_gb * 100), 1) if total_gb > 0 else 0.0

        return {"used_gb": used_gb, "total_gb": total_gb, "pct": pct, "ok": True}

    except Exception as e:
        return {"used_gb": 0.0, "total_gb": 192.0, "pct": 0.0, "ok": False, "err": str(e)}


def get_vllm_stats():
    """
    Read the vLLM Prometheus /metrics feed: token counts, active/queued
    requests, KV cache, and a rolling tokens/sec rate. Returns safe zeros if
    the endpoint is unreachable.
    """
    try:
        text = requests.get(VLLM_METRICS_URL, timeout=3).text

        # Parse "metric_name{label="x",...} value". Strip the {...} label suffix
        # so the bare name is the key, and SUM across label series.
        raw = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].split("{")[0]
                try:
                    raw[name] = raw.get(name, 0.0) + float(parts[1])
                except ValueError:
                    pass

        # Token counts (metric names vary slightly by vLLM version)
        prompt_tokens = raw.get(
            "vllm:prompt_tokens_total_count",
            raw.get("vllm:prompt_tokens_total", 0)
        )
        gen_tokens = raw.get(
            "vllm:generation_tokens_total_count",
            raw.get("vllm:generation_tokens_total", 0)
        )
        total_tokens = prompt_tokens + gen_tokens

        # Tokens/sec from the rolling history window
        now = time.time()
        _token_history.append((now, total_tokens))
        tps = 0.0
        if len(_token_history) >= 2:
            dt   = _token_history[-1][0] - _token_history[0][0]
            dtok = _token_history[-1][1] - _token_history[0][1]
            tps  = round(dtok / dt, 1) if dt > 0 else 0.0

        active_req  = int(raw.get("vllm:num_requests_running", 0))
        queued_req  = int(raw.get("vllm:num_requests_waiting", 0))
        # V1 engine renamed this gauge; accept either. Value is a 0..1 fraction.
        kv_cache    = round(raw.get(
            "vllm:kv_cache_usage_perc",
            raw.get("vllm:gpu_cache_usage_perc", 0)
        ) * 100, 1)

        # Keep the parsed vllm: series for the live "raw metrics" expander.
        raw_vllm = {k: v for k, v in sorted(raw.items()) if k.startswith("vllm:")}

        return {
            "tps":            tps,
            "total_tokens":   int(total_tokens),
            "prompt_tokens":  int(prompt_tokens),
            "gen_tokens":     int(gen_tokens),
            "active_requests": active_req,
            "queued_requests": queued_req,
            "kv_cache_pct":   kv_cache,
            "raw":            raw_vllm,
            "ok":             True
        }

    except Exception as e:
        return {
            "tps": 0.0, "total_tokens": 0, "prompt_tokens": 0,
            "gen_tokens": 0, "active_requests": 0, "queued_requests": 0,
            "kv_cache_pct": 0.0, "raw": {}, "ok": False, "err": str(e)
        }


def get_snapshot():
    """Everything the dashboard needs in one call: vram, inference, timestamp."""
    return {
        "vram":      get_vram(),
        "inference": get_vllm_stats(),
        "timestamp": datetime.now().strftime("%H:%M:%S IST")
    }


# ── Streamlit rendering ──────────────────────────────────────────────────────

def render_telemetry_tab():
    """Render the full ROCm telemetry tab and self-refresh every 2 seconds."""
    import streamlit as st

    snap = get_snapshot()
    vram = snap["vram"]
    inf  = snap["inference"]

    st.markdown("## ⚡ AMD Instinct MI300X — Live Telemetry")
    st.caption(f"Last refreshed: {snap['timestamp']}  •  "
               f"Source: rocm-smi + vLLM /metrics")
    st.divider()

    # Row 1: headline numbers
    st.markdown("### Inference Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label="🚀 Tokens / Second", value=f"{inf['tps']}",
              help="Generated tokens per second across all active requests")
    c2.metric(label="🔢 Total Tokens Generated", value=f"{inf['total_tokens']:,}",
              help="Cumulative tokens generated since vLLM server started")
    c3.metric(label="🔄 Active Requests", value=inf["active_requests"],
              help="KYC agent requests currently being processed by the GPU")
    c4.metric(label="⏳ Queue Depth", value=inf["queued_requests"],
              help="Requests waiting — rises during bulk import stress test")
    st.divider()

    # Row 2: VRAM bar
    st.markdown("### VRAM Utilisation (192 GB HBM3)")
    col_bar, col_num = st.columns([4, 1])
    vram_fraction = min(vram["pct"] / 100, 1.0)
    col_bar.progress(vram_fraction,
                     text=f"{vram['used_gb']} GB used  /  {vram['total_gb']} GB total")
    col_num.metric("VRAM %", f"{vram['pct']}%")

    if vram["pct"] > 85:
        st.warning("⚠️ VRAM usage high — consider reducing batch size")
    elif vram["pct"] > 50:
        st.info("ℹ️ VRAM at moderate utilisation")
    else:
        st.success("✅ VRAM healthy")
    st.divider()

    # Row 3: inference detail
    st.markdown("### Inference Detail")
    d1, d2, d3 = st.columns(3)
    d1.metric(label="KV Cache Used", value=f"{inf['kv_cache_pct']}%",
              help="vLLM PagedAttention KV cache fill — climbs under bulk load")
    d2.metric(label="Prompt Tokens", value=f"{inf['prompt_tokens']:,}",
              help="Tokens from input documents and instructions")
    d3.metric(label="Generated Tokens", value=f"{inf['gen_tokens']:,}",
              help="Tokens produced by the model (agent decisions, explanations)")
    st.divider()

    # Hardware badge
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

    # Raw parsed metrics — X-ray for "why is this 0?" (missing / renamed / zero).
    with st.expander("🔬 Raw parsed metrics (live)"):
        rc1, rc2 = st.columns(2)
        with rc1:
            st.caption("vLLM /metrics (parsed `vllm:` series)")
            raw_vllm = inf.get("raw", {})
            if raw_vllm:
                st.json(raw_vllm)
            else:
                st.warning("No `vllm:` metrics parsed — endpoint unreachable "
                           "or returned no vllm series.")
        with rc2:
            st.caption("rocm-smi VRAM (parsed)")
            st.json({k: v for k, v in vram.items() if k != "raw"})

    # Error messages
    if not vram["ok"]:
        st.error(f"GPU stats unavailable — rocm-smi error: {vram.get('err')}")
    if not inf["ok"]:
        st.error(f"vLLM metrics unavailable — check server at "
                 f"{VLLM_METRICS_URL} — error: {inf.get('err')}")

    # Auto-refresh this tab every 2 seconds.
    time.sleep(2)
    st.rerun()
