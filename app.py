# app.py: Executive Decision Intelligence Control Room (Liquid d1-ultra)
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

DEFAULT_API_KEY = os.getenv("LIQUID_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))

def resolve_client(override_key: str = ""):
    active_key = override_key.strip() if override_key and override_key.strip() else DEFAULT_API_KEY
    if not active_key:
        raise ValueError(
            "Missing API Key. Set LIQUID_API_KEY in Space Secrets or enter it in the Settings drawer."
        )
    base_url = "https://openrouter.ai/api/v1" if active_key.startswith("sk-or-") else "https://api.liquid.ai/v1"
    return TypeSafeClient(api_key=active_key, base_url=base_url, model="liquid/d1-ultra")

# -------------------------------------------------------------
# TAB 1: SINGLE-PASS TELEMETRY ROUTER (From Cell 3)
# -------------------------------------------------------------
def build_domain_request(domain, p1, p2):
    if domain == "Aviation Flight Dispatch":
        state = (
            f"Flight UA2482 (B777-200) inbound to ORD. Severe terminal convective weather active. "
            f"Aircraft fuel remaining above final reserve is {int(p1)} minutes. Destination airport reports "
            f"thunderstorms and crosswinds gusting to {int(p2)} knots. Alternate field MKE available."
        )
        questions = {
            "dispatch_action": Choice(
                instructions="Determine immediate tactical flight dispatch action.",
                criteria={
                    "continue_hold": "Fuel reserves allow >=20 min holding; airport acceptance expected.",
                    "divert_alternate": "Holding margin critically thin or alternate minimums declining.",
                    "declare_emergency": "Final reserve fuel threatened or emergency priority required.",
                }
            ),
            "cancellation_risk": Noul(
                instructions="Does this flight face high probability of route cancellation or return-to-gate?"
            ),
            "disruption_severity": Score(
                instructions="Rate network operational disruption severity.",
                criteria=[
                    "Minor: Localized delay under 30 minutes.",
                    "Moderate: 30-90 min delay, manageable duty time impact.",
                    "Severe: Misconnected international passengers, curfew risks.",
                    "Critical: Stranded aircraft, crew timeout, cascading cancellations."
                ]
            )
        }
    elif domain == "S&P 500 Macro Shock":
        state = (
            f"Macroeconomic release print: Core CPI annualized surprise printed at +0.{int(p1)}% above consensus. "
            f"10-Year Treasury Yield surged by {int(p2)} bps intraday. S&P 500 futures fell sharply in pre-market trade."
        )
        questions = {
            "macro_regime": Choice(
                instructions="Classify dominant equity market regime.",
                criteria={
                    "risk_on_expansion": "Accommodative policy, benign inflation, expanding multiples.",
                    "stagflation_shock": "Sticky inflation, rising yields, compressed equity valuations.",
                    "liquidity_crunch": "Flight to cash, credit spread widening, disorderly liquidation.",
                }
            ),
            "volatility_breakout": Noul(
                instructions="Will implied volatility (VIX) sustain an upside regime shift?"
            ),
            "drawdown_risk": Score(
                instructions="Assess short-term equity index drawdown probability.",
                criteria=[
                    "Low: Consolidation / shallow dip (<1%).",
                    "Moderate: Standard pull-back (1% to 3%).",
                    "High: Accelerated selling pressure (3% to 6%).",
                    "Extreme: Systematic CTA / risk-parity liquidation (>6%)."
                ]
            )
        }
    else:
        state = (
            f"SDR monitoring 2.440 GHz ISM band. Baseband IQ capture shows a periodic narrowband burst. "
            f"Measured Carrier-to-Noise Ratio (CNR) is {int(p1)} dB sweeping across {int(p2)} MHz every 400us, "
            f"causing adjacent-channel desensitization on IEEE 802.15.4 nodes."
        )
        questions = {
            "signal_class": Choice(
                instructions="Identify the primary RF emitter classification.",
                criteria={
                    "standard_wifi_ofdm": "Wideband 20/40/80 MHz continuous packet OFDM transmission.",
                    "frequency_hopping_interferer": "Fast narrowband pulse hopper, Bluetooth, or jammer.",
                    "thermal_ambient_noise": "Gaussian distributed background floor without coherent structures.",
                }
            ),
            "hostile_jamming": Noul(
                instructions="Is the signal an intentional or malicious jamming/denial-of-service source?"
            ),
            "interference_impact": Score(
                instructions="Rate co-channel interference impact on co-located receivers.",
                criteria=[
                    "Negligible: Packet reception rate >99%.",
                    "Moderate: Occasional retries, degraded SNR.",
                    "Severe: Significant packet loss, receiver desensitization.",
                    "Mission Critical: Complete denial of communication link."
                ]
            )
        }
    return state, questions

def evaluate_decision(domain, p1, p2, threshold, user_key):
    state, questions = build_domain_request(domain, p1, p2)
    try:
        client = resolve_client(user_key)
        t0 = time.perf_counter()
        response = client.system_one(state=state, questions=questions)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        err_html = f"""
        <div style="background-color: #2e1015; border-left: 5px solid #ef4444; border-radius: 6px; padding: 14px; font-family: monospace; color: #fca5a5;">
            <div style="font-weight: bold; font-size: 14px;">❌ Execution Error</div>
            <div style="font-size: 12px; margin-top: 6px;">{str(exc)}</div>
        </div>
        """
        fig, ax = plt.subplots(figsize=(6, 2))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#161b22')
        ax.text(0.5, 0.5, "Awaiting Valid API Key...", ha='center', va='center', color='#ef4444')
        ax.axis('off')
        return err_html, fig, state, "{}"

    answers = response.answers
    choice_k = next(k for k, v in answers.items() if v.type == 'choice')
    noul_k = next(k for k, v in answers.items() if v.type == 'noul')
    score_k = next(k for k, v in answers.items() if v.type == 'score')

    c_res, n_res, s_res = answers[choice_k], answers[noul_k], answers[score_k]

    is_confident = c_res.confidence >= (threshold / 100.0)
    is_high_risk = n_res.noul > 0.65 or s_res.score >= 2.0

    if is_confident and not is_high_risk:
        badge_color, badge_title = "#22c55e", "⚡ FAST-PATH DISPATCH"
        routing_desc = f"Auto-execute action for '{c_res.choice.upper()}' (Confidence: {c_res.confidence:.1%})."
    elif is_confident and is_high_risk:
        badge_color, badge_title = "#eab308", "⚠ HIGH-RISK ESCALATION"
        routing_desc = f"Dispatch alert to human controller ({c_res.choice.upper()}) | Risk: {n_res.noul:.1%}."
    else:
        badge_color, badge_title = "#ef4444", "🔄 DELIBERATIVE REASONING FALLBACK"
        routing_desc = f"Confidence {c_res.confidence:.1%} < {threshold}%. Trigger System 2 deep analysis."

    gate_html = f"""
    <div style="background-color: #161b22; border-left: 5px solid {badge_color}; border-radius: 8px; padding: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="font-size: 15px; font-weight: bold; color: {badge_color};">{badge_title}</div>
            <span style="background: #21262d; color: #58a6ff; padding: 3px 8px; border-radius: 4px; font-family: monospace; font-size: 11px;">Backbone: Liquid d1-ultra</span>
        </div>
        <div style="color: #c9d1d9; font-size: 13px; margin-top: 6px;">{routing_desc}</div>
        <div style="display: flex; gap: 24px; margin-top: 12px; font-family: monospace; font-size: 12px; color: #8b949e;">
            <span>⏱️ Latency: <strong style="color: #58a6ff;">{latency_ms:.1f} ms</strong></span>
            <span>🎯 Primary Route: <strong style="color: #f0f6fc;">{c_res.choice.upper()}</strong></span>
            <span>📊 Risk (Noul): <strong style="color: {badge_color};">{n_res.noul:.1%}</strong></span>
            <span>📈 Severity: <strong style="color: #f0f6fc;">{s_res.score:.2f} / 3.0</strong></span>
        </div>
    </div>
    """

    plt.style.use('dark_background')
    fig, (ax_choice, ax_risk) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=130, gridspec_kw={'width_ratios': [1.4, 1]})
    fig.patch.set_facecolor('#0d1117')
    ax_choice.set_facecolor('#161b22')
    ax_risk.set_facecolor('#161b22')

    opts = list(c_res.probabilities.keys())
    probs = [c_res.probabilities[k] for k in opts]
    colors = ['#388bfd' if k == c_res.choice else '#30363d' for k in opts]

    y_pos = np.arange(len(opts))
    bars = ax_choice.barh(y_pos, probs, color=colors, edgecolor='#58a6ff', height=0.55)
    ax_choice.set_yticks(y_pos)
    ax_choice.set_yticklabels([k.replace('_', ' ').title() for k in opts], fontsize=9, color='#c9d1d9')
    ax_choice.set_xlim(0, 1.08)
    ax_choice.set_title(f"Pointer Distribution: {choice_k.upper()}", fontsize=10, fontweight='bold', color='#f0f6fc')
    ax_choice.grid(axis='x', linestyle='--', alpha=0.15)

    for bar, p in zip(bars, probs):
        ax_choice.text(p + 0.02, bar.get_y() + bar.get_height()/2, f"{p*100:.1f}%",
                       va='center', ha='left', fontsize=8.5, fontweight='bold', color='#58a6ff' if p > 0.5 else '#8b949e')

    ax_risk.set_xlim(0, 1)
    ax_risk.set_ylim(0, 2)
    ax_risk.axis('off')
    ax_risk.text(0, 1.7, f"Binary Indicator: {noul_k.upper()}", fontsize=9, fontweight='bold', color='#f0f6fc')
    ax_risk.barh(1.4, 1.0, color='#21262d', height=0.25)
    ax_risk.barh(1.4, n_res.noul, color=badge_color, height=0.25)
    ax_risk.text(1.02, 1.4, f"{n_res.noul:.1%}", va='center', fontsize=9, fontweight='bold', color=badge_color)

    ax_risk.text(0, 0.8, f"Ordinal Severity: {score_k.upper()}", fontsize=9, fontweight='bold', color='#f0f6fc')
    ax_risk.barh(0.5, 1.0, color='#21262d', height=0.25)
    ax_risk.barh(0.5, s_res.score / 3.0, color='#a855f7', height=0.25)
    ax_risk.text(1.02, 0.5, f"{s_res.score:.2f} / 3.0", va='center', fontsize=9, fontweight='bold', color='#c084fc')

    plt.tight_layout()

    audit_payload = json.dumps({
        "engine": "liquid/d1-ultra",
        "domain": domain,
        "latency_ms": round(latency_ms, 2),
        "routing_decision": badge_title,
        "choice": c_res.choice,
        "confidence": round(c_res.confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in c_res.probabilities.items()},
        "noul_risk": round(n_res.noul, 4),
        "ordinal_severity": round(s_res.score, 3)
    }, indent=2)

    return gate_html, fig, state, audit_payload

# -------------------------------------------------------------
# TAB 2: CALIBRATION & RELIABILITY AUDITOR (From Cell 4)
# -------------------------------------------------------------
CALIBRATION_CASES = [
    ("S&P up 0.2%, VIX at 12.8, yields stable, low options activity.", False),
    ("Tech sector earnings beat consensus, markets consolidating normally.", False),
    ("Flight on schedule, CAVOK visual conditions, light winds 6 kts.", False),
    ("Standard 802.11ax WiFi beacon traffic within normal spectral mask.", False),
    ("Bond yields spiked 15 bps, oil up 5%, credit spreads widening.", True),
    ("Flight fuel at minimum divert reserve, runway crosswind 48 kts gusting.", True),
    ("Broadband noise pulsing across ISM band, SNR collapsed to -3 dB.", True),
    ("Commercial paper spread widened 40 bps, liquidity drying up.", True),
    ("High-frequency jammer sweeping telemetry frequencies persistently.", True),
    ("Severe line squall over destination airport, all arrivals halted.", True),
]

def run_calibration_audit(user_key):
    client = resolve_client(user_key)
    preds, targets = [], []
    for state, label in CALIBRATION_CASES:
        res = client.system_one(
            state=state,
            questions={"crisis_indicator": Noul(instructions="Does this telemetry describe an imminent operational crisis or disruption?")}
        )
        preds.append(res.answers["crisis_indicator"].noul)
        targets.append(1.0 if label else 0.0)

    preds, targets = np.array(preds), np.array(targets)
    brier = float(np.mean((preds - targets) ** 2))

    n_bins = 5
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_confs, bin_accs = [], []

    for i in range(n_bins):
        mask = (preds >= bin_limits[i]) & (preds < bin_limits[i+1])
        if np.sum(mask) > 0:
            conf = np.mean(preds[mask])
            acc = np.mean(targets[mask])
            ece += (np.sum(mask) / len(preds)) * np.abs(acc - conf)
            bin_confs.append(conf)
            bin_accs.append(acc)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.plot([0, 1], [0, 1], "--", color="#8b949e", label="Ideal Calibration")
    ax.scatter(bin_confs, bin_accs, color="#388bfd", s=70, zorder=5, label="Observed Bins")
    ax.plot(bin_confs, bin_accs, color="#388bfd", alpha=0.7)
    ax.set_title(f"Reliability Diagram (Brier: {brier:.4f} | ECE: {ece:.4f})", color="#f0f6fc", fontweight="bold")
    ax.set_xlabel("Model Confidence P(crisis)")
    ax.set_ylabel("Empirical Frequency")
    ax.grid(True, alpha=0.15)
    ax.legend()
    plt.tight_layout()

    metrics_md = f"### 📊 Calibration Audit Results\n- **Brier Score:** `{brier:.4f}` *(0.0 = perfect)*\n- **Expected Calibration Error (ECE):** `{ece:.4f}`"
    return metrics_md, fig

# -------------------------------------------------------------
# TAB 3: CONCURRENCY & LATENCY PROFILER (From Cell 5)
# -------------------------------------------------------------
def run_stress_test(num_requests, workers, user_key):
    client = resolve_client(user_key)
    state, questions = build_domain_request("Aviation Flight Dispatch", 35, 45)

    def worker_call(_):
        t0 = time.perf_counter()
        _ = client.system_one(state=state, questions=questions)
        return (time.perf_counter() - t0) * 1000.0

    start_wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        latencies = list(executor.map(worker_call, range(int(num_requests))))
    total_wall = time.perf_counter() - start_wall
    latencies = np.array(latencies)

    summary_md = f"""
    ### ⚡ Concurrency & Tail-Latency Profile
    | Metric | Value |
    | :--- | :--- |
    | **Total Requests** | `{int(num_requests)}` |
    | **Concurrent Workers** | `{int(workers)}` |
    | **Total Wall Time** | `{total_wall:.2f} s` |
    | **Throughput (QPS)** | `{num_requests / total_wall:.1f} req/s` |
    | **P50 (Median) Latency** | `{np.percentile(latencies, 50):.1f} ms` |
    | **P95 Tail Latency** | `{np.percentile(latencies, 95):.1f} ms` |
    | **P99 Tail Latency** | `{np.percentile(latencies, 99):.1f} ms` |
    """
    return summary_md

# -------------------------------------------------------------
# GRADIO EXECUTIVE CONTROL ROOM UI
# -------------------------------------------------------------
css = """
body { background-color: #0a0d14 !important; color: #f0f6fc !important; }
.gradio-container { max-width: 1220px !important; margin: auto; }
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), css=css, title="Liquid d1 Control Room") as demo:
    gr.Markdown("""
    # 🛰️ Decision Intelligence Engine: Executive Control Room
    **Liquid AI d1-ultra Backbone** • Single-pass deterministic probability calibration across Aviation Operations, Macro Financial Shocks, and RF Spectrum Telemetry.
    """)

    with gr.Accordion("⚙️ Security & Infrastructure (API Key Override)", open=False):
        api_key_tb = gr.Textbox(label="Liquid / OpenRouter Key (Optional if Space Secret is set)", type="password", placeholder="liquid-... or sk-or-...")

    with gr.Tabs():
        with gr.TabItem("🎯 1. Live Telemetry Router"):
            with gr.Row():
                with gr.Column(scale=4):
                    domain_dd = gr.Dropdown(
                        label="Operational Domain",
                        choices=["Aviation Flight Dispatch", "S&P 500 Macro Shock", "RF Spectrum Telemetry"],
                        value="Aviation Flight Dispatch"
                    )
                    param1_sl = gr.Slider(minimum=5, maximum=90, value=38, step=1, label="Fuel Reserve Margin (Minutes)")
                    param2_sl = gr.Slider(minimum=10, maximum=65, value=45, step=1, label="Crosswind Velocity (Knots)")
                    thresh_sl = gr.Slider(minimum=50, maximum=98, value=85, step=1, label="Speculative Gate Confidence Threshold (%)")
                    run_btn = gr.Button("⚡ Execute Forward Pass", variant="primary")

                with gr.Column(scale=6):
                    gate_out = gr.HTML(label="Speculative Routing Gate")
                    plot_out = gr.Plot(label="Calibrated Probability Telemetry")

            with gr.Row():
                state_out = gr.Textbox(label="Serialized State Invariant", lines=3)
                audit_out = gr.Code(label="Compliance & Risk Audit Log (JSON)", language="json", lines=6)

        with gr.TabItem("📊 2. Calibration & Reliability Audit"):
            gr.Markdown("Evaluate **Brier Score** and **Expected Calibration Error (ECE)** across 10 multi-domain telemetry crisis states.")
            cal_btn = gr.Button("🔬 Run 10-Sample Calibration Audit", variant="primary")
            cal_md = gr.Markdown()
            cal_plot = gr.Plot()

        with gr.TabItem("⚡ 3. Concurrency & Throughput Stress-Test"):
            gr.Markdown("Benchmark multi-threaded QPS throughput and P50/P95/P99 latency percentiles.")
            with gr.Row():
                req_sl = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Total Requests")
                wrk_sl = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Concurrent Threads")
            stress_btn = gr.Button("🚀 Launch Stress Test", variant="primary")
            stress_md = gr.Markdown()

    def on_domain_change(domain):
        if domain == "Aviation Flight Dispatch":
            return gr.update(minimum=5, maximum=90, value=38, label="Fuel Reserve Margin (Minutes)"), gr.update(minimum=10, maximum=65, value=45, label="Crosswind Velocity (Knots)")
        elif domain == "S&P 500 Macro Shock":
            return gr.update(minimum=1, maximum=10, value=3, label="Core CPI Surprise (Tenths of %)"), gr.update(minimum=0, maximum=40, value=18, label="10-Year Yield Surge (Basis Points)")
        else:
            return gr.update(minimum=5, maximum=40, value=22, label="Jammer Carrier-to-Noise Ratio (dB)"), gr.update(minimum=10, maximum=160, value=20, label="Sweep Frequency Bandwidth (MHz)")

    domain_dd.change(on_domain_change, inputs=[domain_dd], outputs=[param1_sl, param2_sl])
    run_btn.click(evaluate_decision, inputs=[domain_dd, param1_sl, param2_sl, thresh_sl, api_key_tb], outputs=[gate_out, plot_out, state_out, audit_out])
    cal_btn.click(run_calibration_audit, inputs=[api_key_tb], outputs=[cal_md, cal_plot])
    stress_btn.click(run_stress_test, inputs=[req_sl, wrk_sl, api_key_tb], outputs=[stress_md])

if __name__ == "__main__":
    demo.launch()
