"""
LLM Eval Dashboard — loads historical run results and renders metric trends.
Run with: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-value { font-size: 2.2rem; font-weight: 700; margin: 8px 0; }
.metric-label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.pass { color: #00d4aa; }
.fail { color: #ff4b6e; }
.warn { color: #ffb347; }
.badge-pass { background:#00d4aa22; color:#00d4aa; border:1px solid #00d4aa44; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600; }
.badge-fail { background:#ff4b6e22; color:#ff4b6e; border:1px solid #ff4b6e44; border-radius:20px; padding:3px 12px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
SAMPLE_RESULTS = [
    {"timestamp": "2026-05-19T08:00:00+00:00", "model": "gpt-4o-mini", "provider": "openai",
     "total_questions": 105, "gate": {"passed": True},
     "metrics": {"hallucination_rate": 0.02, "mean_answer_relevancy": 0.84, "mean_faithfulness": 0.81,
                 "latency": {"p50_ms": 320, "p95_ms": 890}, "cost": {"mean_per_query_usd": 0.0004}}},
    {"timestamp": "2026-05-18T12:00:00+00:00", "model": "gpt-4o-mini", "provider": "openai",
     "total_questions": 105, "gate": {"passed": True},
     "metrics": {"hallucination_rate": 0.03, "mean_answer_relevancy": 0.82, "mean_faithfulness": 0.79,
                 "latency": {"p50_ms": 340, "p95_ms": 920}, "cost": {"mean_per_query_usd": 0.0004}}},
    {"timestamp": "2026-05-17T10:00:00+00:00", "model": "gpt-4o-mini", "provider": "openai",
     "total_questions": 105, "gate": {"passed": False},
     "metrics": {"hallucination_rate": 0.08, "mean_answer_relevancy": 0.75, "mean_faithfulness": 0.72,
                 "latency": {"p50_ms": 4100, "p95_ms": 5800}, "cost": {"mean_per_query_usd": 0.0004}}},
    {"timestamp": "2026-05-16T09:00:00+00:00", "model": "gpt-4o-mini", "provider": "openai",
     "total_questions": 105, "gate": {"passed": True},
     "metrics": {"hallucination_rate": 0.04, "mean_answer_relevancy": 0.80, "mean_faithfulness": 0.78,
                 "latency": {"p50_ms": 310, "p95_ms": 870}, "cost": {"mean_per_query_usd": 0.0003}}},
    {"timestamp": "2026-05-15T08:00:00+00:00", "model": "claude-3-haiku", "provider": "anthropic",
     "total_questions": 105, "gate": {"passed": True},
     "metrics": {"hallucination_rate": 0.01, "mean_answer_relevancy": 0.88, "mean_faithfulness": 0.85,
                 "latency": {"p50_ms": 280, "p95_ms": 750}, "cost": {"mean_per_query_usd": 0.0002}}},
]


@st.cache_data(ttl=60)
def load_results() -> list[dict]:
    runs = []
    if RESULTS_DIR.exists():
        for f in sorted(RESULTS_DIR.glob("*.json")):
            try:
                runs.append(json.loads(f.read_text()))
            except Exception:
                pass
    return runs if runs else SAMPLE_RESULTS


def to_df(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = r.get("metrics", {})
        rows.append({
            "timestamp": pd.to_datetime(r.get("timestamp")),
            "model": r.get("model", "unknown"),
            "provider": r.get("provider", "unknown"),
            "total_questions": r.get("total_questions", 0),
            "gate_passed": r.get("gate", {}).get("passed", True),
            "hallucination_rate": m.get("hallucination_rate", 0),
            "relevancy": m.get("mean_answer_relevancy", 0),
            "faithfulness": m.get("mean_faithfulness", 0),
            "p50_ms": m.get("latency", {}).get("p50_ms", 0),
            "p95_ms": m.get("latency", {}).get("p95_ms", 0),
            "cost_per_query": m.get("cost", {}).get("mean_per_query_usd", 0),
        })
    return pd.DataFrame(rows).sort_values("timestamp")




# ── Layout ────────────────────────────────────────────────────────────────────
runs = load_results()
df = to_df(runs)

st.markdown("# 🧠 LLM Evaluation Dashboard")
st.markdown("Automated quality gate tracking — hallucination · relevancy · latency · cost · faithfulness")
st.divider()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Filters")
    models = ["All"] + sorted(df["model"].unique().tolist())
    selected_model = st.selectbox("Model", models)
    providers = ["All"] + sorted(df["provider"].unique().tolist())
    selected_provider = st.selectbox("Provider", providers)
    date_range = st.date_input("Date range", [df["timestamp"].min().date(), df["timestamp"].max().date()])
    st.divider()
    st.markdown("### 🎯 Thresholds")
    st.metric("Max Hallucination", "5%")
    st.metric("Max p95 Latency", "5000 ms")
    st.metric("Min Relevancy", "0.70")
    st.metric("Min Faithfulness", "0.75")

filtered = df.copy()
if selected_model != "All":
    filtered = filtered[filtered["model"] == selected_model]
if selected_provider != "All":
    filtered = filtered[filtered["provider"] == selected_provider]
# Apply date range filter (date_range may be a 1- or 2-element tuple)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = filtered[(filtered["timestamp"] >= start_dt) & (filtered["timestamp"] <= end_dt)]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    start_dt = pd.Timestamp(date_range[0])
    filtered = filtered[filtered["timestamp"].dt.date == date_range[0]]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
if not filtered.empty:
    latest = filtered.iloc[-1]
    prev = filtered.iloc[-2] if len(filtered) > 1 else latest

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        delta = (latest["hallucination_rate"] - prev["hallucination_rate"]) * 100
        status = "pass" if latest["hallucination_rate"] <= 0.05 else "fail"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Hallucination Rate</div>
            <div class="metric-value {status}">{latest['hallucination_rate']:.1%}</div>
            <div style="color:#888; font-size:0.8rem">Δ {delta:+.1f}pp vs prev</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        status = "pass" if latest["relevancy"] >= 0.70 else "warn"
        delta = latest["relevancy"] - prev["relevancy"]
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Answer Relevancy</div>
            <div class="metric-value {status}">{latest['relevancy']:.3f}</div>
            <div style="color:#888; font-size:0.8rem">Δ {delta:+.3f} vs prev</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        status = "pass" if latest["faithfulness"] >= 0.75 else "warn"
        delta = latest["faithfulness"] - prev["faithfulness"]
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Faithfulness</div>
            <div class="metric-value {status}">{latest['faithfulness']:.3f}</div>
            <div style="color:#888; font-size:0.8rem">Δ {delta:+.3f} vs prev</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        status = "pass" if latest["p95_ms"] <= 5000 else "fail"
        delta = latest["p95_ms"] - prev["p95_ms"]
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">p95 Latency</div>
            <div class="metric-value {status}">{latest['p95_ms']:.0f}ms</div>
            <div style="color:#888; font-size:0.8rem">Δ {delta:+.0f}ms vs prev</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        status = "pass" if latest["cost_per_query"] <= 0.01 else "warn"
        delta = latest["cost_per_query"] - prev["cost_per_query"]
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Cost / Query</div>
            <div class="metric-value {status}">${latest['cost_per_query']:.5f}</div>
            <div style="color:#888; font-size:0.8rem">Δ ${delta:+.5f} vs prev</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trend charts ──────────────────────────────────────────────────────────
    st.markdown("## 📈 Metric Trends Over Time")
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["hallucination_rate"] * 100,
            name="Hallucination %", line=dict(color="#ff4b6e", width=2),
            fill="tozeroy", fillcolor="rgba(255,75,110,0.1)", mode="lines+markers",
        ))
        fig.add_hline(y=5, line_dash="dash", line_color="#ffb347", annotation_text="5% threshold")
        fig.update_layout(
            title="Hallucination Rate (%)", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,30,46,0.8)",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
        )
        # Gate pass/fail markers
        for _, row in filtered.iterrows():
            color = "green" if row["gate_passed"] else "red"
            fig.add_vline(x=row["timestamp"], line_color=color, line_width=0.5, opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["p95_ms"],
            name="p95 Latency", line=dict(color="#7b68ee", width=2),
            fill="tozeroy", fillcolor="rgba(123,104,238,0.1)", mode="lines+markers",
        ))
        fig2.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["p50_ms"],
            name="p50 Latency", line=dict(color="#00d4aa", width=2, dash="dot"),
            mode="lines+markers",
        ))
        fig2.add_hline(y=5000, line_dash="dash", line_color="#ffb347", annotation_text="5s SLA")
        fig2.update_layout(
            title="Latency (ms)", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,30,46,0.8)",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["relevancy"],
            name="Relevancy", line=dict(color="#00d4aa", width=2),
            mode="lines+markers",
        ))
        fig3.add_trace(go.Scatter(
            x=filtered["timestamp"], y=filtered["faithfulness"],
            name="Faithfulness", line=dict(color="#ffd700", width=2),
            mode="lines+markers",
        ))
        fig3.add_hline(y=0.70, line_dash="dash", line_color="#00d4aa55")
        fig3.add_hline(y=0.75, line_dash="dash", line_color="#ffd70055")
        fig3.update_layout(
            title="Quality Scores (0–1)", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,30,46,0.8)",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=filtered["timestamp"], y=filtered["cost_per_query"] * 1000,
            name="Cost/Query (m$)",
            marker_color=["#00d4aa" if p else "#ff4b6e" for p in filtered["gate_passed"]],
        ))
        fig4.update_layout(
            title="Cost per Query (milli-$)", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,30,46,0.8)",
            height=300, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Run history table ─────────────────────────────────────────────────────
    st.markdown("## 🗃️ Run History")
    display_df = filtered[["timestamp", "model", "provider", "total_questions",
                            "gate_passed", "hallucination_rate", "relevancy",
                            "faithfulness", "p95_ms", "cost_per_query"]].copy()
    display_df["gate_passed"] = display_df["gate_passed"].map(
        {True: "✅ PASS", False: "❌ FAIL"}
    )
    display_df["hallucination_rate"] = display_df["hallucination_rate"].map("{:.1%}".format)
    display_df["relevancy"] = display_df["relevancy"].map("{:.3f}".format)
    display_df["faithfulness"] = display_df["faithfulness"].map("{:.3f}".format)
    display_df["p95_ms"] = display_df["p95_ms"].map("{:.0f} ms".format)
    display_df["cost_per_query"] = display_df["cost_per_query"].map("${:.5f}".format)
    display_df.columns = ["Timestamp", "Model", "Provider", "Questions",
                          "Gate", "Hallucination", "Relevancy", "Faithfulness", "p95", "Cost/Query"]
    st.dataframe(display_df.sort_values("Timestamp", ascending=False), use_container_width=True)
else:
    st.warning("No results match the selected filters.")
