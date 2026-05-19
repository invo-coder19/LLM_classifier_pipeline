# 🧠 LLM Evaluation Pipeline

> **Automated quality gates for LLM systems** — just like unit tests run on code changes, this pipeline runs on every prompt change, model swap, or RAG knowledge-base update.

[![CI](https://github.com/YOUR_ORG/LLM_classifier_pipeline/actions/workflows/llm_eval.yml/badge.svg)](https://github.com/YOUR_ORG/LLM_classifier_pipeline/actions/workflows/llm_eval.yml)
[![Coverage](https://codecov.io/gh/YOUR_ORG/LLM_classifier_pipeline/badge.svg)](https://codecov.io/gh/YOUR_ORG/LLM_classifier_pipeline)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🗺️ Architecture

```
Git Push / PR
     │
     ▼
┌─────────────────────────────────────────────────────┐
│            GitHub Actions CI/CD Pipeline            │
│                                                     │
│  ① validate-dataset   Schema check on golden set   │
│         │                                           │
│  ② unit-tests         Fast metric function tests   │
│         │                                           │
│  ③ eval-suite         4 parallel shards × 26 Qs    │
│      (shard 0..3)     → uploads JSON results        │
│         │                                           │
│  ④ gate-check         Merge shards → enforce SLAs  │
│         │             → posts PR comment            │
│         │             → exit 1 if violations        │
│         │                                           │
│  ⑤ update-dashboard  Commits results → triggers    │
│                       dashboard refresh             │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Metrics

| Metric | How Measured | Blocking Threshold |
|--------|-------------|-------------------|
| **Hallucination Rate** | NLI entailment vs. source docs | > 5% → ❌ blocks merge |
| **p95 Latency** | Wall-clock time across all queries | > 5000ms → ❌ blocks merge |
| **Answer Relevancy** | Cosine similarity (question ↔ answer) | < 0.70 → ⚠️ warning |
| **Faithfulness** | Sentence-level NLI vs. sources | < 0.75 → ⚠️ warning |
| **Cost / Query** | tiktoken × model pricing table | > $0.01 → ⚠️ warning |

---

## ⚡ Quick Start

### 1. Clone & set up virtual environment

```bash
git clone https://github.com/YOUR_ORG/LLM_classifier_pipeline.git
cd LLM_classifier_pipeline

# Create venv and install dependencies
make install
# or manually:
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and API keys
```

### 3. Run evaluation (mock mode — no API key needed)

```bash
make eval-mock
```

### 4. Run with a real LLM

```bash
# Set OPENAI_API_KEY in .env, then:
make eval
```

### 5. Launch dashboard

```bash
make dashboard
# Opens http://localhost:8501
```

### 6. Run unit tests

```bash
make test
```

---

## 📁 Project Structure

```
LLM_classifier_pipeline/
├── .github/
│   └── workflows/
│       └── llm_eval.yml        # CI/CD pipeline
├── data/
│   ├── golden_dataset.json     # 105 curated QA pairs
│   └── schema.json             # Dataset validation schema
├── eval/
│   ├── runner.py               # Main orchestrator
│   ├── gates.py                # Threshold enforcement
│   └── metrics/
│       ├── hallucination.py    # NLI-based detection
│       ├── relevancy.py        # Semantic similarity
│       ├── faithfulness.py     # Source attribution
│       ├── latency.py          # p50/p95 stats
│       └── cost.py             # Token cost estimation
├── pipeline/
│   ├── llm_client.py           # OpenAI / Anthropic / Ollama / Mock
│   ├── rag.py                  # Mock + ChromaDB retriever
│   └── config.py               # Environment-driven config
├── dashboard/
│   └── app.py                  # Streamlit metrics dashboard
├── scripts/
│   ├── merge_and_gate.py       # Shard merger + gate check
│   └── validate_dataset.py     # Schema validator
├── tests/
│   ├── test_gates.py           # Gate logic tests
│   └── test_metrics.py         # Metric function tests
├── results/                    # JSON results per CI run (git-tracked)
├── .env.example
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🔧 Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | `openai` \| `anthropic` \| `ollama` \| `mock` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the chosen provider |
| `RAG_BACKEND` | `mock` | `mock` \| `chromadb` |
| `HALLUCINATION_THRESHOLD` | `0.05` | Max allowed hallucination rate |
| `P95_LATENCY_MS` | `5000` | Max p95 latency in milliseconds |
| `MIN_ANSWER_RELEVANCY` | `0.70` | Minimum semantic relevancy score |
| `MIN_FAITHFULNESS` | `0.75` | Minimum faithfulness score |

---

## 🏃 GitHub Actions Setup

### Required Secrets

In your GitHub repository → **Settings → Secrets and variables → Actions**:

| Secret | Required | Description |
|--------|----------|-------------|
| `OPENAI_API_KEY` | If using OpenAI | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Your Anthropic API key |

> **CI uses mock mode by default** — no secrets needed to run the pipeline. Set `LLM_PROVIDER` to `openai` or `anthropic` in the workflow env to use real LLMs.

### Trigger Conditions

The pipeline triggers automatically on push/PR when any of these paths change:

- `pipeline/**` — prompt changes, model swaps
- `data/**` — RAG knowledge base updates
- `eval/**` — metric or gate logic changes

---

## 🗂️ Golden Dataset

105 curated QA pairs across 6 categories:

| Category | Count | Purpose |
|----------|-------|---------|
| `factual` | 70 | Standard knowledge retrieval |
| `reasoning` | 10 | Multi-step logical inference |
| `multi-hop` | 8 | Cross-document reasoning |
| `adversarial` | 2 | Prompt injection attempts |
| `edge-case` | 8 | Empty input, gibberish, multi-question |
| `out-of-scope` | 7 | Questions not in knowledge base |

Expand the dataset with `scripts/generate_dataset.py` or edit `data/golden_dataset.json` directly.

---

## 📊 Dashboard

The Streamlit dashboard shows:

- **KPI cards** with trend vs. previous run
- **Time-series charts** for all metrics
- **Gate pass/fail markers** on every chart
- **Run history table** filterable by model/provider/date

![Dashboard Preview](dashboard/preview.png)

---

## 🤝 Adding a New Metric

1. Create `eval/metrics/your_metric.py` with a `compute_*` and `mean_*` function
2. Import and call it in `eval/runner.py`
3. Add the threshold to `eval/gates.py` if it should block merges
4. Add unit tests in `tests/test_metrics.py`

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
