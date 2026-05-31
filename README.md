# 🧠 LLM Evaluation Pipeline

> Automated quality gates for LLM systems — runs on every prompt change, model swap, or RAG knowledge-base update, just like unit tests on code.

[![CI](https://github.com/invo-coder19/LLM_classifier_pipeline/actions/workflows/llm_eval.yml/badge.svg)](https://github.com/invo-coder19/LLM_classifier_pipeline/actions/workflows/llm_eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🗺️ Architecture

```
Git Push / PR
     │
     ▼
① validate-dataset   Schema check on golden dataset
② unit-tests         Metric & gate logic tests
③ cache-models       Cache HuggingFace NLI / embedding models
④ eval-suite         4 parallel shards × 105 questions → JSON results
⑤ gate-check         Merge shards → enforce SLAs → post PR comment
⑥ update-dashboard   Commit results → refresh Streamlit dashboard
```

---

## 📊 Quality Gates

| Metric | Threshold | Behaviour |
|--------|-----------|-----------|
| **Hallucination Rate** | > 5% | ❌ blocks merge |
| **p95 Latency** | > 5000 ms | ❌ blocks merge |
| **Answer Relevancy** | < 0.70 | ⚠️ warning |
| **Faithfulness** | < 0.75 | ⚠️ warning |
| **Cost / Query** | > $0.01 | ⚠️ warning |

---

## ⚡ Quick Start

```bash
# 1. Clone & install
git clone https://github.com/invo-coder19/LLM_classifier_pipeline.git
cd LLM_classifier_pipeline
make install

# 2. Configure
cp .env.example .env      # set LLM_PROVIDER + API keys

# 3. Run (mock mode — no API key needed)
make eval-mock

# 4. Run with a real LLM
make eval                 # requires OPENAI_API_KEY in .env

# 5. Launch dashboard
make dashboard            # http://localhost:8501

# 6. Unit tests
make test
```

---

## 📁 Project Structure

```
LLM_classifier_pipeline/
├── .github/workflows/llm_eval.yml   # 6-stage CI/CD pipeline
├── data/
│   ├── golden_dataset.json          # 105 curated QA pairs
│   └── schema.json                  # Dataset validation schema
├── eval/
│   ├── runner.py                    # Main orchestrator
│   ├── gates.py                     # SLA threshold enforcement
│   └── metrics/                     # hallucination · relevancy · faithfulness · latency · cost
├── pipeline/
│   ├── llm_client.py                # OpenAI / Anthropic / Ollama / Mock
│   ├── rag.py                       # Mock + ChromaDB retriever
│   └── config.py                    # Environment-driven config
├── dashboard/app.py                 # Streamlit metrics dashboard
├── scripts/
│   ├── merge_and_gate.py            # Shard merger + gate check
│   └── validate_dataset.py          # Schema validator
├── tests/                           # Gate & metric unit tests
├── results/                         # JSON results per CI run
├── .env.example
├── requirements.txt
└── Makefile
```

---

## 🔧 Configuration

All config via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | `openai` \| `anthropic` \| `ollama` \| `mock` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the chosen provider |
| `RAG_BACKEND` | `mock` | `mock` \| `chromadb` |
| `HALLUCINATION_THRESHOLD` | `0.05` | Max hallucination rate |
| `P95_LATENCY_MS` | `5000` | Max p95 latency (ms) |
| `MIN_ANSWER_RELEVANCY` | `0.70` | Min semantic relevancy score |
| `MIN_FAITHFULNESS` | `0.75` | Min faithfulness score |

---

## 🏃 GitHub Actions Setup

Add secrets under **Settings → Secrets → Actions** (only needed for real LLM runs):

| Secret | When Required |
|--------|---------------|
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | `LLM_PROVIDER=anthropic` |

> CI defaults to **mock mode** — no secrets needed to run the full pipeline.

Pipeline triggers on push/PR when `pipeline/**`, `data/**`, `eval/**`, `tests/**`, or `requirements.txt` change.

---

## 🗂️ Golden Dataset

105 QA pairs across 6 categories:

| Category | Count |
|----------|-------|
| `factual` | 70 |
| `reasoning` | 10 |
| `multi-hop` | 8 |
| `edge-case` | 8 |
| `out-of-scope` | 7 |
| `adversarial` | 2 |

Edit `data/golden_dataset.json` directly to expand the dataset.

---

## 🤝 Adding a New Metric

1. Create `eval/metrics/your_metric.py` with `compute_*` and `mean_*` functions
2. Import and call it in `eval/runner.py`
3. Add threshold logic in `eval/gates.py` if it should block merges
4. Add unit tests in `tests/test_metrics.py`

---

## 📄 License

MIT — see [LICENSE](LICENSE).
