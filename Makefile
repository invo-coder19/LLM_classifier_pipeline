# Makefile — Developer shortcuts for the LLM eval pipeline
# On Windows, use: python -m pytest ... (not venv/bin/python)

PYTHON     := python
VENV       := .venv
DATASET    := data/golden_dataset.json
RESULTS    := results

# Detect OS and set venv python/pip paths
ifeq ($(OS),Windows_NT)
  VENV_PY  := $(VENV)/Scripts/python
  PIP      := $(VENV)/Scripts/pip
else
  VENV_PY  := $(VENV)/bin/python
  PIP      := $(VENV)/bin/pip
endif

.PHONY: help venv install lint fmt test validate eval eval-mock dashboard clean check-all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "✅ venv created at $(VENV)"
	@echo "   Activate with: source $(VENV)/bin/activate  (Linux/Mac)"
	@echo "                  $(VENV)\\Scripts\\activate  (Windows)"

install: venv  ## Install all dependencies into venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

validate:  ## Validate the golden dataset schema
	$(VENV_PY) scripts/validate_dataset.py

lint:  ## Run ruff linter (check only)
	$(VENV_PY) -m ruff check eval/ pipeline/ scripts/ tests/

fmt:  ## Auto-fix lint issues with ruff
	$(VENV_PY) -m ruff check --fix eval/ pipeline/ scripts/ tests/
	$(VENV_PY) -m ruff format eval/ pipeline/ scripts/ tests/

test:  ## Run unit tests with coverage
	PYTHONPATH=. $(VENV_PY) -m pytest tests/ -v --tb=short \
		--cov=eval --cov=pipeline --cov-report=term-missing

test-ci:  ## Run tests with XML coverage report (for CI)
	PYTHONPATH=. $(VENV_PY) -m pytest tests/ -v --tb=short \
		--cov=eval --cov=pipeline \
		--cov-report=xml --cov-report=term-missing

eval:  ## Run full eval suite (real LLM — needs API key)
	@mkdir -p $(RESULTS)
	PYTHONPATH=. $(VENV_PY) -m eval.runner \
		--dataset $(DATASET) \
		--shard 0 --total-shards 1 \
		--output-dir $(RESULTS) \
		--fail-on-gate

eval-mock:  ## Run eval suite with mock LLM (no API key needed)
	@mkdir -p $(RESULTS)
	PYTHONPATH=. LLM_PROVIDER=mock RAG_BACKEND=mock \
	$(VENV_PY) -m eval.runner \
		--dataset $(DATASET) \
		--shard 0 --total-shards 1 \
		--output-dir $(RESULTS)

dashboard:  ## Launch the Streamlit dashboard
	$(VENV_PY) -m streamlit run dashboard/app.py

check-all: validate lint test  ## Run validate + lint + tests (full local CI check)

clean:  ## Remove venv and result files
	rm -rf $(VENV) $(RESULTS)/*.json .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Cleaned"
