# Makefile — Developer shortcuts for the LLM eval pipeline

PYTHON     := python
VENV       := .venv
VENV_PY    := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
DATASET    := data/golden_dataset.json
RESULTS    := results

.PHONY: help venv install lint test validate eval eval-mock dashboard clean

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

test:  ## Run unit tests with coverage
	$(VENV_PY) -m pytest tests/ -v --tb=short \
		--cov=eval --cov=pipeline --cov-report=term-missing

lint:  ## Run ruff linter
	$(VENV_PY) -m ruff check eval/ pipeline/ scripts/ dashboard/

eval:  ## Run full eval suite (real LLM — needs API key)
	@mkdir -p $(RESULTS)
	$(VENV_PY) -m eval.runner \
		--dataset $(DATASET) \
		--shard 0 --total-shards 1 \
		--output-dir $(RESULTS) \
		--fail-on-gate

eval-mock:  ## Run eval suite with mock LLM (no API key needed)
	@mkdir -p $(RESULTS)
	LLM_PROVIDER=mock RAG_BACKEND=mock \
	$(VENV_PY) -m eval.runner \
		--dataset $(DATASET) \
		--shard 0 --total-shards 1 \
		--output-dir $(RESULTS)

dashboard:  ## Launch the Streamlit dashboard
	$(VENV_PY) -m streamlit run dashboard/app.py

clean:  ## Remove venv and result files
	rm -rf $(VENV) $(RESULTS)/*.json __pycache__ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Cleaned"
