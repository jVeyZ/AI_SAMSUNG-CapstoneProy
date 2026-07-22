.PHONY: help bootstrap setup-venv install install-dev setup data data-tomato data-rice data-orange \
        train train-crop predict serve app \
        test test-unit test-e2e lint format \
        android-test android-apk android \
        clean clean-models clean-data clean-results clean-all clean-venv clean-pyc

PYTHON ?= python3
VENV ?= .venv
BIN = $(VENV)/bin

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Setup & dependencies
# ──────────────────────────────────────────────

bootstrap: setup-venv install install-dev install-pkg setup ## Create venv, install deps + create .env

setup-venv: ## Create virtual environment
	@test -d $(VENV) || ($(PYTHON) -m venv $(VENV) && echo "Created venv in $(VENV)") || echo "venv already exists"

install: setup-venv ## Install runtime dependencies
	$(BIN)/python -m pip install -r requirements.txt

install-dev: setup-venv ## Install dev/test dependencies
	$(BIN)/python -m pip install -r requirements-dev.txt

install-pkg: setup-venv ## Install cropguard package in editable mode
	$(BIN)/python -m pip install -e .

setup: ## Create .env file with blank env vars
	@test -f .env || (printf '# CropGuard environment variables\n\n# AI follow-up provider (groq, gemini, opencode)\nCROPGUARD_AI_PROVIDER=\n\n# API keys (fill in the one matching your provider)\nGROQ_API_KEY=\nGEMINI_API_KEY=\nOPENCODE_API_KEY=\n\n# Optional: custom models directory\n# CROPGUARD_MODELS_DIR=models\n' > .env && echo "Created .env") || echo ".env already exists"

data: ## Download datasets for all crops
	$(BIN)/python -m cropguard.setup

data-tomato: ## Download tomato dataset only
	$(BIN)/python -m cropguard.setup --tomato

data-rice: ## Download rice dataset only
	$(BIN)/python -m cropguard.setup --rice

data-orange: ## Download orange dataset (requires manual zip)
	$(BIN)/python -m cropguard.setup --orange

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

train: ## Train all crops
	$(BIN)/python -m cropguard.train

train-tomato: ## Train tomato model
	$(BIN)/python -m cropguard.train --crop tomato

train-rice: ## Train rice model
	$(BIN)/python -m cropguard.train --crop rice

train-orange: ## Train orange model
	$(BIN)/python -m cropguard.train --crop orange

# ──────────────────────────────────────────────
# Inference & serving
# ──────────────────────────────────────────────

predict: ## Predict from image (usage: make predict IMG=path CROP=Crop)
	$(BIN)/python -m cropguard.predict_worker $(IMG) $(CROP)

app: ## Launch Streamlit web app
	$(BIN)/streamlit run src/cropguard/app.py

serve: ## Launch FastAPI backend (port 8000)
	$(BIN)/uvicorn cropguard.server:app --port 8000

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────

test: test-unit test-e2e ## Run all tests

test-unit: ## Run unit tests
	$(BIN)/python -m pytest tests/unit -v

test-e2e: ## Run end-to-end tests
	$(BIN)/python -m pytest tests/e2e -v

# ──────────────────────────────────────────────
# Linting & formatting
# ──────────────────────────────────────────────

lint: ## Run linter (ruff)
	$(BIN)/ruff check src/ tests/

format: ## Auto-format code
	$(BIN)/ruff format src/ tests/

# ──────────────────────────────────────────────
# Android
# ──────────────────────────────────────────────

android-test: ## Run Android unit tests
	bash ./gradlew testDebugUnitTest

android-apk: ## Build debug APK
	bash ./gradlew assembleDebug

android: android-test android-apk ## Run Android tests + build APK

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

clean: clean-models clean-results ## Remove generated models and results

clean-models: ## Remove trained models
	rm -rf models/

clean-results: ## Remove training results/figures
	rm -rf results/

clean-data: ## Remove downloaded datasets
	rm -rf data/

clean-all: clean-models clean-results clean-data clean-venv ## Remove all generated artifacts

clean-venv: ## Remove virtual environment
	rm -rf $(VENV)

clean-pyc: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
