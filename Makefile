.PHONY: help install install-dev test lint format ui cli evals compose-up compose-down clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-18s %s\n", $$1, $$2}'

install:        ## Install runtime deps + the package (editable)
	pip install -e .

install-dev:    ## Install dev deps
	pip install -r requirements-dev.txt
	pip install -e .

test:           ## Run the test suite
	PYTHONPATH=src pytest

lint:           ## Lint with ruff
	ruff check src tests scripts

format:         ## Auto-format with ruff
	ruff format src tests scripts
	ruff check --fix src tests scripts

ui:             ## Run the Streamlit UI locally
	PYTHONPATH=src streamlit run src/genai/ui/streamlit_app.py --server.port=8506 --client.toolbarMode=viewer

cli:            ## Example CLI run; pass TASK="..."
	PYTHONPATH=src python scripts/run_cli.py codegen "$(TASK)"

evals:          ## Run an eval suite; pass SUITE=path/to/suite.yaml [CASE=id] [MODEL=...]
	PYTHONPATH=src python scripts/run_evals.py --suite "$(SUITE)" \
	    $(if $(CASE),--case "$(CASE)",) $(if $(MODEL),--model "$(MODEL)",)

compose-up:     ## Build and start agent + sandbox containers
	docker compose up --build

compose-down:   ## Stop and remove containers
	docker compose down

clean:          ## Remove caches and generated workspaces
	rm -rf .pytest_cache .ruff_cache **/__pycache__
	find workspaces -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
