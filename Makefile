.PHONY: help uv deps lock lint ruff format mypy test test-cov test-llm clean clean-build clean-pyc clean-test
.DEFAULT_GOAL := help
APP_PATH := ruts_adapt
TESTS_PATH := tests

help: ## Show the list of commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

uv: ## Check that uv is installed
	@which uv >/dev/null 2>&1 || { \
		echo "uv is not installed. Run 'curl -LsSf https://astral.sh/uv/install.sh | sh' or 'brew install uv'"; \
		exit 1; \
	}

deps: uv ## Install dependencies
ifeq ($(MODE), ci)
	uv sync --locked --all-groups
else
	uv sync --all-groups
endif

lock: uv ## Upgrade the lock file to the latest dependency versions
	uv lock --upgrade

lint: ruff mypy ## Run all code checks

ruff: deps ## Check and format the code with ruff
ifeq ($(MODE), ci)
	uv run ruff check $(APP_PATH) $(TESTS_PATH)
	uv run ruff format $(APP_PATH) $(TESTS_PATH) --check
else
	uv run ruff check $(APP_PATH) $(TESTS_PATH) --fix
	uv run ruff format $(APP_PATH) $(TESTS_PATH)
endif

format: deps ## Format the code
	uv run ruff format $(APP_PATH) $(TESTS_PATH)

mypy: deps ## Check types with mypy
	uv run mypy

test: deps ## Run the tests
	uv run pytest

test-cov: deps ## Run the tests with a coverage threshold
	uv run pytest --cov $(APP_PATH)

test-llm: deps ## Run live tests that call the model API (costs money; needs a key); ARGS selects tests
	RUTS_ADAPT_LIVE=1 uv run pytest -m llm $(ARGS)

clean: clean-build clean-pyc clean-test ## Remove all artifacts
	rm -f .coverage coverage.xml

clean-build: ## Remove build artifacts
	rm -fr build/ dist/ .eggs/ target/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove bytecode artifacts
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

clean-test: ## Remove test and lint caches
	rm -fr .pytest_cache .mypy_cache .ruff_cache
