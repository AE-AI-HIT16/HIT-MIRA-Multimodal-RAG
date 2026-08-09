.PHONY: help install dev lock serve test integration-tests lint format check

help:
	@echo 'Targets:'
	@echo '  install             Sync runtime dependencies with uv'
	@echo '  dev                 Sync project + dev dependencies with uv'
	@echo '  lock                Regenerate uv.lock from pyproject.toml'
	@echo '  serve               Start the local LangGraph dev server'
	@echo '  test                Run tests if ChatBot/tests exists'
	@echo '  integration-tests   Run integration tests if present'
	@echo '  lint                Run Ruff checks'
	@echo '  format              Format with Ruff'
	@echo '  check               Run lint and tests'

install:
	uv sync --no-dev

dev:
	uv sync

lock:
	uv lock

serve:
	uv run langgraph dev

test:
	@if [ -d tests ]; then uv run python -m pytest tests -q; else echo 'No tests directory found; skipping tests.'; fi

integration-tests:
	@if [ -d tests/integration_tests ]; then uv run python -m pytest tests/integration_tests -q; else echo 'No integration tests found; skipping integration tests.'; fi

lint:
	uv run python -m ruff check src

format:
	uv run python -m ruff format src

check: lint test
