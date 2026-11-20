.PHONY: install lint format typecheck test demo check

install:
	uv venv --python 3.12 && uv pip install -e ".[dev]"

lint:
	uv run ruff check src tests examples && uv run ruff format --check src tests examples

format:
	uv run ruff format src tests examples && uv run ruff check --fix src tests examples

typecheck:
	uv run mypy src

test:
	uv run pytest

demo:
	uv run agentic-search demo

check: lint typecheck test
