import pytest

from agentic_search import demo


@pytest.fixture
def registry():
    return demo.build_registry()


@pytest.fixture
def prompts(registry):
    return demo.build_prompt_builder(registry)
