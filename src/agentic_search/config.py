"""YAML config -> runtime components. One file (``configs/agentic_search.yaml``) drives the online runtime,
the RL rollout workers and evaluation, so the three always agree on tools, budgets and reward weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentic_search.rewards.composite import AgenticSearchReward, RewardWeights
from agentic_search.rewards.judge import OpenAICompatibleJudge
from agentic_search.runtime.agent import AgentRunner, RuntimeConfig, proxy_executor
from agentic_search.runtime.llm import OpenAICompatibleProvider
from agentic_search.runtime.prompt import PromptBuilder
from agentic_search.tools.base import Tool, ToolRegistry
from agentic_search.tools.proxy import Backend, SearchProxy, ToolLimits
from agentic_search.tools.search import CommunitySearch, HTTPSearchBackend


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_runtime_config(cfg: dict[str, Any]) -> RuntimeConfig:
    r = cfg.get("runtime", {})
    return RuntimeConfig(
        max_tool_rounds=r.get("max_tool_rounds", 2),
        max_calls_per_round=r.get("max_calls_per_round", 8),
        default_limit=r.get("default_limit", 5),
    )


class _RemoteTool(CommunitySearch):
    """Any HTTP search backend exposed under a given tool name (tag_search shares the signature)."""

    def __init__(self, name: str, description: str, backend: HTTPSearchBackend):
        super().__init__(backend)
        self.name, self.description = name, description


def build_registry(cfg: dict[str, Any]) -> ToolRegistry:
    tools: list[Tool] = []
    for name, spec in cfg["tools"].items():
        tools.append(_RemoteTool(name, spec.get("description", name), HTTPSearchBackend(spec["url"])))
    return ToolRegistry(tools)


def build_proxy(cfg: dict[str, Any]) -> SearchProxy:
    p = cfg.get("proxy", {})
    backends: dict[str, Backend] = {
        name: HTTPSearchBackend(spec["url"]) for name, spec in cfg["tools"].items()
    }
    limits = {name: ToolLimits(**lim) for name, lim in p.get("limits", {}).items()}
    return SearchProxy(
        backends,
        limits,
        queue_size=p.get("queue_size", 1024),
        num_workers=p.get("num_workers", 32),
        cache_ttl_s=p.get("cache_ttl_s", 600.0),
        index_version=str(p.get("index_version", "live")),
        max_queue_wait_s=p.get("max_queue_wait_s", 30.0),
    )


def _judge(spec: dict[str, Any]) -> OpenAICompatibleJudge:
    return OpenAICompatibleJudge(spec["base_url"], spec["model"], api_key=spec.get("api_key", "EMPTY"))


def build_reward(cfg: dict[str, Any], registry: ToolRegistry) -> AgenticSearchReward:
    r = cfg["reward"]
    return AgenticSearchReward(
        registry=registry,
        query_judge=_judge(r["query_judge"]),
        outcome_judges={dim: _judge(spec) for dim, spec in r["outcome_judges"].items()},
        weights=RewardWeights(**r.get("weights", {})),
        outcome_dim_weights=r.get("outcome_dim_weights"),
    )


def build_training_components(cfg: dict[str, Any]) -> tuple[AgentRunner, AgenticSearchReward]:
    """Runner whose tool calls go through the training search proxy, plus the hybrid reward.

    The LLM provider is only used outside VERL (VERL's agent loop generates through its own rollout
    server); it points at ``policy.base_url`` when present.
    """
    registry = build_registry(cfg)
    rt = build_runtime_config(cfg)
    prompts = PromptBuilder(
        registry, k_shots=cfg.get("runtime", {}).get("k_shots", 2), max_rounds=rt.max_tool_rounds
    )
    pol = cfg.get("policy", {})
    llm = OpenAICompatibleProvider(
        pol.get("base_url", "http://localhost:8000/v1"), pol.get("model", "policy")
    )
    runner = AgentRunner(llm, registry, prompts, executor=proxy_executor(build_proxy(cfg)), config=rt)
    return runner, build_reward(cfg, registry)
