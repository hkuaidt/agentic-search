"""Tool protocol, registry and the canonical action format.

Canonical action (what the policy emits inside ``[Action]``)::

    [{"name": "tag_search",       "arguments": {"query": "成都两日游 老人 少走路", "limit": 5}},
     {"name": "community_search", "arguments": {"query": "成都景点 步行距离", "limit": 5}}]

A list = one *horizontal* action (calls run in parallel). ``call_id`` is added by the runtime for log
lookup and is not part of what the model generates, so the format reward can require keys ==
{name, arguments} while logs still carry a call id.

A fan-out shorthand is also accepted (one call object naming both tools with a
``sub_queries`` list). :func:`normalize_action` expands it into atomic calls so rewards, logging and the
search proxy only ever see one shape.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import jsonschema

from agentic_search.core.types import ToolCall, ToolResult

ParameterCheck = Callable[[dict[str, Any]], bool]

SEARCH_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 128},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    "required": ["query"],
    "additionalProperties": False,
}


class Tool(ABC):
    name: str
    description: str
    schema: dict[str, Any] = SEARCH_ARGS_SCHEMA
    parameter_checks: tuple[ParameterCheck, ...] = ()

    @abstractmethod
    async def __call__(self, arguments: dict[str, Any], call_id: str = "") -> ToolResult: ...

    def function_spec(self) -> dict[str, Any]:
        """OpenAI / Qwen function-calling spec (also used for the VERL tool YAML)."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.schema},
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.function_spec() for t in self._tools.values()]

    def validate_arguments(self, name: str, arguments: Any) -> bool:
        """Schema + business checks (e.g. ``start_date <= end_date``) -- the format-reward contract."""
        tool = self._tools.get(name)
        if tool is None or not isinstance(arguments, dict):
            return False
        try:
            jsonschema.validate(arguments, tool.schema)
        except jsonschema.ValidationError:
            return False
        return all(check(arguments) for check in tool.parameter_checks)


class ActionParseError(ValueError):
    pass


def normalize_action(raw: str | dict[str, Any] | list[Any], default_limit: int = 5) -> list[ToolCall]:
    """Parse an ``[Action]`` payload into atomic :class:`ToolCall` objects.

    Raises :class:`ActionParseError` when the payload is not JSON or not a dict/list of dicts. Calls
    that parse but violate the schema are returned as-is; the format reward scores them.
    """
    obj: Any = raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ActionParseError(f"invalid JSON: {e.msg}") from e
    items = obj if isinstance(obj, list) else [obj]
    calls: list[ToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            raise ActionParseError("each call must be a JSON object")
        name, args = item.get("name"), item.get("arguments")
        if isinstance(name, list) and isinstance(args, dict) and isinstance(args.get("sub_queries"), list):
            limit = args.get("limit", default_limit)
            for sq in args["sub_queries"]:
                q = sq.get("query") if isinstance(sq, dict) else sq
                for tool_name in name:
                    atomic = {"name": tool_name, "arguments": {"query": q, "limit": limit}}
                    calls.append(ToolCall(name=str(tool_name), arguments=atomic["arguments"], raw=atomic))
            continue
        calls.append(
            ToolCall(
                name=name if isinstance(name, str) else "",
                arguments=args if isinstance(args, dict) else {},
                raw=item,  # the original (possibly malformed) call is what the format reward scores
            )
        )
    return calls
