"""Text protocol of one model turn, and deterministic rendering of tool results.

Generated turn (one of)::

    [Thinking]            ...                       [Thinking]   ...
    [Input Decomposition] - sub query-1：...        [Response]   ...
    [Action]              [{"name": ..., "arguments": {...}}, ...]

Rendering is a pure function of the steps so a :class:`~agentic_search.runtime.agent.Checkpoint` can
rebuild the exact context of any prefix (needed to splice flywheel sub-queries into a trajectory).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_search.core.types import Segment, Step, SubQuery

GENERATED_HEADERS = (Segment.THINKING, Segment.DECOMPOSITION, Segment.ACTION, Segment.RESPONSE)
_HEADER_RE = re.compile(
    r"^\s*\[(" + "|".join(re.escape(h.value) for h in GENERATED_HEADERS) + r")\]\s*", re.M
)
_SUBQ_RE = re.compile(r"^\s*-\s*sub\s*query[-_\s]*(\d+)\s*[:：]\s*(.+?)\s*$", re.I | re.M)
TOOL_RESULT_ROLE = "user"  # tool results are context, never trained; the same role is used for SFT export
STOP_SEQUENCES = ("[" + Segment.TOOL_RESULT.value + "]",)


@dataclass
class ParsedTurn:
    thought: str = ""
    decomposition: list[SubQuery] = field(default_factory=list)
    action: str | None = None
    response: str | None = None
    unknown_text: str = ""


def parse_turn(text: str) -> ParsedTurn:
    parts = _HEADER_RE.split(text)
    out = ParsedTurn(unknown_text=parts[0].strip())
    for header, body in zip(parts[1::2], parts[2::2]):
        body = body.strip()
        if header == Segment.THINKING.value:
            out.thought = (out.thought + "\n" + body).strip()
        elif header == Segment.DECOMPOSITION.value:
            out.decomposition = parse_sub_queries(body)
        elif header == Segment.ACTION.value:
            out.action = _strip_code_fence(body)
        elif header == Segment.RESPONSE.value:
            out.response = body
    return out


def parse_sub_queries(body: str) -> list[SubQuery]:
    return [SubQuery(id=f"sub_query_{m.group(1)}", query=m.group(2)) for m in _SUBQ_RE.finditer(body)]


def _strip_code_fence(body: str) -> str:
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.S)
    body = m.group(1) if m else body
    # tolerate a natural-language lead-in such as "调用社区搜索函数：" before the JSON payload
    start = min((i for i in (body.find("["), body.find("{")) if i >= 0), default=0)
    return body[start:].strip()


def render_decomposition(sub_queries: list[SubQuery]) -> str:
    return "\n".join(f"- sub query-{i}：{sq.query}" for i, sq in enumerate(sub_queries, start=1))


def render_prefix_with_decomposition(thought: str, sub_queries: list[SubQuery]) -> str:
    """Assistant prefix used when injecting externally mined sub-queries (data flywheel)."""
    return (
        f"[{Segment.THINKING.value}]\n{thought.strip()}\n"
        f"[{Segment.DECOMPOSITION.value}]\n{render_decomposition(sub_queries)}\n"
        f"[{Segment.ACTION.value}]\n"
    )


def render_tool_results(step: Step, snippet_chars: int = 160) -> str:
    lines = [f"[{Segment.TOOL_RESULT.value}]"]
    if step.action_parse_error:
        lines.append(f"动作解析失败：{step.action_parse_error}。请输出合法的 JSON 工具调用。")
        return "\n".join(lines)
    label = {sq.query: f"sub query-{i}" for i, sq in enumerate(step.sub_queries, start=1)}
    lines.append("已查询到以下相关信息：")
    for k, (call, res) in enumerate(zip(step.tool_calls, step.tool_results), start=1):
        q = call.arguments.get("query", "") if isinstance(call.arguments, dict) else ""
        tag = label.get(q, f"call-{k}")
        head = f"[{tag}] {res.name}(query={q!r})"
        if res.status == "error":
            lines.append(f"{head} - 检索服务暂时不可用")
        elif res.status == "invalid":
            lines.append(f"{head} - 参数不合法：{res.error}")
        elif not res.posts:
            lines.append(f"{head} - 无结果")
        for p in res.posts:
            lines.append(f"{head} - [Post {p.post_id}] {p.title}：{p.snippet(snippet_chars)}")
    return "\n".join(lines)


def render_messages(system_prompt: str, query: str, steps: list[Step]) -> list[dict[str, str]]:
    """Chat messages for the context *after* ``steps`` (all of which must be complete tool steps or
    the final step)."""
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    for s in steps:
        msgs.append({"role": "assistant", "content": s.raw_text})
        if s.is_tool_step:
            msgs.append({"role": TOOL_RESULT_ROLE, "content": render_tool_results(s)})
    return msgs
