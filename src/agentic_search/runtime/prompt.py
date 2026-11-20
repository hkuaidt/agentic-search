"""System prompt with dynamic (query-dependent) few-shot.

Layout: role / task / workflow / tools / constraints / examples / output format.
Examples are 50-60% of prompt tokens, which dominates prefill and therefore time-to-first-token, so only
the ``k`` examples closest to the current query are inserted instead of a fixed cross-category set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agentic_search.core.text import Embedder, HashingEmbedder
from agentic_search.tools.base import ToolRegistry

DEFAULT_ROLE = "你是一个社区搜索助手。"
DEFAULT_TASK = (
    "我会给你一个问题。请检索社区中的相关帖子和讨论，结合内容的相关性、时效性和可信度，整理不同观点，"
    "并给出有依据的回答。"
)
DEFAULT_WORKFLOW = (
    "1. 理解需求；复杂需求在一轮内拆成多个互补的子查询（横向 Action），简单明确的需求直接检索。\n"
    "2. 同时使用 tag_search（精选标签内容）与 community_search（全社区检索）；两者冲突时优先 tag_search。\n"
    "3. 若证据存在盲区、冲突或需要核实时效信息，基于已有结果再补查一轮（纵向 Action），"
    "并在思考中写明“根据第 x 步”所依据的结果。\n"
    "4. 证据充分即停止检索，组织回答。"
)
DEFAULT_CONSTRAINTS = (
    "- 最多进行 {max_rounds} 轮检索。\n"
    "- 回答中的关键结论需标注来源，格式为 [结论](ref: 帖子ID)，只能引用检索到的帖子。\n"
    "- 区分客观事实与个人体验；对来源不足、相互冲突或可能过时的信息作出说明。\n"
    "- 不要把宣传热度或少量评价概括为社区共识。"
)
DEFAULT_OUTPUT_FORMAT = (
    "每一轮按如下格式输出：\n"
    "[Thinking]\n<思考>\n[Input Decomposition]\n- sub query-1：<子查询>\n...\n"
    '[Action]\n[{"name": "tag_search", "arguments": {"query": "...", "limit": 5}}, ...]\n'
    "当证据充分时输出：\n[Thinking]\n<思考>\n[Response]\n<最终回答>"
)


@dataclass
class FewShotExample:
    query: str
    demonstration: str  # a full gold trajectory rendered in the turn protocol
    category: str = ""


@dataclass
class PromptBuilder:
    registry: ToolRegistry
    examples: list[FewShotExample] = field(default_factory=list)
    embedder: Embedder = field(default_factory=HashingEmbedder)
    k_shots: int = 2
    max_rounds: int = 2
    role: str = DEFAULT_ROLE
    task: str = DEFAULT_TASK
    workflow: str = DEFAULT_WORKFLOW
    constraints: str = DEFAULT_CONSTRAINTS
    output_format: str = DEFAULT_OUTPUT_FORMAT

    def __post_init__(self) -> None:
        self._example_vecs = (
            self.embedder.encode([e.query for e in self.examples]) if self.examples else np.zeros((0, 1))
        )

    def select_examples(self, query: str) -> list[FewShotExample]:
        if not self.examples or self.k_shots <= 0:
            return []
        sims = self._example_vecs @ self.embedder.encode([query])[0]
        return [self.examples[i] for i in np.argsort(-sims)[: self.k_shots]]

    def build(self, query: str) -> str:
        tools = "\n".join(
            f"- {s['function']['name']}(query, limit)：{s['function']['description']}"
            for s in self.registry.specs()
        )
        shots = self.select_examples(query)
        parts = [
            f"# 角色\n{self.role}",
            f"# 任务\n{self.task}",
            f"# 工作流\n{self.workflow}",
            f"# 工具\n{tools}",
            f"# 约束\n{self.constraints.format(max_rounds=self.max_rounds)}",
        ]
        if shots:
            parts.append("# 案例\n" + "\n\n".join(f"## 用户：{e.query}\n{e.demonstration}" for e in shots))
        parts.append(f"# 输出格式\n{self.output_format}")
        return "\n\n".join(parts)
