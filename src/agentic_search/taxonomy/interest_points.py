"""Interest-point extraction model (Qwen3-VL-32B: GPT-5.6 CoT distillation -> SFT v2 -> LUFFY-GRPO v3).

Output contract (after an optional ``<think>...</think>``)::

    {"summary": "内容概括", "interest_points": ["沈腾", "喜剧电影", "春节档"]}

RL reward = 0.6 * RLAIF (step 1: summary faithfulness + interest-point quality, judged by a strong model)
          + 0.4 * RLVR  (step 2: the structured output is machine-checkable).
Teacher demonstrations used as LUFFY off-policy samples must be human-verified full-score outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agentic_search.rewards.judge import INJECTION_GUARD, Judge, parse_discrete_score

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class ExtractionSpec:
    min_points: int = 1
    max_points: int = 12
    min_point_chars: int = 2
    max_point_chars: int = 20
    max_summary_chars: int = 300


def parse_extraction(text: str) -> dict[str, Any] | None:
    body = THINK_RE.sub("", text).strip()
    m = re.search(r"\{.*\}", body, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def rlvr_score(text: str, spec: ExtractionSpec | None = None) -> float:
    """Verifiable part: valid JSON, required fields/types, bounded counts and lengths, no duplicates."""
    spec = spec or ExtractionSpec()
    obj = parse_extraction(text)
    if obj is None or set(obj) != {"summary", "interest_points"}:
        return 0.0
    summary, points = obj["summary"], obj["interest_points"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > spec.max_summary_chars:
        return 0.0
    if not isinstance(points, list) or not spec.min_points <= len(points) <= spec.max_points:
        return 0.0
    if not all(
        isinstance(p, str) and spec.min_point_chars <= len(p.strip()) <= spec.max_point_chars for p in points
    ):
        return 0.0
    return 1.0 if len({p.strip() for p in points}) == len(points) else 0.0


RLAIF_PROMPT = """评估内容理解模型的输出。{guard}

原始内容：
{content}

模型输出：
{output}

请判断：内容概括是否忠实准确，兴趣点是否具体、可泛化、能代表用户会关注的兴趣（而不是无意义的词）。
评分标准：
1.0 — 概括准确且兴趣点合理完整
0.5 — 概括基本准确但兴趣点存在遗漏、过泛或冗余
0.0 — 概括失真或兴趣点明显不合理

只输出数字，不要解释。"""


def interest_point_reward(
    content: str, output: str, judge: Judge, rlaif_weight: float = 0.6, spec: ExtractionSpec | None = None
) -> dict[str, float]:
    vr = rlvr_score(output, spec)
    ai = 0.0
    if vr > 0:  # do not spend judge calls on unparseable outputs
        ai = (
            parse_discrete_score(
                judge(RLAIF_PROMPT.format(guard=INJECTION_GUARD, content=content, output=output))
            )
            or 0.0
        )
    return {"score": rlaif_weight * ai + (1 - rlaif_weight) * vr, "rlaif": ai, "rlvr": vr}
