"""Horizontal-action data flywheel: mine query decompositions from online search sessions.

Users rarely hit everything with one query; their reformulations within one information-seeking session
*are* a decomposition of the underlying need. Pipeline:

1. parse logs into ``(user_id, query, timestamp)``;
2. sessionise: consecutive queries of one user with gaps <= ``gap_s``;
3. map each query to a semantic cluster (embedding clustering) and replace it with the cluster's
   canonical query -- literal matching is too sparse for colloquial queries;
4. count 2-grams ``Q_A -> Q_B`` of consecutive canonical queries ("next_queries");
5. ``freq(A -> B) > X``: B is a sub-query of A (*how* to split); a query whose every successor is
   ``<= X`` is a no-split example (*when not* to split);
6. an LLM judge removes pairs without a logical relation (memes, trending noise).

Mined decompositions are spliced into trajectories with :meth:`Checkpoint.with_injected_decomposition`
and the rest of the trajectory is re-rolled by the LLM provider.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from agentic_search.core.text import Embedder, normalize_text
from agentic_search.rewards.judge import INJECTION_GUARD, Judge, parse_discrete_score
from agentic_search.taxonomy.funnel import run_funnel


@dataclass(frozen=True)
class LogRecord:
    user_id: str
    query: str
    ts: float


def parse_log_lines(lines: Iterable[str]) -> list[LogRecord]:
    """JSONL with ``userid``/``user_id``, ``query`` and ``timestamp``/``ts`` fields; bad lines are skipped."""
    out = []
    for line in lines:
        try:
            d = json.loads(line)
            out.append(
                LogRecord(
                    str(d.get("user_id", d.get("userid"))),
                    str(d["query"]).strip(),
                    float(d.get("ts", d.get("timestamp"))),
                )
            )
        except (ValueError, KeyError, TypeError):
            continue
    return [r for r in out if r.query]


def sessionize(records: Iterable[LogRecord], gap_s: float = 1800.0) -> list[list[LogRecord]]:
    by_user: dict[str, list[LogRecord]] = defaultdict(list)
    for r in records:
        by_user[r.user_id].append(r)
    sessions: list[list[LogRecord]] = []
    for recs in by_user.values():
        recs.sort(key=lambda r: r.ts)
        cur = [recs[0]]
        for prev, r in itertools.pairwise(recs):
            if r.ts - prev.ts > gap_s:
                sessions.append(cur)
                cur = []
            cur.append(r)
        sessions.append(cur)
    return sessions


def canonicalize_queries(
    queries: Sequence[str], embedder: Embedder, distance_threshold: float = 0.3
) -> dict[str, str]:
    """query -> canonical query of its semantic cluster (the most frequent surface form)."""
    res = run_funnel(queries, embedder, distance_threshold)
    canon = {m: c.representative for c in res.clusters for m in c.members}
    return {q: canon.get(normalize_text(q), normalize_text(q)) for q in set(queries)}


def bigram_counts(
    sessions: Sequence[Sequence[LogRecord]], canon: dict[str, str], once_per_session: bool = True
) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for s in sessions:
        seq = [canon.get(r.query, r.query) for r in s]
        seq = [q for i, q in enumerate(seq) if i == 0 or q != seq[i - 1]]  # drop immediate repeats
        pairs = list(itertools.pairwise(seq))
        counts.update(set(pairs) if once_per_session else pairs)
    return counts


@dataclass
class DecompositionMining:
    splits: dict[str, list[tuple[str, int]]] = field(default_factory=dict)  # main -> [(sub, freq)]
    no_split: set[str] = field(default_factory=set)

    def examples(self) -> list[dict]:
        """Training targets for the [Input Decomposition] step."""
        out = [{"query": a, "sub_queries": [b for b, _ in subs]} for a, subs in self.splits.items()]
        out += [{"query": q, "sub_queries": []} for q in sorted(self.no_split)]
        return out


def mine_decompositions(
    counts: Counter[tuple[str, str]], min_freq: int, max_subs: int = 6
) -> DecompositionMining:
    succ: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (a, b), f in counts.items():
        succ[a].append((b, f))
    res = DecompositionMining()
    for a, lst in succ.items():
        strong = sorted([(b, f) for b, f in lst if f > min_freq], key=lambda x: -x[1])
        if strong:
            res.splits[a] = strong[:max_subs]
        else:
            res.no_split.add(a)
    return res


PAIR_JUDGE_PROMPT = """判断用户在搜索“{a}”之后紧接着搜索“{b}”，二者是否存在合理的逻辑关联
（后者是前者需求的子问题或必要补充），而不是热点玩梗、随意跳转等无关联行为。{guard}
1.0 — 明确是子问题或必要补充
0.5 — 有一定关联但不适合作为拆解
0.0 — 无逻辑关联
只输出数字，不要解释。"""


def judge_filter(mining: DecompositionMining, judge: Judge, min_score: float = 1.0) -> DecompositionMining:
    out = DecompositionMining(no_split=set(mining.no_split))
    for a, subs in mining.splits.items():
        kept = [
            (b, f)
            for b, f in subs
            if (parse_discrete_score(judge(PAIR_JUDGE_PROMPT.format(a=a, b=b, guard=INJECTION_GUARD))) or 0.0)
            >= min_score
        ]
        if kept:
            out.splits[a] = kept
    return out
