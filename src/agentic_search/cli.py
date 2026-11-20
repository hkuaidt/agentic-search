"""``agentic-search`` command line.

agentic-search demo                      # offline end-to-end: rollouts -> hybrid reward -> 3-level advantages
agentic-search gigpo-example             # reproduce the GiGPO three-trajectory worked example
agentic-search mine-flywheel LOGS.jsonl  # session logs -> query decompositions (JSONL)
agentic-search build-rl-data Q.jsonl OUT.parquet   # queries -> VERL RL dataset (needs pandas/pyarrow)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from agentic_search import demo
from agentic_search.algos import gigpo_step_advantage, group_normalized_advantage
from agentic_search.rewards import AgenticSearchReward
from agentic_search.runtime import AgentRunner, ScriptedProvider


def _drifting_turns() -> list[str]:
    t1 = demo.chengdu_policy_turns()[0]
    t2 = (
        "[Thinking]\n再看看别的热门内容。\n[Input Decomposition]\n- sub query-1：沈腾 综艺\n"
        '[Action]\n[{"name": "community_search", "arguments": {"query": "沈腾 综艺 名场面", "limit": 3}}]'
    )
    answer = "[Thinking]\n直接回答。\n[Response]\n成都很好玩，推荐宽窄巷子。"
    return [t1, t2, answer]


async def _demo() -> None:
    registry = demo.build_registry()
    prompts = demo.build_prompt_builder(registry)
    reward = AgenticSearchReward(registry, demo.rule_query_judge(), demo.rule_outcome_judges())
    query = "成都两日游，带老人，希望少走路"
    policies = {
        "horizontal+vertical": ScriptedProvider(demo.chengdu_policy_turns(vertical=True)),
        "horizontal only": ScriptedProvider(demo.chengdu_policy_turns(vertical=False)),
        "drifting": ScriptedProvider(_drifting_turns()),
    }
    trajs, rows = [], []
    for name, llm in policies.items():
        t = await AgentRunner(llm, registry, prompts).run_once(query)
        r = reward(t)
        trajs.append(t)
        rows.append((name, t, r))

    rewards = [r["score"] for _, _, r in rows]
    traj_adv = group_normalized_advantage(rewards, [0] * len(rows))
    step_adv = gigpo_step_advantage(
        rewards, [0] * len(rows), [[s.anchor_key() for s in t.steps] for t in trajs]
    )

    print(f"query: {query}\n")
    print(
        f"{'policy':<22}{'rounds':>7}{'format':>8}{'query':>7}{'outcome':>9}{'R':>7}{'A^E':>8}   A^S per step"
    )
    for (name, t, r), ae, a_s in zip(rows, traj_adv.tolist(), step_adv):
        steps = " ".join(f"{x:+.2f}" for x in a_s)
        print(
            f"{name:<22}{t.num_tool_rounds:>7}{r['format']:>8.2f}{r['query']:>7.2f}{r['outcome']:>9.2f}"
            f"{r['score']:>7.3f}{ae:>+8.2f}   [{steps}]"
        )
    print("\nStep 0 shares the empty anchor state across all three rollouts, so GiGPO compares the")
    print("first-round queries directly; later steps only get a signal when their held-post sets coincide.")


def _gigpo_example() -> None:
    keys = [
        ["s0", "A", "x1", "x2", "x3", "B", "x4", "R1"],
        ["s0", "A", "B", "y1", "R2"],
        ["A", "B", "R3"],
    ]
    adv = gigpo_step_advantage([1.0, 1.0, 1.0], [0, 0, 0], keys, gamma=0.9, eps=0.0)
    print("node  traj  steps_to_R  Q=0.9^k   A^S")
    for node in ("A", "B"):
        for i, k in enumerate(keys):
            t = k.index(node)
            dist = len(k) - (t + 1)
            print(f"{node:>4}  {i + 1:>4}  {dist:>10}  {0.9**dist:.4f}  {adv[i][t]:+.3f}")


def _mine_flywheel(args: argparse.Namespace) -> None:
    from agentic_search.core.text import HashingEmbedder
    from agentic_search.data import (
        bigram_counts,
        canonicalize_queries,
        mine_decompositions,
        parse_log_lines,
        sessionize,
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    recs = parse_log_lines(Path(args.logs).read_text(encoding="utf-8").splitlines())
    canon = canonicalize_queries([r.query for r in recs], HashingEmbedder(), cfg.get("cluster_distance", 0.3))
    counts = bigram_counts(
        sessionize(recs, cfg.get("session_gap_s", 1800)), canon, cfg.get("once_per_session", True)
    )
    mining = mine_decompositions(counts, args.min_freq or cfg.get("min_freq", 50), cfg.get("max_subs", 6))
    out = Path(args.out)
    out.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in mining.examples()) + "\n", encoding="utf-8"
    )
    print(
        f"{len(recs)} records -> {len(mining.splits)} split patterns, "
        f"{len(mining.no_split)} no-split -> {out}"
    )


def _build_rl_data(args: argparse.Namespace) -> None:
    try:
        import pandas as pd
    except ImportError:
        sys.exit("build-rl-data needs pandas + pyarrow (installed with the [train] extra)")
    rows = []
    for i, line in enumerate(Path(args.queries).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(
            {
                "data_source": d.get("source", "agentic_search"),
                "agent_name": "agentic_search",
                "prompt": [{"role": "user", "content": d["query"]}],
                "reward_model": {"style": "agent_loop", "ground_truth": d.get("rubric", "")},
                "extra_info": {"index": i, "query": d["query"], "category": d.get("category", "")},
            }
        )
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {len(rows)} prompts -> {args.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agentic-search")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="offline end-to-end demo on a toy corpus")
    sub.add_parser("gigpo-example", help="reproduce the GiGPO worked example")
    f = sub.add_parser("mine-flywheel", help="mine query decompositions from search logs")
    f.add_argument("logs")
    f.add_argument("--out", default="decompositions.jsonl")
    f.add_argument("--min-freq", type=int, default=None)
    f.add_argument("--config", default=None)
    r = sub.add_parser("build-rl-data", help="queries JSONL -> VERL parquet")
    r.add_argument("queries")
    r.add_argument("out")
    args = p.parse_args(argv)
    if args.cmd == "demo":
        asyncio.run(_demo())
    elif args.cmd == "gigpo-example":
        _gigpo_example()
    elif args.cmd == "mine-flywheel":
        _mine_flywheel(args)
    elif args.cmd == "build-rl-data":
        _build_rl_data(args)


if __name__ == "__main__":
    main()
