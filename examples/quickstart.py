"""Library-level walkthrough on the offline toy world (CPU, no network).

    python examples/quickstart.py

1. run the agent (horizontal round -> vertical verification -> cited answer)
2. score it with the hybrid reward and the RaR rubric
3. export it as an SFT sample with the loss mask of PROPOSAL §4.2(3)
"""

import asyncio
import json
import tempfile
from pathlib import Path

from agentic_search import demo
from agentic_search.data import export_sft_jsonl
from agentic_search.eval import P0Constraint, RaREvaluator, RaRTask
from agentic_search.rewards import AgenticSearchReward, CallableJudge
from agentic_search.runtime import AgentRunner, render_tool_results


async def main() -> None:
    registry = demo.build_registry()
    runner = AgentRunner(demo.scripted_policy(), registry, demo.build_prompt_builder(registry))
    query = "成都两日游，带老人，每天步行不超过 3 公里"
    traj = await runner.run_once(query)

    for step in traj.steps:
        if step.is_tool_step:
            print(
                f"--- step {step.index + 1}: {len(step.tool_calls)} parallel calls, held posts before = "
                f"{sorted(step.state_post_ids)}"
            )
            print(render_tool_results(step))
    print("--- answer\n" + (traj.final_response or ""))

    reward = AgenticSearchReward(registry, demo.rule_query_judge(), demo.rule_outcome_judges())
    print("\nreward:", json.dumps({k: round(v, 3) for k, v in reward(traj).items()}, ensure_ascii=False))

    judge = CallableJudge(lambda p: "0" if "硬性要求" in p else "1")  # stand-in for an LLM judge
    task = RaRTask("demo-1", query, p0=[P0Constraint("walk<=3km", "每天步行不超过 3 公里")])
    res = RaREvaluator(registry, judge).evaluate(task, traj)
    print(f"RaR: F={res.F} Q={res.Q} A={res.A} S_quality={res.S_quality} G_P0={res.G_P0} pass={res.passed}")

    out = Path(tempfile.mkdtemp()) / "sft.jsonl"
    export_sft_jsonl([traj], out)
    roles = [m["role"] for m in json.loads(out.read_text().splitlines()[0])["messages"]]
    print(f"SFT sample roles (assistant = trained, others masked): {roles}")


if __name__ == "__main__":
    asyncio.run(main())
