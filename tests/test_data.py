import json

import pytest

from agentic_search import demo
from agentic_search.core.text import HashingEmbedder
from agentic_search.data import (
    LogRecord,
    bigram_counts,
    canonicalize_queries,
    export_sft_jsonl,
    judge_filter,
    mine_decompositions,
    parse_log_lines,
    sessionize,
    to_chat_sample,
    tokenize_trajectory,
)
from agentic_search.rewards import CallableJudge
from agentic_search.runtime import AgentRunner, Checkpoint, ScriptedProvider


def _logs():
    recs = []
    t = 0.0
    for u in range(12):  # 12 users who split "成都旅游" into the same follow-ups (with paraphrases)
        follow = ["成都景点推荐", "成都美食攻略"] if u % 2 else ["成都景点 推荐", "成都 美食攻略"]
        for q in ["成都旅游", *follow]:
            recs.append(LogRecord(f"u{u}", q, t))
            t += 60
        t += 10_000  # session break
        recs.append(LogRecord(f"u{u}", "成都旅游", t))  # a later single-query session
    for u in range(3):  # noise: rare jump to an unrelated meme
        recs += [LogRecord(f"n{u}", "沈腾电影", 0.0), LogRecord(f"n{u}", "沈腾表情包", 30.0)]
    return recs


def test_parse_log_lines_skips_bad_lines():
    lines = [
        '{"userid": "u1", "query": "成都", "timestamp": 1}',
        "garbage",
        '{"user_id": "u2", "query": " ", "ts": 2}',
    ]
    assert parse_log_lines(lines) == [LogRecord("u1", "成都", 1.0)]


def test_sessionize_splits_on_gap():
    sessions = sessionize(_logs(), gap_s=1800)
    u0 = [s for s in sessions if s[0].user_id == "u0"]
    assert [len(s) for s in u0] == [3, 1]


def test_flywheel_mines_how_and_when_to_split():
    recs = _logs()
    canon = canonicalize_queries([r.query for r in recs], HashingEmbedder(), distance_threshold=0.3)
    assert canon["成都景点 推荐"] == canon["成都景点推荐"]  # paraphrases collapse onto one cluster query
    counts = bigram_counts(sessionize(recs), canon)
    mining = mine_decompositions(counts, min_freq=5)
    main = canon["成都旅游"]
    subs = [b for b, _ in mining.splits[main]]
    assert canon["成都景点推荐"] in subs
    assert canon["沈腾电影"] in mining.no_split  # successors only seen 3 times (< X) -> do not split
    examples = mining.examples()
    assert {"query": canon["沈腾电影"], "sub_queries": []} in examples


def test_judge_filter_drops_illogical_pairs():
    counts = bigram_counts(
        [[LogRecord("u", "a", 0), LogRecord("u", "b", 1)]] * 3
        + [[LogRecord("u", "a", 0), LogRecord("u", "meme", 1)]] * 3,
        {},
    )
    mining = mine_decompositions(counts, min_freq=2)
    judged = judge_filter(mining, CallableJudge(lambda p: "0" if "meme" in p else "1"))
    assert [b for b, _ in judged.splits["a"]] == ["b"]


async def _gold(registry, prompts):
    return await AgentRunner(demo.scripted_policy(), registry, prompts).run_once(
        "成都两日游，带老人，希望少走路"
    )


async def test_chat_sample_roles_follow_loss_mask(registry, prompts):
    sample = to_chat_sample(await _gold(registry, prompts))
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert sample["messages"][3]["content"].startswith("[Tool Result]")
    assert "[Thinking]" in sample["messages"][2]["content"]


async def test_tokenized_mask_and_step_ids(registry, prompts):
    traj = await _gold(registry, prompts)
    tok = tokenize_trajectory(traj, lambda role, text: [ord(ch) for ch in f"<{role}>{text}"])
    gen = [s for s, m in zip(tok.step_ids, tok.loss_mask) if m]
    assert set(gen) == {0, 1, 2}
    assert all(s == -1 for s, m in zip(tok.step_ids, tok.loss_mask) if not m)
    tool_text = "[Tool Result]"
    flat = "".join(chr(i) for i in tok.input_ids)
    pos = flat.index(tool_text)
    assert tok.loss_mask[pos] == 0  # tool results are never trained


async def test_export_drops_truncated_and_strips_hints(tmp_path, registry, prompts):
    runner = AgentRunner(demo.scripted_policy(), registry, prompts)
    hinted = await runner.run_once("成都两日游，带老人，希望少走路", extra_instructions="SECRET HINT")
    assert "SECRET HINT" in hinted.system_prompt
    truncated = await AgentRunner(ScriptedProvider(["[Thinking]\n...\n"]), registry, prompts).run_once("q")
    n = export_sft_jsonl([hinted, truncated], tmp_path / "sft.jsonl")
    lines = (tmp_path / "sft.jsonl").read_text().splitlines()
    assert n == 1 and len(lines) == 1
    assert "SECRET HINT" not in json.loads(lines[0])["messages"][0]["content"]


async def test_flywheel_injection_via_checkpoint(registry, prompts):
    traj = await _gold(registry, prompts)
    ckpt = Checkpoint.with_injected_decomposition(traj, 0, ["成都景点推荐", "成都美食攻略"])
    prefix_seen = []

    def provider(messages, prefix):
        prefix_seen.append(prefix)
        if prefix:
            return json.dumps(
                [{"name": "tag_search", "arguments": {"query": "成都景点推荐"}}], ensure_ascii=False
            )
        return demo.chengdu_policy_turns()[-1]

    new = await AgentRunner(ScriptedProvider(provider), registry, prompts).run_once(traj.query, resume=ckpt)
    assert prefix_seen[0] and "成都美食攻略" in prefix_seen[0]
    assert new.steps[0].tool_results[0].status == "ok"


def test_checkpoint_bounds():
    from agentic_search.core.types import Trajectory

    with pytest.raises(IndexError):
        Checkpoint.with_injected_decomposition(Trajectory("q"), 1, ["a"])
