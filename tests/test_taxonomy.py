import pytest

from agentic_search.core.text import HashingEmbedder
from agentic_search.rewards import CallableJudge
from agentic_search.taxonomy import (
    BLANK_TAG,
    AdmissionController,
    BlankTagMonitor,
    ClusterInducer,
    EmbeddingMounter,
    InteractionWeights,
    TagTree,
    TaxonomyBuilder,
    coverage_test,
    exact_dedup,
    granularity_report,
    interest_point_reward,
    rlvr_score,
    run_funnel,
    uniform_cluster_sample,
)

EMB = HashingEmbedder()


def test_tree_paths_and_l3_to_l2_lookup():
    tree = TagTree.from_paths(
        [("旅行", "国内游", "成都旅行"), ("旅行", "国内游", "北京旅行"), ("旅行", "出境游", "日本旅行")]
    )
    assert tree.path("日本旅行") == ("旅行", "出境游", "日本旅行")
    assert sorted(tree.l3_under_l1("旅行")) == ["北京旅行", "成都旅行", "日本旅行"]
    assert tree.stats() == {"L1": 1, "L2": 2, "L3": 3}
    with pytest.raises(ValueError):
        tree.add("成都旅行", 3, "国内游")  # names are globally unique
    with pytest.raises(ValueError):
        tree.add("x", 3, "旅行")  # wrong parent level


def test_tree_roundtrip(tmp_path):
    tree = TagTree.from_paths([("影视综艺", "电影", "喜剧电影")])
    tree.save(tmp_path / "t.json")
    assert TagTree.load(tmp_path / "t.json").path("喜剧电影") == ("影视综艺", "电影", "喜剧电影")


def test_exact_dedup_normalises():
    counts = exact_dedup(["沈腾", " 沈腾", "ＳＨＥＮ", "shen", "", "王者荣耀"])
    assert counts["沈腾"] == 2 and counts["shen"] == 2 and "" not in counts


def test_funnel_merges_synonyms_and_keeps_frequent_surface_form():
    raw = (
        ["沈腾"] * 5
        + ["沈腾电影"] * 2
        + ["沈腾喜剧人"]
        + ["王者荣耀"] * 3
        + ["王者荣耀攻略"]
        + ["成都火锅"] * 2
    )
    res = run_funnel(raw, EMB, distance_threshold=0.6)
    reps = {c.representative: c for c in res.clusters}
    assert "沈腾" in reps and set(reps["沈腾"].members) >= {"沈腾", "沈腾电影"}
    assert reps["沈腾"].count >= 7
    assert res.stats["raw"] == len(raw) and res.stats["unique"] == 6
    assert res.stats["clusters"] < res.stats["unique"]


def test_funnel_leader_method_and_reviewer():
    raw = ["成都火锅", "成都火锅推荐", "北京烤鸭", "北京烤鸭店"]
    res = run_funnel(
        raw, EMB, distance_threshold=0.6, method="leader", reviewer=lambda c: "北京" not in c.representative
    )
    assert all("北京" not in c.representative for c in res.clusters)
    assert res.stats["effective"] < res.stats["clusters"]


def test_uniform_cluster_sample_respects_budget_and_diversity():
    pts = [f"成都火锅{i}" for i in range(60)] + [f"王者荣耀英雄{i}" for i in range(6)]
    sample = uniform_cluster_sample(pts, EMB, budget=12, n_clusters=4)
    assert len(sample) == 12 and len(set(sample)) == 12
    assert any("王者" in s for s in sample)  # the minority topic survives down-sampling


def test_builder_end_to_end_with_reflection_retry():
    points = [
        "成都火锅",
        "成都串串",
        "成都冒菜",
        "重庆火锅",
        "重庆小面",
        "重庆酸辣粉",
        "王者荣耀",
        "王者荣耀出装",
        "王者荣耀排位",
        "英雄联盟",
        "英雄联盟排位",
        "英雄联盟皮肤",
    ]
    calls = {"n": 0}

    class FlakyReflector:
        def review(self, parent, pts, children):
            calls["n"] += 1
            return (calls["n"] > 1, "覆盖不全")

    builder = TaxonomyBuilder(
        EMB,
        EmbeddingMounter(
            EMB,
            descriptions={
                "饮食餐饮": "火锅 小面 串串 冒菜 酸辣粉 美食",
                "游戏娱乐": "王者荣耀 英雄联盟 游戏",
            },
        ),
        [ClusterInducer(EMB, "gpt"), ClusterInducer(EMB, "claude")],
        FlakyReflector(),
    )
    tree, report = builder.build(["饮食餐饮", "游戏娱乐"], points)
    assert tree.stats()["L2"] >= 2 and tree.stats()["L3"] >= 2
    for l3 in tree.level(3):
        assert tree.path(l3)[0] in {"饮食餐饮", "游戏娱乐"}
    assert report.mounted["饮食餐饮"] == 6 and report.mounted["游戏娱乐"] == 6
    assert calls["n"] >= 2  # the first proposal was rejected and regenerated
    assert coverage_test(["成都火锅"], tree, EmbeddingMounter(EMB)) == 1.0


def test_granularity_report():
    assert granularity_report({"a": 900, "b": 3, "c": 50}, min_points=10, max_points=500) == {
        "split": ["a"],
        "merge": ["b"],
    }


def test_admission_threshold_is_strict_and_incremental_fires_once():
    ctl = AdmissionController(tau=100.0, weights=InteractionWeights(1, 0, 0, 0))
    assert not ctl.admits({"likes": 100}) and ctl.admits({"likes": 101})
    assert ctl.batch_stock([("old1", {"likes": 500}), ("old2", {"likes": 5})]) == ["old1"]
    assert ctl.rescore_incremental([("new1", {"likes": 10})]) == []  # slow start: not locked out ...
    assert ctl.rescore_incremental([("new1", {"likes": 150})]) == ["new1"]  # ... enqueued on first crossing
    assert ctl.rescore_incremental([("new1", {"likes": 999})]) == []  # and only once
    assert ctl.pin_official(["official", "old1"]) == ["official"]
    with pytest.raises(ValueError):
        InteractionWeights(-1, 0, 0, 0)


def test_blank_tag_monitor_triggers_taxonomy_extension():
    mon = BlankTagMonitor(trigger_count=3)
    assert mon.observe("p0", "影视综艺", "喜剧电影") is None
    assert mon.observe("p1", "影视综艺", BLANK_TAG) is None
    assert mon.observe("p2", "影视综艺", BLANK_TAG) is None
    assert mon.observe("p3", "影视综艺", BLANK_TAG) == ["p1", "p2", "p3"]
    assert mon.buckets["影视综艺"] == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('<think>先概括</think>{"summary": "沈腾新片口碑", "interest_points": ["沈腾", "喜剧电影"]}', 1.0),
        ('{"summary": "x", "interest_points": ["沈腾", "沈腾"]}', 0.0),  # duplicates
        ('{"summary": "x", "interest_points": []}', 0.0),
        ('{"summary": "x", "interest_points": ["沈腾"], "extra": 1}', 0.0),
        ("not json", 0.0),
    ],
)
def test_interest_point_rlvr(text, expected):
    assert rlvr_score(text) == expected


def test_interest_point_reward_mixes_rlaif_and_rlvr():
    good = '{"summary": "沈腾新片口碑", "interest_points": ["沈腾", "喜剧电影"]}'
    out = interest_point_reward("...", good, CallableJudge(lambda p: "0.5"))
    assert out["score"] == pytest.approx(0.6 * 0.5 + 0.4 * 1.0)
    calls = []
    out = interest_point_reward("...", "bad", CallableJudge(lambda p: calls.append(p) or "1"))
    assert out["score"] == 0.0 and not calls  # judge is skipped for unparseable outputs
