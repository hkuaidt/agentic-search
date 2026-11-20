"""A small, fully offline world (corpus, tag tree, tools, scripted policy, rule-based judges).

Used by ``agentic-search demo``, the examples and the test-suite so the whole pipeline -- dual-supply
tools, horizontal + vertical ReAct, hybrid reward, GiGPO/OPD advantages -- runs on a laptop CPU.
"""

from __future__ import annotations

import json
import re

from agentic_search.core.types import Post
from agentic_search.rewards.judge import CallableJudge
from agentic_search.runtime.llm import Messages, ScriptedProvider
from agentic_search.runtime.prompt import FewShotExample, PromptBuilder
from agentic_search.taxonomy.tree import TagTree
from agentic_search.tools.base import ToolRegistry
from agentic_search.tools.search import (
    CascadeTagRouter,
    CommunitySearch,
    InMemoryCommunityBackend,
    TagPool,
    TagSearch,
    keyword_scorer,
)

TREE_PATHS = [
    ("旅行", "国内游", "成都旅行"),
    ("旅行", "国内游", "北京旅行"),
    ("影视综艺", "电影", "喜剧电影"),
    ("求职学习", "实习面试", "大模型算法面试"),
]
L1_LEXICON = {
    "旅行": ["旅行", "旅游", "游", "景点", "攻略", "成都", "北京", "小吃"],
    "影视综艺": ["电影", "影视"],
    "求职学习": ["面试", "面经", "实习"],
}
L3_LEXICON = {
    "成都旅行": ["成都"],
    "北京旅行": ["北京"],
    "喜剧电影": ["喜剧", "沈腾"],
    "大模型算法面试": ["大模型", "算法"],
}

TAG_POSTS = [
    Post(
        "t1",
        "成都两日游经典路线（带老人版）",
        "宽窄巷子-人民公园喝茶-杜甫草堂，地铁直达，每天步行约2公里，适合老人。",
        "tag",
        ("旅行", "国内游", "成都旅行"),
    ),
    Post(
        "t2",
        "成都本地人推荐的老街小吃",
        "奎星楼街、玉林路小吃集中，人均50元，不用爬坡。",
        "tag",
        ("旅行", "国内游", "成都旅行"),
    ),
    Post(
        "t3",
        "沈腾近年喜剧电影口碑盘点",
        "《独行月球》《抓娃娃》口碑较好，观众评价集中在笑点密度和情感线。",
        "tag",
        ("影视综艺", "电影", "喜剧电影"),
    ),
    Post(
        "t4",
        "大模型算法实习高频考点汇总",
        "Transformer、RLHF/GRPO、KV Cache、MoE 路由是高频考点。",
        "tag",
        ("求职学习", "实习面试", "大模型算法面试"),
    ),
]
COMMUNITY_POSTS = [
    *TAG_POSTS,
    Post("c1", "杜甫草堂近期部分区域施工", "11月起草堂东区施工，主要景点仍开放，建议从正门进入。"),
    Post("c2", "人民公园鹤鸣茶社营业时间", "每天8:30-18:00营业，周末人多需排队。"),
    Post("c3", "成都景点步行距离与地铁接驳", "宽窄巷子到人民公园地铁2站，步行约800米。"),
    Post("c4", "沈腾综艺片段合集", "综艺名场面剪辑，非电影内容。"),
    Post("c5", "大模型算法面试手撕代码题", "常见：手写 attention、top-k 采样、LoRA 前向。"),
]


def build_tree() -> TagTree:
    return TagTree.from_paths(TREE_PATHS)


def build_registry(tree: TagTree | None = None) -> ToolRegistry:
    tree = tree or build_tree()
    router = CascadeTagRouter(tree, keyword_scorer(L1_LEXICON), keyword_scorer(L3_LEXICON))
    return ToolRegistry(
        [
            TagSearch(router, TagPool(TAG_POSTS)),
            CommunitySearch(InMemoryCommunityBackend(COMMUNITY_POSTS)),
        ]
    )


def build_prompt_builder(registry: ToolRegistry, max_rounds: int = 2) -> PromptBuilder:
    shots = [
        FewShotExample(
            "北京三日游怎么安排",
            "[Thinking]\n按景点、交通、美食拆分。\n[Input Decomposition]\n"
            "- sub query-1：北京三日游路线\n[Action]\n"
            '[{"name": "tag_search", "arguments": {"query": "北京三日游路线", "limit": 5}}]',
            "旅行",
        ),
        FewShotExample(
            "大模型面试准备什么",
            "[Thinking]\n按考点、coding、项目拆分。\n[Input Decomposition]\n"
            "- sub query-1：大模型算法面试高频考点\n[Action]\n"
            '[{"name": "tag_search", "arguments": {"query": "大模型算法面试高频考点", "limit": 5}}]',
            "求职",
        ),
    ]
    return PromptBuilder(registry, examples=shots, k_shots=1, max_rounds=max_rounds)


def _action(calls: list[tuple[str, str]], limit: int = 3) -> str:
    return json.dumps(
        [{"name": n, "arguments": {"query": q, "limit": limit}} for n, q in calls], ensure_ascii=False
    )


def chengdu_policy_turns(vertical: bool = True) -> list[str]:
    """A gold-style trajectory: horizontal round, optional vertical verification round, answer."""
    t1 = (
        "[Thinking]\n用户要带老人玩成都两天，要求少走路。拆成路线和美食两个方向，同时查标签池与社区。\n"
        "[Input Decomposition]\n- sub query-1：成都两日游 老人 少走路\n- sub query-2：成都 老街 小吃\n"
        "[Action]\n"
        + _action(
            [
                ("tag_search", "成都两日游 老人 少走路"),
                ("community_search", "成都两日游 老人 少走路"),
                ("tag_search", "成都 老街 小吃"),
            ]
        )
    )
    t2 = (
        "[Thinking]\n根据第 1 步，候选路线包含杜甫草堂和人民公园，需要核实近期施工与营业情况。\n"
        "[Input Decomposition]\n- sub query-1：杜甫草堂 施工\n- sub query-2：人民公园 茶社 营业时间\n"
        "[Action]\n"
        + _action([("community_search", "杜甫草堂 施工"), ("community_search", "人民公园 茶社 营业时间")])
    )
    answer = (
        "[Thinking]\n证据已覆盖路线、步行距离与营业信息，可以回答。\n"
        "[Response]\n第一天宽窄巷子 + 人民公园喝茶，地铁直达、步行约2公里[路线适合老人](ref: t1)；"
        "茶社 8:30-18:00 营业[营业时间](ref: c2)。第二天杜甫草堂，东区施工但主要景点开放[施工情况](ref: c1)，"
        "晚上去玉林路吃小吃[小吃推荐](ref: t2)。以上为社区网友经验，出行前建议再确认。"
    )
    return [t1, t2, answer] if vertical else [t1, answer]


def scripted_policy(vertical: bool = True) -> ScriptedProvider:
    return ScriptedProvider(chengdu_policy_turns(vertical))


OFF_TOPIC = ("股票", "基金", "综艺", "八卦")


def rule_query_judge() -> CallableJudge:
    """Toy relevance judge: a round scores 1.0 if none of its queries drift into an off-topic domain,
    0.5 if some do, 0.0 if all do. (Multi-hop queries need not repeat the user's keywords.)"""

    def judge(prompt: str) -> str:
        data = json.loads(prompt.split("待评估数据：", 1)[1].rsplit("只返回", 1)[0])
        drift = [any(w in q for w in OFF_TOPIC) for q in data["queries"]]
        score = 0.0 if drift and all(drift) else 0.5 if any(drift) else 1.0
        return json.dumps({"relevance": score})

    return CallableJudge(judge)


def rule_outcome_judges() -> dict[str, CallableJudge]:
    def has_refs(prompt: str) -> str:
        answer = prompt.split("系统答案：", 1)[1].split("请判断", 1)[0]
        n = len(re.findall(r"\(ref:", answer))
        return "1.0" if n >= 2 else "0.5" if n == 1 else "0.0"

    return {
        "safety": CallableJudge(lambda p: "1.0"),
        "style": CallableJudge(lambda p: "1.0"),
        "trust": CallableJudge(has_refs),
        "coherence": CallableJudge(lambda p: "1.0"),
    }


def demo_messages_contain(messages: Messages, needle: str) -> bool:
    return any(needle in m["content"] for m in messages)
