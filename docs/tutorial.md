# 教程：每个角色负责什么、需要懂什么、怎么上手

这份教程配合 [team_roles.md](team_roles.md) 使用：分工文档告诉你**做什么**，这里告诉你**为什么、怎么做**。

- 所有人先读 [第 0 章](#第-0-章全组必读)，再读自己角色的那一章。
- 每章结构相同：你负责什么 → 核心概念 → 代码导读 → 动手试一试 → 常见误区 → 自检问题。
- 文中"§"指 [PROPOSAL.md](../PROPOSAL.md) 的章节。教程只补充 proposal 没展开的直觉和代码细节，不重复全部公式。
- 所有 ` ```python ` 代码块都能直接运行，并且由 `tests/test_tutorial.py` 自动执行，保证教程和代码不会脱节。
  首行写着"需要 API"的代码块除外，它们要接真实模型。

| 章 | 读者 | 内容 |
|---|---|---|
| [0](#第-0-章全组必读) | 全组 | 项目全景、环境搭建、读懂一条轨迹 |
| [1](#第-1-章a-标签架构师) | A 标签架构师 | 兴趣点模型、LUFFY、漏斗、三级标签树、打标与准入 |
| [2](#第-2-章b-agent-导演) | B Agent 导演 | 运行框架、回合协议、横纵 Action、预算、Checkpoint |
| [3](#第-3-章c-搜索管家) | C 搜索管家 | 双供给工具、BERT 路由、训练专用搜索代理层 |
| [4](#第-4-章d-裁判长) | D 裁判长 | 三段式奖励、LLM 裁判、RaR 评测、出题 |
| [5](#第-5-章e-优势分配官) | E 优势分配官 | 奖励与优势、GRPO、GSPO、GiGPO、OPD |
| [6](#第-6-章f-数据炼金师) | F 数据炼金师 | 数据飞轮、Rubric 蒸馏、损失掩码、pass@k 分流 |
| [7](#第-7-章集成实验) | 全组 | 不训练的集成实验 |
| [附录](#附录术语表) | 全组 | 术语表（A 维护） |

---

## 第 0 章：全组必读

### 0.1 项目一句话

用户在内容社区里问一个复杂问题（"成都两日游，带老人，少走路"），Agent 在 **1~2 轮**内把问题拆开、
用两个工具检索、核实证据，最后给出带引用的回答。训练上，SFT 教会模型这套格式和基本策略，
RL 再教它"怎么拆、何时补查、何时停"。

### 0.2 一条 query 的一生

```
           离线（A）                         在线（B、C）                      训练与评测（D、E、F）
高热帖子 → 兴趣点 → 三级标签树 → 精选池 ─┐
                                          ├→ tag_search ──┐
全社区帖子 ───────────────────────────────┴→ community_search ─┤
                                                           │
用户 query → Agent 拆解（横向）→ 并行检索 → 补查（纵向）→ 带引用回答
                                                           │
                         奖励（D）→ 优势（E）→ 策略更新；评测与 badcase 回流（D、F）→ 新数据（F）→ …
```

### 0.3 环境

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

```bash
make check
```

```bash
uv run agentic-search demo
```

### 0.4 读懂一条轨迹

仓库自带一个完全离线的"玩具世界"（`agentic_search.demo`）：十几篇帖子、一棵小标签树、两个工具，
再加一个"照剧本说话"的假模型。下面让它回答一个问题：

```python
import asyncio

from agentic_search import demo
from agentic_search.runtime import AgentRunner

registry = demo.build_registry()  # 两个工具：tag_search、community_search
prompts = demo.build_prompt_builder(registry)  # 系统提示词（含动态 few-shot）
traj = asyncio.run(AgentRunner(demo.scripted_policy(), registry, prompts).run_once("成都两日游，带老人，希望少走路"))

for step in traj.steps:
    if step.is_tool_step:
        print(f"第 {step.index + 1} 步 检索：{step.queries()}")
    else:
        print(f"第 {step.index + 1} 步 回答：{step.response[:40]}…")
```

一条轨迹由若干**回合**组成，每个回合是模型的一次生成（§3.1、§4.2(1)）：

| 片段 | 谁产生 | 训练时算不算 loss |
|---|---|---|
| `[System Prompt]`、`[User Input]` | 离线构造 | 不算 |
| `[Thinking]` `[Input Decomposition]` `[Action]` | 模型 | **算** |
| `[Tool Result]` | 运行框架 | 不算 |
| `[Thinking]` `[Response]` | 模型 | **算** |

记住两个词，后面所有人都会用到：

- **横向 Action**：一个回合里并行发出多个子查询（一次往返搜完多个方向，省时延）。
- **纵向 Action**：下一回合基于上一回合的结果继续补查（多跳推理，提准确率）。

---

## 第 1 章：A 标签架构师

### 你负责什么

内容供给侧：好内容从哪里来、怎样被组织成三级标签、哪些帖子能进入精选池。对应 §1.1、§2.2、§4.1，代码在 `src/agentic_search/taxonomy/`。

### 核心概念

**为什么需要标签池（§1.1）。** 社区搜索的排序偏爱"新"和"热"，所以一篇三年前写得极好的避坑指南会被新帖子挤到下面。
标签池收录经过筛选的优质内容，并且**不随时间降权**，高频需求就能稳定召回这些长青内容。

**兴趣点提取模型（§4.1(1)）。** 想给千万级帖子抽"兴趣点"，全部调用 GPT 太贵，所以先让 GPT-5.6 标约 2.5 万条，
再蒸馏到开源的 Qwen3-VL-32B。分两步：

1. SFT 冷启动：先学会输出合法的 JSON（`{"summary": ..., "interest_points": [...]}`）。
2. LUFFY-GRPO：奖励 = 0.6 × RLAIF（裁判评概括与兴趣点质量）+ 0.4 × RLVR（规则校验 JSON）。

**为什么用 LUFFY 而不是普通 GRPO。** GRPO 的优势是"我比同组平均好多少"。如果一道难题采样 8 次全得 0 分，
那么人人都等于平均，优势全是 0，这组样本就没有梯度。LUFFY 往组里混一条人工校验过的满分教师示范：

```python
from agentic_search.algos import luffy_mixed_group_advantage

on, _ = luffy_mixed_group_advantage([0.0] * 8, [])
print("只有学生：", on.tolist())  # 全 0：没有学习方向

on, off = luffy_mixed_group_advantage([0.0] * 8, [1.0])
print("混入教师：学生", round(on[0].item(), 2), "教师", round(off.item(), 2))  # 学生 -0.35，教师 +2.83
```

教师 token 的概率还会经过塑形函数 $f(p)=p/(p+0.1)$。它让学生"还不会"的低概率 token 学得更用力，避免只模仿自己本来就会的部分。

**漏斗（§4.1(2)）。** 1000 万篇种子帖抽出 3000 万个兴趣点，再分两步缩减两个数量级：

1. **精确去重**（字面完全相同，410 万+）；
2. **向量聚类**（"沈腾 / 沈腾电影 / 沈腾喜剧人"合并成一个，24 万+），人工剔除聚错的簇。

```python
from agentic_search.core.text import HashingEmbedder
from agentic_search.taxonomy import run_funnel

raw = ["沈腾"] * 5 + ["沈腾电影"] * 2 + ["沈腾喜剧人", "王者荣耀", "王者荣耀", "王者荣耀攻略", "成都火锅", " 成都火锅"]
res = run_funnel(raw, HashingEmbedder(), distance_threshold=0.6)
print(res.stats)  # raw → unique → clusters
for c in res.clusters:
    print(c.representative, "<-", c.members)
# 沈腾 <- ['沈腾', '沈腾电影']；"沈腾喜剧人"仍单独成簇：玩具 embedding 只看字面字符，
# 换成真正的语义 embedding（如 BGE）后才能合并，阈值也要重新调。
```

**三级标签树（§4.1(2)）。**

- **自顶向下**：专家先定 50+ 个一级标签，把每个兴趣点"挂"到某个一级下面。
- **自底向上**：每个一级下面先聚类、再均匀采样约 1200 个兴趣点，交给 GPT、Claude、Gemini 分别归纳出二级标签，合并去重后由裁判检查"有没有覆盖、彼此是否重叠"，不合格就打回重写。
- **递归**：对每个二级重复同样的流程（采样约 300 个），得到三级标签。

仓库里所有"LLM 角色"都是接口，玩具版用向量相似度当替身：

```python
from agentic_search.taxonomy import ClusterInducer, EmbeddingMounter, TaxonomyBuilder

emb = HashingEmbedder()
points = ["成都火锅", "成都串串", "成都冒菜", "重庆火锅", "重庆小面", "重庆酸辣粉",
          "王者荣耀", "王者荣耀出装", "王者荣耀排位", "英雄联盟", "英雄联盟排位", "英雄联盟皮肤"]
mounter = EmbeddingMounter(emb, descriptions={"饮食餐饮": "火锅 小面 串串 冒菜 酸辣粉 美食",
                                              "游戏娱乐": "王者荣耀 英雄联盟 游戏"})
builder = TaxonomyBuilder(emb, mounter, [ClusterInducer(emb, "gpt"), ClusterInducer(emb, "claude")])
tree, report = builder.build(["饮食餐饮", "游戏娱乐"], points)
for l3 in tree.level(3):
    print(" / ".join(tree.path(l3)))
# 玩具归纳器只是按字面聚类起名，树长得很粗糙；重点看"挂载 → 采样 → 归纳 → 递归"的流程。
```

**打标与准入（§4.1(3)）。**

- **只训一级和三级打标模型。** 一级定了以后，三级候选只剩约 100 个，一步就能分对；二级可以由三级查表反推。二级类目仍然保留，留给未来扩展。
- **一级打标靠"多模型投票"**（Model Scaling）。
- **三级打标靠"三步思考"**（Test-Time Scaling）：先概括内容，再逐个辨析候选，最后决断。都不合适就打**空白标签**。
- **空白标签攒够了，说明社区出现了新兴趣。** 这既是给运营的预警，也会触发给三级标签扩容。
- **不是所有帖子都打标。** 互动分 $S$ 严格大于阈值 $\tau$ 才进入打标队列：

```python
from agentic_search.taxonomy import AdmissionController, InteractionWeights, rlvr_score

ctl = AdmissionController(tau=100, weights=InteractionWeights(likes=1, favorites=2, shares=3, comments=1.5))
print(ctl.admits({"likes": 40, "favorites": 30}))  # 40 + 60 = 100，等于阈值不触发 → False
print(ctl.rescore_incremental([("new_post", {"likes": 40, "favorites": 31})]))  # 102 > 100，首次越线入队

print(rlvr_score('<think>先概括</think>{"summary": "沈腾新片口碑", "interest_points": ["沈腾", "喜剧电影"]}'))  # 1.0
print(rlvr_score('{"summary": "x", "interest_points": ["沈腾", "沈腾"]}'))  # 0.0，兴趣点重复
```

### 代码导读

| 文件 | 看什么 |
|---|---|
| `taxonomy/interest_points.py` | 输出格式、RLVR 校验、0.6/0.4 混合奖励 |
| `algos/losses.py::luffy_loss` | LUFFY 目标函数（与 §4.1(1) 公式逐项对应） |
| `taxonomy/funnel.py` | 精确去重、层次聚类、大规模时用的 leader 聚类、均匀采样 |
| `taxonomy/builder.py` | 挂载 → 采样 → 多模型归纳 → 合并去重 → 反思校验 → 递归 |
| `taxonomy/admission.py` | 准入公式、存量与增量、空白标签监控 |

### 常见误区

- 二级"不训练"不等于"不存在"：二级类目保留在树里，由三级反推得到。
- 互动分等于阈值时**不**进入打标；没进池的帖子仍然可以被 `community_search` 搜到。
- 空白标签不是模型出错，而是开放集发现的信号。

### 自检问题

1. 为什么全 0 奖励的组在 GRPO 里没有梯度？LUFFY 用什么办法补救？
2. 漏斗里"精确去重"和"向量聚类"各解决什么问题？
3. 为什么训练一级和三级打标模型，却不训练二级？

---

## 第 2 章：B Agent 导演

### 你负责什么

Agent 运行框架：把模型、工具和预算组织成一个可靠的循环。对应 §1.2、§3、§3.1、§4.2(1)，代码在 `src/agentic_search/runtime/`。

### 核心概念

**为什么只能 1~2 轮（§1.2）。** B 端工具（写代码的 Agent）可以等几分钟，C 端用户只肯等几秒。
每多一轮 ReAct，就多一次"模型生成 + 检索"的往返。所以要用**横向 Action** 把原本 4 轮的串行检索压到 1 轮，
需要时再用 1 轮**纵向 Action** 补查。proposal 表 4 中的结果是 4 轮/15 s 降到 1~2 轮/6 s。

**双层循环（§3.1）。** Main Loop 是常驻服务，不断接收请求；Agent Once Loop 处理单个请求，有最大迭代次数。
一次迭代分三步：先生成回合并解析，再校验参数、并发执行工具，最后把结果写回上下文。

**回合协议。** 模型每回合输出带标签的文本，框架用 `parse_turn` 解析：

```python
from agentic_search import demo
from agentic_search.runtime import parse_turn

turn = demo.chengdu_policy_turns()[0]
p = parse_turn(turn)
print(p.thought)
print([q.query for q in p.decomposition])
print(p.action)  # JSON 列表；一个列表 = 一次横向 Action
```

`[Action]` 里每个调用只有 `name` 和 `arguments` 两个键。`call_id` 由框架分配，用来查日志，模型不需要生成它。

**预算与兜底。** 模型不肯停怎么办？轮数用完后，框架会追加一句"请直接回答"；如果模型仍然调用工具，就强制结束本次请求：

```python
import asyncio

from agentic_search.runtime import AgentRunner, RuntimeConfig, ScriptedProvider

registry = demo.build_registry()
prompts = demo.build_prompt_builder(registry)
looping = ScriptedProvider([turn, turn, turn])  # 一个只会不停搜索的"模型"
runner = AgentRunner(looping, registry, prompts, config=RuntimeConfig(max_tool_rounds=2))
traj = asyncio.run(runner.run_once("成都两日游"))
print(traj.num_tool_rounds, traj.meta["truncated"])  # 2 budget
print(looping.calls[-1][0][-1]["content"])  # 框架追加的提示
```

解析失败（JSON 写坏了）和参数非法（例如 `limit=99`）都算**策略错误**：框架把错误提示写回去，让模型改正，同时消耗一轮。
检索服务宕机则是**基础设施故障**，不能怪模型（见第 3、4 章）。

**动态 few-shot（§4.2(1)）。** 示例通常占提示词的 50%～60%，直接拉长首字时延（prefill 阶段要处理所有输入 token）。
所以只按 query 的语义相似度挑最相关的 1~2 个示例：

```python
print(prompts.select_examples("北京旅游攻略")[0].query)  # 选中旅行类示例
print(prompts.select_examples("大模型面试题")[0].query)  # 选中求职类示例
```

**Checkpoint 与 LLM Provider（§3.1）。** Checkpoint 保存一条轨迹的任意前缀，包括写到一半的回合；
LLM Provider 是可替换的生成后端。两者组合起来，就能把某一步的拆解换掉，再让模型接着写下去。F 的数据飞轮依赖这个机制：

```python
from agentic_search.runtime import Checkpoint

gold = asyncio.run(AgentRunner(demo.scripted_policy(), registry, prompts).run_once("成都两日游，带老人，希望少走路"))
ckpt = Checkpoint.with_injected_decomposition(gold, step_index=0, sub_queries=["成都 带老人 景点", "成都 小吃"])
print(ckpt.assistant_prefix)  # 模型将从 [Action] 处继续生成
```

接真实模型时，任何 OpenAI 兼容接口都可以：

```python
# 需要 API：把剧本模型换成真实模型（不训练，纯 prompt）
from agentic_search.runtime import OpenAICompatibleProvider

llm = OpenAICompatibleProvider(base_url="https://<your-endpoint>/v1", model="<model-name>", api_key="<key>",
                               temperature=0.7)
traj = asyncio.run(AgentRunner(llm, registry, prompts).run_once("北京三日游，预算 2000 元"))
```

真实模型不一定严格遵守回合格式。遇到格式问题，先调系统提示词里的"输出格式"和 few-shot 示例，再统计解析失败率。

### 代码导读

| 文件 | 看什么 |
|---|---|
| `runtime/protocol.py` | 回合解析、`[Tool Result]` 渲染、由历史重建消息 |
| `runtime/agent.py` | `run_once` 主循环、预算、并发执行、`Checkpoint` |
| `runtime/llm.py` | `OpenAICompatibleProvider`（支持续写半个回合）、`ScriptedProvider` |
| `runtime/prompt.py` | 提示词七段结构、动态 few-shot |
| `tools/base.py::normalize_action` | 把 `[Action]` 解析成原子调用（兼容列表式简写） |

### 常见误区

- `[Tool Result]` 写在上下文里，但永远不参与训练。
- 横向 Action 的多个调用是并发执行的，不是一个接一个。
- 超过轮数被截断的轨迹不能直接当 SFT 数据（导出时会被过滤）。

### 自检问题

1. 横向 Action 和纵向 Action 分别优化什么指标？
2. 模型输出了非法 JSON，框架怎么处理？这算谁的错？
3. Checkpoint 在数据飞轮里起什么作用？

---

## 第 3 章：C 搜索管家

### 你负责什么

两条供给链路，以及训练时保护搜索服务的代理层。对应 §1.1、§3（IMPORTANT 说明）、§3.1 的工具表、§5.2，代码在 `src/agentic_search/tools/`。

### 核心概念

**双供给（§1.1、§3）。**

| 工具 | 搜什么 | 适合什么需求 |
|---|---|---|
| `tag_search` | 精选标签池 | 高频、主题明确的需求，返回筛过的长青好内容 |
| `community_search` | 全社区 | 长尾、新话题、具体对象的核实（"这家店现在还开吗"） |

两者签名完全一样，都是 `(query, limit)`，模型不需要学两套用法。内容冲突时以 `tag_search` 为准。

**`tag_search` 内部怎么走。** 在线收到的是短短一句子查询，用多模态大模型打标太慢，所以用两个轻量 BERT 级联分类：
一级 BERT 先定大类，三级 BERT 只在这个大类的子树里选三级标签。一个 query 可以命中多个簇，再在簇内按相关性排序。
玩具版用关键词表代替 BERT：

```python
import asyncio

from agentic_search import demo

registry = demo.build_registry()
tag = registry["tag_search"]
print(tag.router.route("成都两日游 老人"))  # [('旅行', '成都旅行')]
res = asyncio.run(tag({"query": "成都两日游 老人", "limit": 3}))
print(res.status, [p.post_id for p in res.posts])
res = asyncio.run(registry["community_search"]({"query": "杜甫草堂 施工", "limit": 3}))
print(res.status, [p.post_id for p in res.posts])
```

**训练时的流量风暴（§5.2）。** RL 每步对 128 个 prompt 各采样 8 条轨迹，每条轨迹横向发出 3~4 个查询，
一瞬间就是上千个请求，直接打线上搜索会排队、超时，甚至拖垮生产服务。代理层的五个机制：

1. **有界队列 + 背压**：队列满了就让 rollout 慢下来，并丢弃已取消轨迹的请求；
2. **按工具限流**：令牌桶加最大在途数，全局统一协调；
3. **精确缓存 + 合并请求**：同组 8 条轨迹发出同一个查询，只打一次后端；
4. **超时与有限重试**：重试也受限流管控，并严格区分"没搜到"和"服务挂了"；
5. **监控**：QPS、排队时间、命中率、合并率。

```python
from agentic_search.tools import SearchProxy, ToolLimits
from agentic_search.tools.search import InMemoryCommunityBackend


async def burst():
    backend = InMemoryCommunityBackend(demo.COMMUNITY_POSTS)
    limits = {"community_search": ToolLimits(rate_per_s=50, burst=5)}
    async with SearchProxy({"community_search": backend}, limits) as proxy:
        # 1 个 prompt × 8 条 rollout，每条都发出同样的 2 个子查询
        calls = [proxy.call("community_search", {"query": q, "limit": 3}, trajectory_id=f"r{i}")
                 for i in range(8) for q in ("杜甫草堂 施工", "人民公园 茶社")]
        await asyncio.gather(*calls)
        return proxy.metrics()


m = asyncio.run(burst())
print("请求", m["community_search/requests"], "实际打到后端", m["community_search/backend_calls"],
      "合并率", round(m["coalesce_rate"], 2))  # 16 → 2
```

**为什么缓存必须"精确"。** "成都周末去哪玩"和"成都周末带娃去哪玩"语义很像，但后者多了一个硬约束。
语义缓存会把两者混为一谈，所以缓存键必须精确到 `(工具, query, limit, 过滤条件, 索引版本)`。

### 代码导读

| 文件 | 看什么 |
|---|---|
| `tools/base.py` | 工具基类、注册表、参数 Schema 校验 |
| `tools/search.py` | 两个工具、级联路由 `CascadeTagRouter`、HTTP 后端适配 |
| `tools/proxy.py` | 五个机制各自对应的代码段 |

### 常见误区

- 工具结果的 `empty`（真的没搜到）和 `error`（服务故障）必须分开。后者会让整条轨迹标记为无效，不参与训练。
- 限流要在所有 rollout worker 之间统一协调，每个 worker 各自限流的话，加起来仍会超标。

### 自检问题

1. 为什么不让 `tag_search` 直接调用 Qwen3-VL-8B 打标模型来理解 query？
2. 一组 8 条轨迹发出相同查询时，代理层做了什么？
3. 为什么不能用语义相似度做缓存？

---

## 第 4 章：D 裁判长

### 你负责什么

两件事：训练时的**奖励**（告诉 RL 这条轨迹有多好），和评测时的 **RaR 打分**（告诉我们模型到底行不行）。
对应 §4.3(1)、§6.1、§6.2(3)，代码在 `src/agentic_search/rewards/` 和 `src/agentic_search/eval/rar.py`。

### 核心概念

**三段式奖励（§4.3(1)）：$R = 0.1 \times 格式 + 0.3 \times \text{query 相关性} + 0.6 \times 最终效果$。**

- **格式**（代码判定）：工具调用是否合法占 60%；需要多跳的步骤是否写了"根据第 x 步"占 40%。
  为什么要奖励这句话？模型是逐 token 生成的，前面写出"根据第 1 步"，后面的生成就会更多地关注第 1 步的检索结果。

```python
from agentic_search import demo
from agentic_search.rewards import check_multi_hop_attribution, check_tool_format

registry = demo.build_registry()
print(check_tool_format({"name": "tag_search", "arguments": {"query": "成都", "limit": 5}}, registry))  # 1.0
print(check_tool_format({"name": "tag_search", "arguments": {"query": "成都", "limit": 99}}, registry))  # 0.0，越界
print(check_multi_hop_attribution(2, "根据第 1 步的结果，需要核实施工情况", num_steps=3))  # 1.0
print(check_multi_hop_attribution(2, "根据第 3 步", num_steps=3))  # 0.0，引用了自己
```

- **query 相关性**（LLM 裁判）：每一轮检索单独打分，再取**平均**。这样搜得越多不会分越高，搜跑题了会被拉低分。
- **最终效果**（LLM 裁判）：安全、社区语气、可信引用、逻辑自洽四个维度，**每个维度一个裁判**，只打 1.0 / 0.5 / 0.0，只输出数字。
  引用另有代码闸门：引用了没检索到的帖子，就是伪造来源。

```python
import asyncio

from agentic_search.rewards import AgenticSearchReward
from agentic_search.runtime import AgentRunner

prompts = demo.build_prompt_builder(registry)
traj = asyncio.run(AgentRunner(demo.scripted_policy(), registry, prompts).run_once("成都两日游，带老人，希望少走路"))
reward = AgenticSearchReward(registry, demo.rule_query_judge(), demo.rule_outcome_judges())
print(reward(traj)["score"])  # 1.0

traj.steps[-1].response += "[编造的结论](ref: not_exist)"
r = reward(traj)
print(r["outcome/citation_gate"], round(r["score"], 2))  # 0.0 0.85：可信维度被闸门清零
```

**裁判的三条铁律**是"一个 prompt 只评一个维度、分档且档位清楚、只输出数字"。原因有两个：
多维度混评会互相干扰；输出解释会让 RL 训练时的打分吞吐慢一个数量级。

**专用奖励模型（§4.3(1)、表 1）。** 训练中要调用几十万次裁判，全用闭源 API 太贵，所以蒸馏一个专用 RM，分三步：

1. 先用提示词工程把大模型裁判的效果榨到上限；
2. 用 2000+ 条专家标注做 SFT；
3. 针对顽固误判做 DPO：专家答案作为 chosen，模型误判作为 rejected。

上线标准是与人类专家一致率 ≥ 95%。

**RaR 评测（§6.1）。** 社区问答没有标准答案，所以每道题配一份**评分量表**：

- **P0 红线**：一票否决。只收**明确可判定**的硬约束，比如"每天步行不超过 3 公里"；"尽量少走路"属于偏好，不算 P0。证据不足、无法判定的，交给人工复核。
- **P1 搜索过程 + P2 回答质量**：合计 100 分，其中格式 10、搜索 30、回答 60，和训练奖励同一尺度。
- **通过条件**：P0 全过，且总分 ≥ 80。

```python
from agentic_search.eval import P0Constraint, RaREvaluator, RaRTask
from agentic_search.rewards import CallableJudge

task = RaRTask("cd-001", "成都两日游，带老人，每天步行不超过 3 公里",
               p0=[P0Constraint("walk<=3km", "每天步行不超过 3 公里")],
               rubric_notes="需覆盖两天路线、每日步行距离、适合老人的餐饮")
# 替身裁判：P0 题回答 0（未违反），其余维度都给满分。真实评测换成 LLM 裁判。
judge = CallableJudge(lambda prompt: "0" if "硬性要求" in prompt else "1")
res = RaREvaluator(registry, judge).evaluate(task, asyncio.run(
    AgentRunner(demo.scripted_policy(), registry, prompts).run_once(task.query)))
print(res.F, res.Q, res.A, res.S_quality, res.G_P0, res.passed)
```

**出题（第二身份）。** mini benchmark 建议每题一行 JSON，四类场景都要覆盖：

```json
{"task_id": "cd-001", "category": "旅行", "scenario": "多约束",
 "query": "成都两日游，带老人，每天步行不超过 3 公里，不去爬山景点",
 "p0": ["每天步行不超过 3 公里", "不去爬山景点"],
 "rubric_notes": "需覆盖两天路线、每日步行距离、适合老人的餐饮；需说明信息时效"}
```

四类场景是：需要补查、多约束、检索无结果（看会不会编）、需要澄清（看会不会反问）。

### 代码导读

| 文件 | 看什么 |
|---|---|
| `rewards/format_reward.py` | 与 §4.3(1) 伪代码逐行对应 |
| `rewards/query_relevance.py` | 逐轮打分与防注入声明 |
| `rewards/outcome.py` | 四个维度的提示词模板、引用闸门 |
| `rewards/composite.py` | 10/30/60 合成，故障轨迹标记 `valid=0` |
| `eval/rar.py` | 十维评分表、P0 判定、通过规则 |

### 常见误区

- 奖励是"间接信号"，优势才是"直接信号"（第 5 章）。奖励设计好了，还要看组内是否有差异。
- 裁判判不出时不要硬给 0 分，要计入失败统计；P0 判不出要送人工。
- 权重取 10/30/60，是为了和 RaR 的分值对齐；需要时可以在 `configs/agentic_search.yaml` 里调整。

### 自检问题

1. 为什么 query 相关性要逐轮打分再取平均，而不是取总和？
2. "尽量少走路"和"每天步行不超过 3 公里"，哪个可以作为 P0？为什么？
3. 裁判为什么只能输出数字？

---

## 第 5 章：E 优势分配官

### 你负责什么

把奖励变成训练信号：轨迹级、步骤级、token 级三种粒度的优势，以及 MoE 模型上稳定的策略损失。
对应 §1.3、§2.3、§4.3(2)、§5.3、§6.2(1)，代码在 `src/agentic_search/algos/` 和 `src/agentic_search/integrations/verl/`。

### 核心概念

**奖励和优势的区别。** 奖励回答"这条轨迹得了几分"；优势回答"这条轨迹（这一步、这个 token）比同类平均好多少"。
策略梯度直接乘的是优势。

**轨迹级：GRPO 式组内标准化。** 同一个问题采样一组回答，用组内均值作基线，因此不需要额外训练价值网络（critic）：

```python
from agentic_search.algos import group_normalized_advantage

print(group_normalized_advantage([1.0, 0.85, 0.66], [0, 0, 0]).tolist())  # 高于均值为正，低于均值为负
```

缺点是，整条轨迹的每个 token 都拿同一个优势。一条"前面绕了几步弯路、最后碰巧答对"的轨迹里，弯路也会被一起强化。

**MoE 的麻烦与 GSPO（§1.3、§4.3(2)）。** Qwen3-30B-A3B 每层有 128 个专家，每个 token 只激活其中几个。
推理引擎（vLLM）和训练引擎（Megatron）的数值差异，或者参数更新本身，都可能让同一个 token 换了专家，
它的概率就会突然跳变。PPO/GRPO 按单个 token 的概率比做裁剪，这些跳变会把更新信号切得支离破碎。
GSPO 改为看**整条序列**的几何平均概率比：单个 token 的跳变被平均掉了。

序列比率非常接近 1，所以裁剪区间要用 $3\times10^{-4}$ / $4\times10^{-4}$，而不是 token 级常用的 0.8 / 1.2。

我们的优势随步骤、随 token 变化，所以实际使用 **GSPO-token**：裁剪仍然看序列比率，梯度按 token 携带各自的优势。
当优势在序列内是常数时，它和 GSPO 完全一样：

```python
import torch

from agentic_search.algos import gspo_sequence_loss, gspo_token_loss

torch.manual_seed(0)
old = -torch.rand(2, 5)
new = (old + 1e-4 * torch.randn(2, 5)).requires_grad_(True)
mask = torch.ones(2, 5)
seq_adv = torch.tensor([1.0, -1.0])
l_tok = gspo_token_loss(new, old, seq_adv[:, None].expand(2, 5), mask).loss
l_seq = gspo_sequence_loss(new, old, seq_adv, mask)
print(round(l_tok.detach().item(), 6), round(l_seq.detach().item(), 6))  # 相等
```

**步骤级：GiGPO（§4.3(2)）。** 关键是分清两个概念：**状态 = 决策前已经拿到的帖子集合；动作 = 这一步发出的 query**。
把不同轨迹里"手上帖子完全相同"的步骤放在一组比较：从同一个起点出发，谁更快到达终点，谁的动作就更好。
回报是 $Q=\gamma^{\text{距终点步数}}\times R$，再在组内标准化。

下面三条轨迹终局奖励都是 1。A、B 是三条轨迹里状态相同的两个节点：

```python
from agentic_search.algos import gigpo_step_advantage

keys = [
    ["s0", "A", "x1", "x2", "x3", "B", "x4", "R1"],  # 轨迹 1：A 之后绕了很多步
    ["s0", "A", "B", "y1", "R2"],  # 轨迹 2
    ["A", "B", "R3"],  # 轨迹 3：最快
]
adv = gigpo_step_advantage([1.0, 1.0, 1.0], [0, 0, 0], keys, gamma=0.9, eps=0.0)
for i, k in enumerate(keys):
    print(f"轨迹 {i + 1}", "A:", round(float(adv[i][k.index("A")]), 3), "B:", round(float(adv[i][k.index("B")]), 3))
# A: -1.357 / +0.332 / +1.024    B: -0.707 / -0.707 / +1.414
```

只出现一次的状态没有比较对象，所以没有步骤级信号。实际训练中，所有轨迹的第一步都从"空手"出发，必然同组；
后面的步骤只有在拿到了同样的帖子时才能比较。

**token 级：OPD（§4.3(2)）。** 一个搜索 query 的主题可能是对的，却漏掉了"带老人"这个约束词。
这种错误要细到 token 才看得出来。做法是：让更大的冻结教师模型（Qwen3-235B-A22B）对学生生成的轨迹逐 token 打分，
$\hat A^{OPD}_t = \log \pi_{教师} - \log \pi_{学生}$：教师更认可的 token 被推高，教师不认可的被压低。
前提是师生的词表一致（同属 Qwen3 家族）。

```python
from agentic_search.algos import combine_advantages, opd_token_advantage

teacher = torch.tensor([[-3.0, -1.0]])  # 教师明显否定第 1 个 token
student = torch.tensor([[-1.0, -1.0]])
opd = opd_token_advantage(teacher, student, torch.ones(1, 2))  # [[-2, 0]]
print(combine_advantages(torch.tensor([-1.0]), torch.ones(1, 2), opd_adv_tokens=opd, lambda_opd=0.5).tolist())
# [[-2.0, -1.0]]：同一条错误回答里，被教师否定的 token 罚得更重，普通 token 不受额外影响
```

**最终组合：$\hat A = \hat A^E + \omega\hat A^S + \lambda\hat A^{OPD}$**，只作用于模型生成的 token。

**如果真要训（§5.1、§5.3）。** 读 `scripts/rl/run_gspo_gigpo_opd_qwen3_30b_a3b.sh`，能说清这几项就够了：

- `n=8`：每题采样 8 条；
- `loss_mode=gspo` 与 3e-4/4e-4 裁剪；
- `expert_model_parallel_size=8`：每张卡负责 16 个专家；
- 三阶段上线检查：见 [rl_debug_protocol.md](rl_debug_protocol.md)。

### 代码导读

| 文件 | 看什么 |
|---|---|
| `algos/advantages.py` | 三种优势与组合，每个函数的文档字符串都有公式 |
| `algos/losses.py` | GSPO-token、PPO、LUFFY |
| `integrations/verl/advantage.py` | 如何把三种优势接进 VERL |
| `tests/test_advantages.py`、`tests/test_losses.py` | 最好的"例题集" |

### 常见误区

- 把 GiGPO 的"动作"理解成"状态"：锚点是**手上的帖子**，不是 query。
- 组内奖励全 0 或全 1 时没有轨迹级信号。这不是 bug，但比例过高说明题目太难、太简单，或者工具坏了。
- 不要把 0.8/1.2 用在 GSPO 上。

### 自检问题

1. 为什么 GSPO 对 MoE 更友好？
2. GiGPO 中，哪些步骤会被分到同一组？单独一组的步骤拿到什么优势？
3. OPD 为什么要求师生词表一致？

---

## 第 6 章：F 数据炼金师

### 你负责什么

SFT 数据从哪里来，以及失败案例往哪里回流。对应 §4.2(2)(3)、§6.3，代码在 `src/agentic_search/data/` 和 `src/agentic_search/eval/pass_at_k.py`。

### 核心概念

**数据是重中之重。** 业务算法工程师的大部分时间花在数据上。SFT 的 37,000 条轨迹（§5.1）由两类数据组成：
横向（怎么拆问题）和纵向（怎么多跳补查）。

**横向数据：会话日志飞轮（§4.2(2)）。** 用户一次没搜到想要的，会接着换词再搜。这串连续查询本身就是一份"把需求拆开"的示范。流程分四步：

1. 按用户和时间切分会话；
2. 把查询向量化并聚类，同义改写归为一个"簇查询"；
3. 统计相邻两个查询的共现次数（2-gram）：$\text{freq}(A\to B) > X$ 说明 B 是 A 的子查询（**怎么拆**）；某个查询的所有后继都不超过 $X$，说明它不必拆（**何时不拆**）；
4. 强模型做逻辑质检，过滤掉玩梗之类的无关跳转。

```python
from agentic_search.core.text import HashingEmbedder
from agentic_search.data import LogRecord, bigram_counts, canonicalize_queries, mine_decompositions, sessionize

logs, t = [], 0.0
variants = ["成都美食攻略", "成都景点 推荐", "成都 美食攻略", "成都景点推荐"]  # 含同义改写
for u in range(12):
    for q in ["成都旅游", variants[u % 4]]:
        logs.append(LogRecord(f"u{u}", q, t))
        t += 60
    t += 10_000  # 超过会话间隔，下一位用户从新会话开始
for u in range(3):  # 少量无关跳转
    logs += [LogRecord(f"n{u}", "沈腾电影", 0.0), LogRecord(f"n{u}", "沈腾表情包", 30.0)]

canon = canonicalize_queries([r.query for r in logs], HashingEmbedder(), distance_threshold=0.3)
mining = mine_decompositions(bigram_counts(sessionize(logs, gap_s=1800), canon), min_freq=5)
for example in mining.examples():
    print(example)
# 成都旅游 → 成都美食攻略、成都景点推荐（同义改写已合并，各 6 次 > 5）；沈腾电影 → []（只有 3 次，不拆）
```

挖到的子查询通过 Checkpoint 注入轨迹（见第 2 章），再让模型续写后面的内容。

> 加分思路：2-gram 只看紧挨着的两个查询。如果用户的顺序是"成都旅游 → 景点 → 美食"，"美食"就只会被算成"景点"的后继。
> 把它扩展成"会话内跳跃计数"（skip-gram）可以修正这个问题。

**纵向数据：Rubric 两阶段蒸馏（§4.2(3)）。**

- **第一阶段（人工）**：每个类目取种子 query，让 GLM-5.2、Kimi-K3、DeepSeek V4-Pro 在真实框架里边想边搜，产出轨迹 A、B、C；裁判模型按人工定的"元规则"融合出最优轨迹，人工审校后成为**黄金轨迹**；再从黄金轨迹里提炼出这个类目的**评价标准（Rubric）** 和**生成提示（Hint）**。
- **第二阶段（自动）**：同类目的更多 query 带着 Hint 生成，用 Rubric 自动挑出好轨迹。
- **query 来源**：按标签树覆盖所有类目（保证覆盖面），再从线上日志随机捞取（保证真实分布：口语、错别字、中英混杂）。

一个容易忽略的细节：**Hint 只能用于生成，不能出现在训练数据里**。否则模型学会的是"看到提示才会做"。
导出时，仓库会自动换回干净的系统提示词。

**损失掩码（§4.2(3)）**：只有模型生成的部分算 loss。

```python
import asyncio
import json
import tempfile
from pathlib import Path

from agentic_search import demo
from agentic_search.data import export_sft_jsonl
from agentic_search.runtime import AgentRunner

registry = demo.build_registry()
prompts = demo.build_prompt_builder(registry)
traj = asyncio.run(AgentRunner(demo.scripted_policy(), registry, prompts).run_once("成都两日游，带老人，希望少走路"))
path = Path(tempfile.mkdtemp()) / "sft.jsonl"
export_sft_jsonl([traj], path)
for m in json.loads(path.read_text(encoding="utf-8").splitlines()[0])["messages"]:
    print(f"{m['role']:<10} {'算 loss' if m['role'] == 'assistant' else '不算  '}  {m['content'][:20]!r}")
```

**badcase 往哪里回流：pass@k（§6.3）。** 同一个 query 采样多次（§6.3 用 32 次）：

- pass@1：一次就成功的概率；
- pass@k：k 次里至少成功一次的概率。

| 现象 | 说明 | 去向 |
|---|---|---|
| pass@1 低，pass@k 明显高 | 会做但不稳定 | RL |
| 都低，但教师或人能做 | 模型缺方法或缺知识 | 先补 SFT 数据 |
| 都低，教师也做不了 | 检索、工具或题目有问题 | 修系统 |
| pass@1 已经高 | 稳了 | 降低优先级 |

```python
from agentic_search.eval import diagnose, pass_at_k

print(round(pass_at_k(32, 4, 1), 3), round(pass_at_k(32, 4, 16), 3))  # 0.125 vs 0.949
outcomes = {
    "q-unstable": [True] * 4 + [False] * 28,
    "q-hard": [False] * 32,
    "q-broken": [False] * 32,
    "q-easy": [True] * 30 + [False] * 2,
}
for d in diagnose(outcomes, teacher_solvable={"q-hard": True, "q-broken": False}):
    print(d.task_id, {k: round(v, 2) for k, v in d.pass_at.items()}, d.route.value)
```

### 代码导读

| 文件 | 看什么 |
|---|---|
| `data/flywheel.py` | 会话切分、查询聚类、2-gram、阈值、质检 |
| `data/rubric_distill.py` | 两阶段蒸馏的编排（各个 LLM 角色都是接口） |
| `data/trajectory_format.py` | SFT 导出、token 级掩码与步骤编号 |
| `eval/pass_at_k.py` | 无偏估计与分流规则 |

### 常见误区

- 盲目给不稳定的 case 补 SFT 数据，会扭曲整体数据分布。这类 case 应该交给 RL。
- pass@k 是诊断工具。对 C 端产品来说，最终要优化的是 pass@1。

### 自检问题

1. 为什么飞轮要先聚类再统计 2-gram？
2. Hint 为什么不能进入训练数据？
3. 一条 query 的 pass@1 是 0.1、pass@16 是 0.9，该补 SFT 还是交给 RL？

---

## 第 7 章：集成实验

我们不训练模型，但可以把"训练前的一切"真实地跑一遍。流程是 D 出题 → B 采样 8 次 → D 打分 → E 算优势 → F 分流，
A、C 负责扩充语料和标签。下面是离线替身版本，结构和真实实验完全一样：
把 `sample_policy` 换成真实 API 模型（temperature 大于 0），把奖励里的裁判换成 LLM 裁判，就是正式实验。

```python
import asyncio

from agentic_search import demo
from agentic_search.algos import gigpo_step_advantage, group_normalized_advantage, group_reward_std_stats
from agentic_search.eval import diagnose
from agentic_search.rewards import AgenticSearchReward
from agentic_search.runtime import AgentRunner, ScriptedProvider

registry = demo.build_registry()
prompts = demo.build_prompt_builder(registry)
reward = AgenticSearchReward(registry, demo.rule_query_judge(), demo.rule_outcome_judges())
DRIFT = [
    demo.chengdu_policy_turns()[0],
    '[Thinking]\n再看看别的。\n[Action]\n[{"name": "community_search", "arguments": {"query": "沈腾 综艺", "limit": 3}}]',
    "[Thinking]\n直接回答。\n[Response]\n成都很好玩。",
]


def sample_policy(i):  # 替身：三种"性格"轮流出现，模拟同一模型的 8 次采样
    if i % 3 == 0:
        return demo.scripted_policy(vertical=True)
    if i % 3 == 1:
        return demo.scripted_policy(vertical=False)
    return ScriptedProvider(list(DRIFT))


async def rollouts(query, n=8):
    return [await AgentRunner(sample_policy(i), registry, prompts).run_once(query) for i in range(n)]


query = "成都两日游，带老人，希望少走路"
trajs = asyncio.run(rollouts(query))
scores = [reward(t)["score"] for t in trajs]
gid = [query] * len(trajs)
traj_adv = group_normalized_advantage(scores, gid)
step_adv = gigpo_step_advantage(scores, gid, [[s.anchor_key() for s in t.steps] for t in trajs])

print("奖励：", [round(s, 2) for s in scores])
print("轨迹级优势：", [round(a, 2) for a in traj_adv.tolist()])
print("组内方差：", group_reward_std_stats(scores, gid))
signal = sum(int((a != 0).sum()) for a in step_adv) / sum(len(a) for a in step_adv)
print("有步骤级信号的步骤占比：", round(signal, 2))
print(diagnose({query: [s >= 0.8 for s in scores]}, ks=(1, 8))[0])  # 正式实验用 RaR 的通过判定
```

正式实验建议的对照组（数字写进结题报告）：

| 实验 | 对照 | 指标 | 主要负责 |
|---|---|---|---|
| 双供给 | 只开 `community_search` vs 两个工具都开 | RaR 得分、引用正确率 | C、D |
| 横向 Action | 每轮只许一个 query vs 并行 | 平均轮次、端到端耗时 | B |
| 纸上 RL | 每题 8 次采样 | 奖励方差、GiGPO 有信号步骤占比、pass@1 / pass@8 | E |
| badcase 分流 | 按 pass@k 分三类 | 各类数量与典型例子 | F |

---

## 附录：术语表

由 A 维护，欢迎所有人补充。

| 术语 | 一句话解释 | PROPOSAL | 代码 |
|---|---|---|---|
| 双供给 | 精选标签池与全社区检索两条内容链路 | §1.1、§3 | `tools/search.py` |
| 横向 Action | 一个回合内并行发出多个子查询 | §4.2(1) | `runtime/agent.py` |
| 纵向 Action | 基于上一回合结果继续补查 | §4.2(1) | `runtime/agent.py` |
| Agent Once Loop | 单次请求内有最大轮数的 ReAct 循环 | §3.1 | `AgentRunner.run_once` |
| Checkpoint | 从轨迹任意前缀继续生成的机制 | §3.1、§4.2(2) | `runtime.Checkpoint` |
| 兴趣点 | 从帖子中抽出的开放式主题词，标签体系的原材料 | §4.1(1) | `taxonomy/interest_points.py` |
| 空白标签 | 三级打标时所有候选都不合适的输出，代表新兴趣 | §4.1(3) | `taxonomy.BlankTagMonitor` |
| 准入阈值 τ | 互动分严格大于 τ 才进入打标队列 | §4.1(3) | `taxonomy.AdmissionController` |
| LUFFY | 把教师满分示范混入采样组，解决全 0 奖励无梯度 | §4.1(1) | `algos.luffy_loss` |
| RLVR / RLAIF | 规则可验证的奖励 / AI 裁判给出的奖励 | §4.1(1)、§4.3(1) | `rewards/` |
| 优势 | 比同组平均好多少，策略梯度的直接系数 | §4.3(2) | `algos/advantages.py` |
| GSPO / GSPO-token | 用序列级几何平均概率比做裁剪；token 版允许 token 级优势 | §4.3(2) | `algos.gspo_token_loss` |
| 锚点状态 | GiGPO 中决策前已持有的帖子集合 | §4.3(2) | `Step.anchor_key()` |
| OPD | 教师对学生生成的 token 打分，给出 token 级优势 | §4.3(2) | `algos.opd_token_advantage` |
| 训推偏差 | 推理引擎与训练引擎对同一 token 算出的概率不同 | §1.3 | — |
| EP | 专家并行，把 MoE 的专家分到多张卡上 | §5.1 | `scripts/sft/` |
| singleflight | 相同的在途请求合并成一次后端调用 | §5.2 | `tools/proxy.py` |
| 损失掩码 | 只对模型生成的片段计算 loss | §4.2(3) | `data/trajectory_format.py` |
| RaR | 用逐题评分量表打分的评测方式（Rubrics as Rewards） | §6.1 | `eval/rar.py` |
| P0 | 一票否决的明确硬约束 | §6.1 | `eval.P0Constraint` |
| pass@k | k 次采样中至少一次通过的概率 | §6.3 | `eval.pass_at_k` |
| TTFT | 首字时延，主要受输入长度（prefill）影响 | §4.2(1) | — |
