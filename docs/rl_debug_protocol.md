# RL 上线前三阶段检查

多组件耦合的 Agentic RL 很容易"静默训崩"。每次改奖励、工具或数据后，按顺序过三关再开长跑。

## 阶段一：离线校验（不跑模型）

- query、约束、评估字段完整；`tag_search` / `community_search` 的名称、参数和值域合法。
- 用对抗样例单测奖励（`tests/test_rewards.py` 是起点）：非法 JSON、缺参、越界参数、未注册工具、自引用"根据第 x 步"、
  偏题的纵向补查、伪造引用、无引用回答。每类都应得到预期的低分。
- 用黄金轨迹确认满分路径确实拿满分（`agentic-search demo` 第一行应为 R = 1.000）。

## 阶段二：在线 rollout，不做梯度更新

对少量 query、每个采样多条轨迹，检查 **拆解 → 工具调用 → 结果写回上下文 → 补查或回答 → 奖励**：

- 请求确实经过训练代理层（看 `SearchProxy.metrics()` 的 QPS、合并率、缓存命中率）。
- 工具结果被正确写回上下文；轮数和调用预算能兜底结束（`meta["truncated"]` 的比例）。
- **组内奖励方差**：`group_reward_std_stats` 的 `frac_zero_std_groups` 过高，说明题目过难/过易，或工具对任何 query 返回同样内容。
  总体奖励有变化不代表每组都有有效优势。
- `valid=0`（基础设施故障）样本的比例应接近 0。

## 阶段三：小规模全链路更新

只跑几个 step：rollout → 奖励 → 优势 → GSPO 损失 → 反传 → 更新 → checkpoint。

- loss、优势、梯度范数均有限（无 NaN / Inf）；GSPO 的 `clip_frac` 在合理范围（区间是 3e-4 / 4e-4，不是 0.2）。
- 显存、EP All-to-All 通信、工具请求负载可控。
- 新权重正确同步到 vLLM rollout 引擎（同一 prompt 在同步前后 log-prob 应变化）。
- checkpoint 能保存，并能从 checkpoint 恢复出一致的训练状态。
