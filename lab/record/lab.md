# ALFWorld Prefix 删除与 Anchor 保留实验

## 实验设置

- 数据：`assets/datasets/alfworld/valid_seen.parquet`
- 模型：Qwen2.5-1.5B，step-222 checkpoint
- 轨迹：只使用 step-222 baseline 成功轨迹，共 133 条
- 在原轨迹约 60% 处保存环境状态，并从该状态重新 rollout
- System prompt 始终保留
- 删除按完整 turn/chunk 进行，不切分单个 turn
- Full 条件保留完整历史 context

## Prefix 删除结果

| 条件 | 成功数 / 总数 | Success rate | 累计输入 tokens | 平均累计输入 tokens | 相对 Full 的失败数 |
|---|---:|---:|---:|---:|---:|
| Full | 130 / 133 | 97.74% | 556,007 | 4,180.5 | 0 |
| Drop-oldest-25% | 127 / 133 | 95.49% | 612,178 | 4,602.8 | 4 |
| Drop-oldest-50% | 124 / 133 | 93.23% | 741,548 | 5,575.5 | 7 |
| Drop-oldest-75% | 110 / 133 | 82.71% | 1,318,563 | 9,914.0 | 21 |

### 分析

删除最早的 25% 和 50% turn 时，成功率分别下降 2.25 和 4.51 个百分点，说明较早 context 中确实存在一定冗余信息，但并非全部可永久删除。删除 75% 后成功率下降到 82.71%，性能明显恶化，表明过度删除会丢失任务目标、已完成子目标或环境定位信息。

删除 context 后累计输入 token 没有下降，反而上升。这是因为重新 rollout 时模型更容易产生额外探索、重复动作或失败动作；因此“单次 prompt 更短”不等于“整条 episode 的累计输入更少”。

## Anchor + 最近一个 turn

Anchor 根据人工筛选的关键动作类型保留；同时保留截断点之前最近的一个 turn。Anchor 数量不设上限。结果文件：`outputs/prefix_drop/step_222_seen_anchors_unlimited/results.json`。

| 条件 | 成功数 / 总数 | Success rate | 累计输入 tokens | 平均累计输入 tokens | 相对 Full 的失败数 |
|---|---:|---:|---:|---:|---:|
| Full | 130 / 133 | 97.74% | 548,718 | 4,125.7 | 0 |
| Anchors + latest-1 | 119 / 133 | 89.47% | 936,291 | 7,039.8 | 13 |

Anchor 保留实验的 Full 累计输入为 548,718；由于与前一组实验使用了不同 batch 配置，Full 的 token 数存在少量运行差异，应主要比较同一组实验内部的相对结果。

### 分析

仅保留关键 anchor 和最近一个 turn 仍然造成 8.27 个百分点的成功率下降，说明当前人工 anchor 规则还不能完整替代被删除的历史 context。可能原因包括：

1. 部分关键状态信息并不出现在预定义 anchor action 中；
2. 同一个动作的历史 observation 仍可能包含物体位置、容器开闭状态等必要信息；
3. 过度压缩 context 会导致模型重新探索，增加后续 turn 和累计输入 token。

因此，当前结果不能支持“去除这些历史 turn 不降准确率”。更准确的结论是：ALFWorld 中存在可删除的冗余历史，但需要更可靠的环境状态表示或 state-based anchor，而不是只依赖动作类型筛选。

## 可复现结果文件

- Prefix 删除：`outputs/prefix_drop/step_222_seen/results.json`
- Anchor（最多保留 2 个 anchor 的探索版本）：`outputs/prefix_drop/step_222_seen_anchors/results.json`
- Anchor（不限制 anchor 数量的最终版本）：`outputs/prefix_drop/step_222_seen_anchors_unlimited/results.json`
