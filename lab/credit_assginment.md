# Credit-assignment experiments for ALFWorld

> 记录文件名沿用当前仓库中的 `credit_assginment` 拼写。

## 当前方法

我们保持 ALFWorld 的 multi-turn rollout、工具调用和环境交互流程不变，只改进 GRPO 的训练期 advantage estimator。

`pte_grpo` 在每条 rollout 上记录每轮的 `prefill_tokens` 和 `decode_tokens`，并用 PTE 估计推理成本：

```text
PTE = sum_i(prefill_i + gamma * context_i * decode_i)
```

在同一个 GRPO prompt group 内：

1. 只对成功样本进行 PTE 排序；
2. 成功样本的 utility 为 `score * (1 - cost_coef * normalized_relative_PTE)`；
3. 失败样本 utility 为 0；
4. 对 group utility 做标准化后，沿 response mask 形成 token-level advantage。

当前实现位于：

- `verl/experimental/pte_grpo/core.py`
- `verl/experimental/pte_grpo/agent_loop.py`
- `lab/qwen2_5_alfworld_pte_grpo.sh`

该方法不依赖 GiGPO 的实现；GiGPO rollout 只作为已有 PTE 对照数据来源。

## Smoke 实验

运行目录：

`outputs/hydra/alfworld_pte_smoke/qwen2.5_1.5b_lambda010_exact/2026-09-05_15-55-17`

配置：

- Qwen2.5-1.5B-Instruct base model
- `data.train_max_samples=64`
- `rollout.n=4`
- `train_batch_size=16`
- `total_training_steps=4`
- `cost_coef=0.10`
- `gamma=0.00704`
- `save_freq=16`

结果：

| 指标 | 结果 |
|---|---:|
| 训练进度 | 4/4 |
| 验证任务数 | 140 |
| 验证成功数 | 2 |
| validation success rate | 1.43% |
| validation mean turns | 14.25 |
| validation mean PTE | 40,224 |
| validation mean decoded tokens | 1,029.7 |
| 训练 rollout mean turns | 10.67 → 7.89 → 8.72 → 8.42 |

训练日志显示每个训练 step 的 reward/score 都为 0，`actor/pg_loss=0`，说明这个小 batch 中没有形成成功/失败混合的 prompt group，PTE 排序没有有效学习信号。因此该实验只证明了 Ray/vLLM/FSDP、PTE metadata 传递、estimator 注册和 actor update 链路正确，不能作为 PTE 效果结论。

另一个重要观察是第 4 步 checkpoint 保存约 1,661 秒，后续实验必须提高 `save_freq`，否则保存开销会主导总实验时间。

## 已有 PTE 对照

从已有 GRPO/GiGPO validation rollout 重建 PTE（H100 operational intensity=756.5，Qwen2.5-1.5B，gamma=0.00704）：

| 指标 | GRPO | GiGPO |
|---|---:|---:|
| mean PTE | 1284.6 | 829.3 |
| median PTE | 626.0 | 320.6 |
| mean PTE on successful tasks | 929.7 | 607.6 |
| mean decoded tokens | 153.8 | 116.3 |

GiGPO 在 120/140 个任务上 PTE 更低；两者都成功的任务上 PTE 约降低 29.7%。这只是 rollout 对照，不等价于 PTE-GRPO 的训练收益。

## 下一步实验

### 1. `success_signal`

目的：确认在已有较强 checkpoint 上，group 内是否能获得成功/失败混合样本，以及 PTE estimator 是否产生非零 advantage。

- 初始化：已有 `global_step_222` GRPO 或 GiGPO checkpoint
- `rollout.n=8`
- `train_batch_size=64`
- `save_freq=1000`
- `test_freq=10`
- 记录 success-group fraction、advantage std、PTE/turns

该历史续训实验的配置和结果保留在本节；一次性启动脚本已在实验完成后删除。

结果（`global_step_222 → 230`，`cost_coef=0.10`）：训练 rollout 的
success rate 为 0.92--0.97，advantage 和 `actor/pg_loss` 始终非零，故
PTE 信号实际参与了策略更新。最终 140 个 validation task 与 step-222 的
GiGPO checkpoint 相比，success 均为 136/140；mean PTE 从 8,838.5 降至
8,169.1（-7.6%），successful-task mean PTE 从 7,582.0 降至 6,921.7
（-8.7%），mean turns 从 9.19 降至 8.71。这是鼓励性结果，但尚不能归因：
需要下面的 matched GRPO continuation control 排除短期训练和 rollout
随机性的影响。

## Matched GRPO continuation control

两个实验均从相同的 GiGPO `global_step_222` 恢复、使用 512 个训练样本、
`rollout.n=8`，并在 step 230 对相同的 140 个 task 验证。PTE 采用逐任务
配对的相同重建方式。

| 指标 | vanilla GRPO | PTE-GRPO (`cost_coef=0.10`) |
|---|---:|---:|
| success | 137/140 | 136/140 |
| mean PTE | 8,637.7 | 8,169.1 |
| mean PTE（两者均成功的 135 task） | — | -10.1% vs GRPO |
| mean turns | 9.09 | 8.71 |
| mean decoded tokens | 115.2 | 110.8 |

因此 PTE-GRPO 相对 matched GRPO 的全任务 mean PTE 低 5.4%，mean turns
低 4.1%，且两者均成功 task 的 PTE 低 10.1%。但 vanilla GRPO 多成功了 1
个 task，故该结果支持效率方向、却不满足“success rate 不低于 baseline”
的严格判定；不能把它报告为无损加速。下一轮应以多个 seed 和 cost 系数
消融确认 Pareto 关系，并优先测试较弱的 cost coefficient（0.05）。

按研究目标，success rate 允许小幅下降而非必须严格持平。因此这个 matched
结果符合当前预期：以 1/140（0.71 个百分点）的成功率差异，换取 5.4% 的
全任务 PTE 降幅、4.1% 的平均 turn 降幅和 10.1% 的共同成功任务 PTE 降幅。
下一阶段的评价应报告 efficiency--success Pareto 曲线，而非将 success
严格持平作为唯一通过条件。

## From-scratch run

`lab/qwen2_5_alfworld_pte_grpo.sh` 从 Qwen2.5-1.5B-Instruct
base model 直接训练一个完整 ALFWorld epoch（`train_batch_size=16`、
`rollout.n=8`），不加载任何 GiGPO 或 continuation checkpoint。它检验
PTE credit assignment 能否在能力形成阶段发挥作用；`save_freq=50`，以保留
中间 checkpoint 供按训练阶段做效率曲线分析。

### Step-200 comparison with from-scratch GiGPO

当前 PTE-GRPO 的 `step 200` validation 与已有从零训练 GiGPO
`global_step_222` validation 在相同 140 个 task 上逐项配对。两者成功率均为
136/140；以下 PTE 使用相同 tokenizer、cache-miss 假设与 gamma 重建。

| 指标 | GiGPO step 222 | PTE-GRPO step 200 | PTE-GRPO 相对变化 |
|---|---:|---:|---:|
| mean PTE | 8,838.5 | 8,258.8 | -6.6% |
| successful-task mean PTE | 7,582.0 | 6,958.2 | -8.2% |
| mean turns | 9.19 | 8.76 | -4.7% |
| mean decoded tokens | 116.3 | 111.7 | -4.0% |
| both-success PTE（134 task） | baseline | — | -10.6% |

PTE-GRPO 在 34 个 task 的 PTE 更低、79 个相同、27 个更高；在尚少两个
training step 的情况下保持相同成功率且 PTE 更低，支持其优于当前 GiGPO
baseline 的效率方向。该结论目前仅来自单次 run，仍需在 step 222 后和多个
seed 下复现。

### From-scratch training curve and analysis procedure

运行目录：
`outputs/hydra/alfworld_pte_grpo_from_scratch/qwen2.5_1.5b_cost0.10_full/2026-09-06_10-37-06`。
训练从 Qwen2.5-1.5B-Instruct 初始化，不恢复任何 checkpoint；配置为
`train_batch_size=16`、`rollout.n=8`、`cost_coef=0.10`、`save_freq=50`。
按用户要求在 step 200 checkpoint 和 validation 完成后停止，未继续到 222。

| step | success | mean turns |
|---:|---:|---:|
| 10 | 8/140 | 24.56 |
| 20 | 4/140 | 32.83 |
| 30 | 27/140 | 30.04 |
| 50 | 83/140 | 19.63 |
| 80 | 123/140 | 11.36 |
| 100 | 126/140 | 10.69 |
| 140 | 133/140 | 9.34 |
| 180 | 138/140 | 8.67 |
| 190 | 136/140 | 8.50 |
| 200 | 136/140 | 8.76 |

分析流程如下：

1. 每 10 steps 保存的 140 条 validation JSONL 按 `score>0` 计算成功数，
   从 transcript 中恢复 turn 数；曲线显示能力在 step 30--100 快速形成，
   step 140 后进入 131--138/140 的平台区。
2. 使用 `lab/analyze_alfworld_pte.py`、同一 Qwen tokenizer 和
   `gamma=0.0070423` 重建每轮 prefill/decode PTE。
3. 将 PTE-GRPO step 200 与从零 GiGPO step 222 的 140 个相同任务按顺序
   配对；success 均为 136/140。比较全任务、成功任务和共同成功任务，避免
   用失败轨迹长度变化伪装效率收益。
4. 使用 `lab/analyze_alfworld_pte.py` 恢复动作序列、定位首次动作
   分叉，并分别统计分叉后的 tail PTE/turns、重复动作和失败环境反馈。
5. 结果显示共同成功任务 PTE 降 10.6%，且主要差异集中在首次分叉后的
   tail；因此提出真实环境 action-deletion intervention，而不是继续使用
   trajectory/turn length heuristic。

### 2. Matched `grpo_continuation` control

目的：将 PTE-GRPO 的验证变化与同一 GiGPO `global_step_222`、相同训练样本数、`n=8`、更新步数（222 → 230）下的 vanilla GRPO 区分开。

该配对续训控制的配置和结果保留在本节；一次性启动脚本已在实验完成后删除。

### 3. `cost_ablation`

目的：在相同 rollout seed/config 下比较 `cost_coef ∈ {0, .05, .10, .20}`，区分纯 GRPO、弱 PTE 正则和强 PTE 正则。

每个系数单独运行，脚本名应包含系数，避免覆盖日志和 checkpoint。

### 4. `counterfactual_credit`

目的：比较 group 内最低 PTE 成功轨迹作为 counterfactual baseline，验证 PTE credit 是否能减少无效后续 turn；该实验需要在当前 PTE estimator 稳定产生非零 advantage 后进行。

核心指标：success rate、mean turns、mean PTE、successful-task PTE、最后四分之一 turns 的 PTE share，以及首次 trajectory divergence turn。

## 判定标准

只有在以下条件同时满足时，才认为方法值得扩大到正式训练：

- 至少 20% 的 prompt group 含有成功/失败混合样本；
- PTE estimator 的 group advantage 标准差显著大于 0；
- success rate 不低于 vanilla GRPO；
- 在相同成功率下，平均 PTE 或平均 turns 有稳定下降；
- 至少 3 个随机种子方向一致。

## Turn-level counterfactual extension

### Existing-data diagnosis

用 `lab/analyze_alfworld_pte.py` 对 GiGPO step 222 与 PTE-GRPO
step 200 的 134 个共同成功任务做动作级对齐：

- 56/134 个任务的动作序列发生分叉，其余 78 个完全相同；
- 平均共同前缀为 4.77 turns；
- 分叉后 GiGPO/PTE-GRPO 的 tail PTE 分别为 3,817.7/3,007.7（-21.2%）；
- tail turns 为 3.70/3.10（-16.3%）；
- 每条成功轨迹的重复动作由 0.93 降至 0.74（-20.2%）；
- 带失败反馈的环境转移由 1.04 降至 0.78（-25.7%）。

这说明主要 headroom 位于策略首次分叉后的冗余/无效 turn，而不是统一缩短
每轮文本。仅按 trajectory length 或 turn length 奖励会把必要步骤和冗余步骤
混在一起。

### Proposed method: interventional PTE credit (working name)

对成功 on-policy trajectory `tau`，定义
`U(tau) = R(tau) - lambda * PTE(tau)`。训练期间 reset 同一环境并重放已有
action sequence，构造删除第 `t` 个 action 的 `tau^{-t}`；不调用 LLM。
局部边际 credit 为：

```text
m_t = U(tau) - U(tau^{-t})
```

- 若删除后仍成功，`m_t < 0`，原 turn 是可避免计算，应得到负 credit；
- 若删除后失败，成功差值主导 `m_t > 0`，该 turn 是当前 suffix 下的必要动作；
- 对候选 turn 做预算化 deletion，可得到 success-preserving causal skeleton。
  候选排序不能只使用该 turn 的局部 PTE：早期 action 即使自身较短，删除后
  也会降低所有 suffix turns 的 prefill。应使用包含 suffix context 外部性的
  marginal-PTE 上界，或在预算内同时覆盖早期 turn 与高局部 PTE turn。

策略更新将 group-level outcome advantage 与局部干预 credit 分离：

```text
A_{i,t} = A_i^outcome + beta * normalize_within_trajectory(m_{i,t})
```

只把 `A_{i,t}` 写到对应 assistant turn 的 token span；environment observation
仍 mask 掉。失败轨迹继续使用 outcome advantage，避免早期成功稀疏时无信号。

### Novelty boundary

该方向必须明确区别于：GiGPO 的重复状态 anchor grouping、一般 multi-turn
turn-level advantage、iStar 的隐式 PRM、PGPO/G2PO 的状态势能/图 TD credit，
以及 PlanPO 的成功轨迹/单轮长度相对优势。预期独立贡献是：

1. 对 agent action 做真实环境 intervention，而非从 observational return 推断；
2. 用硬件相关 PTE 作为反事实 utility，而非 turn/token length；
3. counterfactual 只需环境 replay，无额外 LLM rollout、critic 或测试时开销；
4. success-preserving trajectory minimization 同时产生可解释的 causal skeleton。

首个验证实验应离线对 step-200 成功轨迹执行 action deletion，测量：可删除
turn 比例、删除后 PTE 节省、必要/冗余标签与重复动作及失败反馈的对应关系。
若至少 10% 的 turn 可在保持成功时删除，再实现在线 token-span advantage。

### Action-deletion replay smoke

脚本：`lab/validate_alfworld_action_deletion.py`。它把 step-200 JSONL 按行与
`valid_seen.parquet` 的 game file 对齐，先无修改重放动作验证确定性；随后
只在可复现成功轨迹上测试单 turn 删除，并按原始 turn PTE 从高到低做贪心
删除。反事实 PTE 使用删除后真实环境 observations 重新计算，不调用 LLM。

8 条成功轨迹 smoke 的原始成功复现为 8/8，无 task mapping mismatch。在
43 个被测 turns 中，15 个删除后仍成功（34.9%）；5/8 个任务至少含一个
可删除 turn。贪心 success-preserving deletion 删除 15/52（28.8%）的全部
turns，任务平均 PTE 降 23.3%。结果超过 10% 的预设继续门槛，下一步扩大到
32 条成功轨迹，检查置信度和任务类型覆盖。

### Action-deletion expanded validation (32 successes)

结果文件：
`outputs/analysis/alfworld_action_deletion/step200_success32.json`。实验仍使用
step-200 from-scratch PTE-GRPO validation，取其中前 32 条成功轨迹；每条
轨迹最多优先测试 8 个原始 PTE 最高的 turns。实验不运行模型，只做真实
ALFWorld 环境 reset/replay，运行约 9 分钟。

分析过程和逐层 sanity check 如下：

1. 先完整重放原动作序列。32/32 条均再次成功，且重建 PTE 与日志最大绝对
   误差仅 `9.1e-13`，说明 dataset 行映射、环境确定性和 PTE 计算一致，后续
   删除结果不是 replay mismatch。
2. 对 174 个候选 turn 分别做单点删除并重放 suffix。48 个删除后仍成功，
   可删除率为 27.6%；朴素 Wilson 95% 区间为 `[21.5%, 34.7%]`。turn 属于
   同一任务并非独立样本，因此该区间只作为描述性 sanity check。
3. 在任务层面，18/32（56.2%）至少含一个可删除 turn；描述性 Wilson 95%
   区间为 `[39.3%, 71.8%]`。这排除了收益只来自一两条极端长轨迹的解释。
4. 按原始 turn PTE 从高到低做 success-preserving 贪心删除，共删除
   48/186 turns（25.8%）。32 个任务的平均 PTE 降幅为 23.1%，中位数为
   22.4%，最大为 57.4%；18 个任务得到正收益，其余 14 个保持不变。
5. 单次成功删除节省 PTE 的均值/中位数分别为 1,001.9/973.9，范围为
   689.8--1,431.4。收益量级明显大于数值误差，也与 8 条 smoke 的平均
   PTE 降幅 23.3% 基本一致。

因此，该实验支持两个较窄但关键的结论：第一，成功轨迹内确实存在可由真实
环境干预识别、且数量显著超过 10% 门槛的冗余 action；第二，删除标签对应的
不是纯 token-length proxy，而是保持任务成功前提下约 23% 的可实现 PTE
headroom。这足以继续实现 turn-local interventional credit 的训练验证。

它目前**不能**证明新 credit 一定能被策略优化吸收，也不能证明跨 seed、跨
任务分布或对 vanilla GRPO 的显著优势。下一阶段应先做小规模在线训练：对
成功 rollout 预算化测试高 PTE turns，把删除边际只写回对应 assistant token
span；与相同训练流程、环境交互预算和 seed 的 trajectory-level PTE-GRPO
比较 success--PTE Pareto 曲线。若在线收益不存在，则说明离线 oracle
headroom 存在，但 credit estimator 或优化接口无效。

### Full-turn control and budget-ranking correction

结果文件：
`outputs/analysis/alfworld_action_deletion/step200_success32_all_turns.json`。
为排除 top-8 高局部 PTE 筛选造成的偏差，对相同 32 条轨迹的全部 186 个
turn 做了 deletion；32/32 原轨迹仍可复现成功，无 mismatch。

| 指标 | top-8 local-PTE candidates | all turns |
|---|---:|---:|
| tested turns | 174 | 186 |
| success-preserving deletions | 48 (27.6%) | 56 (30.1%) |
| tasks with removable turn | 18/32 | 18/32 |
| greedy removed turns | 48/186 (25.8%) | 56/186 (30.1%) |
| mean task PTE reduction | 23.1% | 25.4% |

新增 8 个可删除 turn 只出现在 4 条长度为 9--15 的轨迹，均包含此前被
top-8 排除的早期 turn。对这 4 条轨迹，全 turn 贪心相对 top-8 的 PTE 降幅
分别由 37.3%→60.1%、57.4%→77.2%、36.4%→57.6% 和 55.4%→65.5%。

这否定了“局部 turn PTE 越高，干预价值越高”的简单排序假设。PTE 含
prefill，删除早期 turn 会同时缩短后续每一次 prefill；因此真实的删除边际
包含 suffix context 外部性。好消息是 top-8 在总体平均上只漏掉 2.3 个百分点，
说明预算化干预仍可行；但正式方法应将候选优先级改为例如：

```text
priority_t = local_PTE_t + suffix_turns_t * context_tokens_removed_t
```

该式只是无需环境 replay 的排序上界，不直接作为 credit；最终标签仍来自
真实 action-deletion 后的 success 与重算 PTE。下一次验证应对 validation
成功轨迹做随机或按 ALFWorld task type 分层抽样，避免当前“前 32 条成功”
的顺序偏差，并比较固定 replay budget 下 local-PTE、suffix-aware 与随机
候选排序的 oracle-recall/PTE-regret。

### 完整 trajectory 与 action 删除标记

原始 deletion 结果 JSON 只保存汇总统计和 turn index，没有直接展开每条 trajectory。为便于
组会检查，已根据同一份 step-200 JSONL 重新导出逐轮标注文件：每一行保留完整原始
`output`，并为每个 turn 写入 action、local-PTE、是否被测试、单 action 删除后是否仍成功、
以及是否被贪心删除；另外单独保存刚进入游戏时的初始 observation（来自 `input`）。

| 实验 | 完整逐轮 JSONL | Markdown 示例 |
|---|---|---|
| top-8 local-PTE | `outputs/analysis/alfworld_action_deletion/step200_success32_annotated.jsonl` | `step200_success32_annotated.md` |
| all-turns | `outputs/analysis/alfworld_action_deletion/step200_success32_all_turns_annotated.jsonl` | `step200_success32_all_turns_annotated.md` |

Markdown 中展示第一条含可删除 action 的完整 trajectory；JSONL 则覆盖全部 32 条任务。标记
含义为：`可删除` 表示单独删除该 action 后 replay 仍成功，`贪心删除` 表示顺序删除实验
最终接受该 action，`未测试` 只出现在 top-8 预算之外的 action，其余 action 是测试过但
删除会破坏成功的必要 action。导出脚本为
`lab/export_action_deletion_trajectories.py`。

## RECAP-GRPO from-scratch 在线训练：step 200

### 实验设置

本实验首次把预算化 action-deletion replay 和 turn-local credit 放进完整的
from-scratch 训练。运行目录为：

```text
outputs/hydra/alfworld_recap_grpo_from_scratch/
qwen2.5_1.5b_replay2_weight0.10/2026-09-07_00-10-55
```

主要配置如下：

- base model：Qwen2.5-1.5B-Instruct；
- `rollout.n=8`，train batch size 为 16；
- 每条成功轨迹最多 replay 2 个候选 turn；
- `local_weight=0.10`，trajectory cost coefficient 为 `0.10`；
- validation 每 10 steps，checkpoint 每 50 steps；
- validation 不执行 deletion replay，测试时推理流程没有改变；
- PTE 仍按 `gamma=0.0070423`、每次环境交互后 cache miss 的相同口径重建。

以下分析使用 step 200 的 140 个 `valid_seen` 任务。开发阶段使用的
trajectory-only 版本只作为内部消融，用来判断 turn-level credit 是否真的
带来增益；论文中不把它包装成一个单独方法。论文主方法统一称为
**RECAP-GRPO**，外部 baseline 使用 GRPO、GiGPO 等已有方法。

### 聚合结果

| 指标 | RECAP-GRPO step 200 | trajectory-only 内部消融 step 200 | GiGPO step 222 |
|---|---:|---:|---:|
| success | 136/140 (97.14%) | 136/140 (97.14%) | 136/140 (97.14%) |
| mean PTE | **8,142.6** | 8,258.8 | 8,838.5 |
| successful-task mean PTE | **6,842.1** | 6,958.2 | 7,582.0 |
| median PTE | **5,958.5** | 5,997.3 | 6,049.7 |
| mean turns | **8.664** | 8.757 | 9.193 |
| mean decoded tokens | **110.6** | 111.7 | 116.3 |
| successful last-quarter PTE share | **0.3454** | 0.3484 | 0.3505 |

相对 GiGPO step 222，RECAP-GRPO 在少训练 22 steps 且成功率相同的情况下，
全任务 mean PTE 降低 7.9%，成功任务 mean PTE 降低 9.8%，mean turns
降低 5.7%，mean decoded tokens 降低 4.9%。在两者共同成功的 134 个任务
上，RECAP-GRPO 的配对 mean PTE 低 8.7%，turns 低 5.8%，decoded tokens
低 4.9%。这是当前最强的正面结果，说明完整方法至少保持了此前相对 GiGPO
的效率优势。

### 为什么不能仅凭聚合均值断言 turn-level credit 有效

RECAP-GRPO 相对 trajectory-only 内部消融的聚合 mean PTE 低 1.4%，成功
任务 mean PTE 低 1.7%，turns 和 decoded tokens 均约低 1%。但是两种方法
各自有 3 个独有的成功任务，成功任务的难度和轨迹长度不同，因此聚合均值
可能被任务集合差异影响。

只保留双方共同成功的 133 个相同任务后：

| 配对指标 | RECAP-GRPO 相对 trajectory-only 内部消融 |
|---|---:|
| mean PTE | **+2.59%（更高）** |
| mean turns | +1.64% |
| mean decoded tokens | +1.84% |
| RECAP PTE 更低 / 更高 / 相同 | 20 / 32 / 81 tasks |
| mean paired PTE difference | +173.7 |
| median paired PTE difference | 0.0 |

step 160、170、180、190、200 的共同成功任务 PTE 相对变化依次为
`-0.27%`、`+6.43%`、`+3.81%`、`+5.33%`、`+2.59%`。除 step 160 外，
后半程没有观察到 turn-level 版本稳定优于内部消融。对 step 200 配对任务做
20,000 次 task-level bootstrap，PTE 相对差异的描述性 95% 区间为
`[-3.62%, +9.09%]`，包含 0；单个 seed 不能区分小幅真实收益与训练波动。

因此，当前结果支持“RECAP-GRPO 相比 GiGPO 在不损失成功率时降低推理
PTE”，但还不能支持“turn-level counterfactual credit 本身进一步优于只用
trajectory efficiency signal”。后者必须通过多 seed、配对任务和针对局部
credit 的消融来验证。

### 训练开销

对 step 161--199 排除 validation steps 后统计 TensorBoard 墙钟时间：

| 指标 | RECAP-GRPO | trajectory-only 内部消融 |
|---|---:|---:|
| mean training step time | 304.9 s | 162.4 s |
| median training step time | 299.2 s | 160.0 s |
| mean generation/agent-loop time | 237.9 s | 96.9 s |
| mean train-batch turns | 12.13 | 10.51 |

RECAP-GRPO 的平均训练 step 增加约 88%。该开销来自训练期间的真实环境
replay 和更长的部分训练轨迹，不会增加部署时的推理步骤；但以当前结果看，
turn-local credit 相对内部消融的额外收益尚不足以证明这部分训练成本合理。

当前日志还没有分别记录 `replay_count`、`removable_turn_count` 和
`recap_replay_time`，所以无法把约 88% 的墙钟增量精确拆成环境 replay、
轨迹长度差异和系统波动。下一轮必须补齐这三个指标，并记录每个被删除 turn
实际节省的 `delta_PTE`。

### 截至 step 200 的判断与下一步门槛

当前实验达到以下较弱目标：完整 RECAP-GRPO 可以稳定训练；成功率达到
97.14%；相对 GiGPO 的 PTE、turns 和 decoded tokens 均有清晰改善；测试时
流程保持不变。

但核心 turn-level 假设尚未通过。跑完 step 222 后仍应首先查看共同成功任务
的 paired PTE，而不是只看全任务均值。若最终仍没有超过内部消融，下一轮
不应直接扩大 replay budget；应优先：

1. 把二值 `removable -> -1` 改为由实际 `delta_PTE` 决定的连续局部 credit；
2. 在 trajectory 内归一化局部 credit，避免固定惩罚与不同任务的 PTE 尺度
   不匹配；
3. 增加 replay/removable/time 日志，确认局部信号的覆盖率与训练成本；
4. 使用至少 3 个 seeds，并报告 success--PTE Pareto、共同成功任务 paired
   PTE 和真实推理时间；
5. 将 trajectory-only 配置在论文中呈现为 `w/o turn-level counterfactual
   credit` 消融，不作为独立命名的方法。

## RECAP-GRPO V2 完整训练与对比分析

### 实验与 checkpoint 选择

V2 使用连续 counterfactual utility credit，从 Qwen2.5-1.5B-Instruct
训练到 222 steps：

```text
outputs/hydra/alfworld_recap_cfutility_from_scratch/
qwen2.5_1.5b_replay2_signed_delta_beta0.10/2026-09-07_18-23-30
```

配置与 V1、GiGPO 和 `w/o turn-level credit` 内部消融在 train/validation
batch size、`rollout.n=8`、学习率、最大 response、最大 turns 和 validation
temperature 上一致。V2 与 V1 的 replay budget 都为 2，因此二者的训练计算
预算基本匹配。

V2 的最佳 validation checkpoint 是 **step 200**，不是最后的 step 222：

| step | success | mean PTE | successful PTE | mean turns | decoded tokens |
|---:|---:|---:|---:|---:|---:|
| 150 | 137/140 | 7,460 | 6,412 | 8.09 | — |
| 160 | 133/140 | 8,705 | 6,252 | 8.76 | 112.4 |
| 180 | 134/140 | 8,552 | 6,393 | 8.69 | 110.9 |
| 190 | 137/140 | 7,376 | 6,333 | 8.06 | 103.2 |
| **200** | **137/140** | **7,232** | **6,281** | **8.01** | **102.7** |
| 210 | 135/140 | 8,030 | 6,392 | 8.49 | 108.1 |
| 220 | 135/140 | 8,332 | 6,604 | 8.60 | 109.4 |
| 222 | 135/140 | 8,147 | 6,456 | 8.56 | 108.8 |

step 190 和 200 连续处于较好区域，因此 step 200 不是单次孤立尖峰；但继续
训练后 success 和全任务 PTE 均退化，说明当前设置存在轻微过训练。后续应保留
基于 validation 的 checkpoint selection，并将选出的 step 200 只在
`valid_unseen` 上做一次最终报告，避免在同一 split 上选择和报告最好结果。

### 训练结果的分析过程（可复查）

本节记录本次结论是怎样得到的，避免只保留最终表格而无法复查。分析使用了
合并后的 `lab/analyze_alfworld_pte.py` 中的
同一套 PTE 与动作解析口径。

首先读取各方法在 `valid_seen` 上的 140 条结果，并按数据集行号一一对齐。
主要文件为：

```text
RECAP V2:
outputs/validation_log/alfworld_recap_cfutility_from_scratch/
qwen2.5_1.5b_replay2_signed_delta_beta0.10/2026-09-07_18-23-30/200.jsonl

RECAP V1:
outputs/validation_log/alfworld_recap_grpo_from_scratch/
qwen2.5_1.5b_replay2_weight0.10/2026-09-07_00-10-55/200.jsonl

w/o turn-level credit:
outputs/validation_log/alfworld_pte_grpo_from_scratch/
qwen2.5_1.5b_cost0.10_full/2026-09-06_10-37-06/200.jsonl

GiGPO 原训练过程中的 validation（主要对比）：
outputs/validation_log/alfworld_gigpo/
qwen2.5_1.5b_lr1e-6_kl0.001/2026-09-03_19-20-36/{200,222}.jsonl

GiGPO step-222 单独重评（补充检查）：
outputs/validation_log/alfworld_gigpo_val/qwen2.5_1.5b_step_222/
2026-09-04_15-57-10/generations.jsonl
```

具体分析顺序如下：

1. 从每条记录的完整对话中恢复各轮输入、模型输出和环境反馈。使用与训练模型
   相同的 Qwen2.5 tokenizer 和 chat template 重新计数，避免简单按字符数估算。
2. 按论文采用的统一公式计算每条轨迹的 PTE：
   `sum(prefill + 0.007042327272727272 * context * decode)`。ALFWorld 每次与
   环境交互后都按下一轮需要重新读取上下文计算，不假定跨环境轮次复用缓存。
3. 对每个 checkpoint 计算成功任务数、140 个任务的平均 PTE、只在成功任务上
   的平均 PTE、平均 turns 和生成 token 数。沿 step 150--222 查看曲线，而不是
   只看最后一步；据此选择连续两个点都较好的 step 200。
4. 为避免失败长轨迹扭曲均值，再只保留两种方法都成功的同一批任务，逐任务做
   差值比较。记录 RECAP 更省、baseline 更省和二者相同的任务数，并以任务为
   单位重复抽样 20,000 次计算 95% 区间。这里的区间只能表示这 140 个任务上的
   波动，不能替代多个训练 seed。
5. 从 `<action>...</action>` 中取出实际动作，找到两种方法动作序列第一次不同
   的位置。分别统计该位置之后的 turns、PTE、重复动作，以及环境返回的失败
   提示，从而判断 PTE 下降来自少绕路，还是仅仅来自输出文字变短。
6. 从 `assets/datasets/alfworld/valid_seen.parquet` 的
   `extra_info.task_type` 读取任务类别，分别比较六类任务，检查总体收益是否只由
   某一类任务贡献。
7. 从训练 TensorBoard 日志汇总 deletion replay 的触发数、删除后仍成功的
   次数、删除后失败的次数和实际节省的 PTE。训练耗时只统计非 validation、
   非保存 checkpoint 的 step；`recap/replay_seconds` 因包含并发等待，不直接
   当成总训练时间。

这套流程把三个问题分开回答：任务是否还能完成、双方都完成时谁更省计算、
模型具体少做了哪些动作。当前最强证据是共同成功任务上的 PTE 和行为变化；
当前仍缺的证据是额外训练 seeds、冻结后的 `valid_unseen` 结果，以及拆开的
模型推理真实时间。

### 与外部 baseline 的主要结果

论文主结果应使用与 V2 相同训练流程产生的 in-training validation。GiGPO
后来单独重评的 step-222 文件与其原训练 validation 存在差异，且缺少对应
Hydra config，因此只作为补充敏感性检查，不作为唯一 baseline 数字。

| 指标 | RECAP V2 step 200 | GiGPO step 200 | GiGPO step 222 | GRPO step-222 reval |
|---|---:|---:|---:|---:|
| success | 137/140 | **139/140** | **138/140** | 133/140 |
| mean PTE | **7,232.5** | 7,726.1 | 7,814.4 | 10,633.5 |
| successful-task mean PTE | **6,280.9** | 7,423.0 | 7,209.8 | 8,700.0 |
| mean turns | **8.007** | 8.521 | 8.557 | 10.236 |
| decoded tokens | **102.7** | 109.1 | 109.1 | 153.8 |

相对相同步数的 GiGPO step 200，V2 少成功 2/140 个任务（-1.43 percentage
points），但全任务 mean PTE 低 6.4%，成功任务 mean PTE 低 15.4%，mean
turns 低 6.0%，decoded tokens 低 5.9%。在双方共同成功的 136 个任务上：

- paired mean PTE 降低 **14.3%**；
- paired turns 降低 10.1%；
- paired decoded tokens 降低 9.5%；
- V2/GiGPO PTE 更低分别为 65/42 个任务，29 个完全相同；
- task bootstrap 95% 区间为 `[-24.08%, -4.36%]`。

相对 GiGPO step 222，V2 step 200 少成功 1/140 个任务，全任务 mean PTE
低 7.4%；共同成功的 136 个任务上 paired PTE 低 11.8%，task bootstrap
95% 区间为 `[-19.56%, -3.77%]`。相对 GRPO step-222 revalidation，V2
多成功 4 个任务；共同成功任务 PTE 低 25.3%，区间为
`[-33.66%, -17.07%]`。

最终 step 222 并非失败：它相对 GiGPO step 222 在共同成功的 134 个任务上
PTE 仍低 9.6%，区间为 `[-18.46%, -0.10%]`。但其成功率为 135/140，低于
GiGPO 的 138/140，而且失败长轨迹使全任务 mean PTE 反而高 4.3%。因此正式
比较应选择 V2 step 200，并明确 checkpoint selection 规则。

### V2 是否改进了 turn-level credit

V2 step 200 与相同步数的 V1 二值 credit 比较：

| 指标 | V2 | V1 | V2 相对变化 |
|---|---:|---:|---:|
| success | 137/140 | 136/140 | +1 task |
| mean PTE | 7,232.5 | 8,142.6 | -11.2% |
| successful-task PTE | 6,280.9 | 6,842.1 | -8.2% |
| mean turns | 8.007 | 8.664 | -7.6% |
| decoded tokens | 102.7 | 110.6 | -7.1% |

在共同成功的 134 个任务上，V2 paired PTE 低 7.0%、turns 低 5.2%、decoded
tokens 低 5.0%；V2/V1 PTE 更低分别为 65/41 个任务，28 个相同。task
bootstrap 95% 区间为 `[-13.84%, +0.61%]`，非常接近但仍包含 0。因此单个
seed 支持连续 credit 优于二值 credit 的方向和效果量，但还不能给出 seed-level
稳定性结论。

与 `w/o turn-level credit` 内部消融比较，V2 多成功 1 个任务，聚合 mean PTE
低 12.4%，成功任务 PTE 低 9.7%，turns 低 8.6%，decoded tokens 低 8.1%。
共同成功的 133 个任务上 paired PTE 低 5.7%，task bootstrap 区间为
`[-11.17%, +0.25%]`。该结果已经明显强于 V1 的对应配对结果，但仍需至少
两个额外训练 seeds 才能把方法贡献与训练随机性分开。

### 行为层面的变化

只在双方共同成功任务上分析动作序列。相对 GiGPO step 200：

| 指标（每条成功轨迹） | RECAP V2 | GiGPO | 相对变化 |
|---|---:|---:|---:|
| repeated actions | 0.596 | 1.140 | -47.7% |
| failure observations | 0.684 | 0.787 | -13.1% |
| divergence 后 tail turns | 5.93 | 6.75 | -12.1% |
| divergence 后 tail PTE | 5,197 | 6,238 | -16.7% |

相对 V1，V2 的 repeated actions 低 12.8%，failure observations 低 18.5%，
divergence 后 tail PTE 低 8.3%。相对 `w/o turn-level credit`，repeated
actions 低 15.7%，failure observations 低 4.1%，tail PTE 低 6.8%。这说明
收益不只是把每个回答写短，而是减少了分叉后的绕路、重复操作和失败反馈。

按 ALFWorld 六种任务类型分组，V2 相对 GiGPO 在 5/6 类共同成功任务上 PTE
更低；`pick_cool_then_place_in_recep` 基本持平（+0.2%）。相对内部消融同样
在 5/6 类更低，但 `pick_heat_then_place_in_recep` 高 9.3%。因此收益覆盖多种
任务，而不是只由一个类型贡献；加热类任务是后续误差分析的重点。

### Counterfactual signal 覆盖率

222 个训练 steps 的新日志汇总如下：

- 总训练轨迹约 28,416 条；其中 18,529 条成功轨迹触发 replay（65.2%）；
- 共执行 37,058 次单 turn deletion replay；
- 5,395 次删除后仍成功且节省 PTE，可删除率为 14.56%；
- 31,663 次删除后失败，被标记为当前 suffix 下的必要 action；
- 每次成功删除平均节省约 1,630 PTE；
- removable rate 从前 50 steps 的 12.5% 上升到 step 151--200 的 15.6%，
  最后 22 steps 为 14.0%。

这证明 V2 的连续负 credit 和必要动作正 credit 都在大规模实际触发，不是配置
存在但没有训练信号。不过正/负标签明显不平衡：约 85.4% 的被测 action 是
必要 action。`signed_absmax` 避免了正 credit 完全淹没负 credit，但它与
连续 `delta_PTE` 同时改变，当前实验不能单独判断收益来自哪一项；正式消融
需要拆开 binary/continuous、negative-only/signed 和 normalization。

日志中的 `recap/replay_seconds` 是并发 trajectory 的 elapsed time 求和，包含
环境锁等待，不能直接当作训练总墙钟时间。它适合计算单 replay 的等待量，不能
与端到端训练时间直接相加。

### 训练成本与真实延迟限制

对 step 151--199 排除 validation/save steps 后：

| 方法 | mean step time | mean generation/agent-loop time |
|---|---:|---:|
| RECAP V2 | 301.8 s | 236.1 s |
| RECAP V1 | 302.8 s | 235.7 s |
| GiGPO | 227.2 s | 96.6 s |
| w/o turn-level credit | 162.3 s | 97.0 s |

V2 与 V1 的训练成本基本相同，因此连续 credit 获得了更好的模型而没有增加
replay budget。相对 GiGPO，V2 每个训练 step 慢约 33%；相对不做环境 replay
的内部消融慢约 86%。训练成本仍是论文必须正面报告的限制。

现有 `timing_s/testing` 不能证明真实推理加速：step 200 的 140-task 并行验证
墙钟时间，V2/GiGPO 分别约为 81.2/79.4 秒，V2 没有更快。该计时混合了
ALFWorld CPU 环境、并行调度和模型推理，不能否定 PTE 的模型计算收益，但也
不能替代真实速度证据。下一阶段需要固定并发度，分别记录模型 prefill/decode
时间、环境时间和单任务端到端 latency，并验证 PTE 与模型时间的相关性。

### 当前结论与阶段门槛

V2 已达到“可以进入下一 benchmark 和下一模型规模”的 proof-of-concept
门槛：相对 matched GiGPO 的 paired PTE 降幅显著，成功率只少 1--2 个任务；
相对两个内部消融的效果量为 5.7%--7.0%，并且行为分析显示重复和无效 action
减少。

但它还不是完整投稿证据，原因是只有一个训练 seed、最佳 checkpoint 在
`valid_seen` 上选择、真实 latency 尚未改善、内部消融的 task-bootstrap 区间
仍轻微跨 0。推荐立即执行顺序为：

1. 用 step-200 checkpoint 在 `valid_unseen` 上做一次冻结评估；
2. 开始第二个支持确定性 reset/replay 的 benchmark；
3. 同时补两个 ALFWorld seeds；
4. 在 1.5B 跨环境成立后扩到 Qwen2.5-3B；
5. 做 binary/continuous、negative-only/signed、normalization 和候选排序消融；
6. 增加严格拆分的真实推理 latency 实验。
