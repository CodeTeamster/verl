# RECAP-GRPO 方法笔记

## 1. 方法名称

**RECAP-GRPO**：Replay-based Efficiency-aware Counterfactual Action Credit for
Policy Optimization。

中文可以理解为：**基于环境重放的效率感知动作级 GRPO**。

这个名字强调三件事：

1. 训练时重新执行环境；
2. 分别判断每个 action 的作用；
3. 在保持任务成功的同时减少实际推理计算量。

## 2. 先说最重要的结论

### 最早完成的版本不是 turn-level credit assignment

在接入环境重放前，先从零训练到 step 200 的是 trajectory-level 版本。

它会先给整条轨迹计算一个分数：成功的轨迹得正奖励，成功但 PTE 很高的轨迹
会被适当扣分。然后，同一条轨迹里的所有 assistant turns 共用同一个
advantage。

例如一条成功轨迹有 5 个 turns：

```text
turn 1: 正确导航
turn 2: 无效动作
turn 3: 重复观察
turn 4: 拿起目标
turn 5: 完成任务
```

这个早期版本会把同一个轨迹分数发给这 5 个 turns。它能够鼓励模型选择
整体更短、更省 PTE 的成功轨迹，但不能直接告诉模型“turn 2 和 turn 3 是
多余的”。

### RECAP-GRPO 才是真正的 turn-level credit assignment

RECAP-GRPO 会分别测试每个 turn：暂时删除这个 action，再重新执行环境，观察
任务是否仍然成功。

- 删除后仍成功：这个 turn 很可能是多余的，应得到负 credit；
- 删除后失败：这个 turn 对当前执行过程很重要，应得到正 credit；
- 原轨迹失败、删除后成功：被删除的 turn 很可能有害，应得到较强的负 credit；
- 原轨迹和删除后的轨迹都失败：无法判断这个 turn 是否有用，暂时不给额外
  turn credit。

上面四种情况描述的是完整想法。**当前 V2 只对原轨迹成功的样本做删除实验**，
所以实际训练中只会遇到前两种。失败轨迹如何可靠地测试，仍是后续工作。

最后，只把这个 credit 写到对应 assistant turn 的 token 上，而不是整条轨迹。
因此，从算法设计上说，RECAP-GRPO 是 turn-level credit assignment。

这部分现在已经接入在线训练。第二版 RECAP-GRPO 已从基础模型完整训练 222
steps，最佳结果在 step 200：成功 137/140。它相对 GiGPO 和“不逐个测试
action”的内部版本都降低了 PTE。当前可以说“完整链路已跑通，而且一个 seed
上看到了符合预期的收益”；但还不能说结果已经稳定，因为仍缺额外 seeds 和
第二个环境。

## 3. 方法想解决什么问题

普通 GRPO 只知道一整条轨迹最后成功还是失败。

如果一条轨迹成功，里面所有 action 都会受到鼓励，包括：

- 正确且必要的 action；
- 没有作用的观察；
- 重复执行的 action；
- 走错后又恢复的绕路。

如果直接惩罚 turn 数或 token 数，又可能错误地惩罚必要步骤。复杂任务本来就
需要更多步骤，不能简单要求所有任务都变短。

RECAP-GRPO 的思路是：**不猜某个 action 有没有用，而是在训练时真的删除它，
让环境告诉我们结果。**

## 4. 为什么使用 PTE，而不只统计 token 或 turns

multi-turn agent 每一轮都要把已有上下文重新输入模型。越靠后的 turn，输入
上下文通常越长，因此一个早期多余 action 不只浪费它自己生成的几个 token，
还会增加后面每一轮的输入计算。

PTE 同时考虑：

- 每轮输入模型的上下文长度；
- 每轮生成的 token 数；
- 随着上下文增长而增加的推理计算。

所以我们真正想优化的是：

```text
任务保持成功，同时整条轨迹的 PTE 更低
```

这比“turn 越少越好”或“token 越少越好”更接近真实推理速度。

## 5. RECAP-GRPO 如何计算每个 turn 的 credit

先给一条轨迹定义一个简单分数：

```text
轨迹分数 = 是否成功 - lambda × PTE 成本
```

实际实现时只对成功轨迹施加 PTE 成本，避免模型通过快速失败获得高分。

对于第 `t` 个 turn：

```text
turn_credit_t = 原轨迹分数 - 删除 turn t 后的轨迹分数
```

直观解释：

- `turn_credit_t > 0`：保留这个 turn 更好；
- `turn_credit_t < 0`：删除这个 turn 更好；
- `turn_credit_t ≈ 0`：当前实验无法判断它的重要性。

最终训练 credit 由两部分组成：

```text
第 t 个 turn 的 credit
    = 整条轨迹的 GRPO credit
    + beta × 第 t 个 turn 的删除实验 credit
```

第一部分保证模型继续学习完成任务，第二部分负责区分必要 action 和多余 action。

## 6. 删除一个 turn 后如何重新执行

对于已经生成的动作序列：

```text
a1, a2, a3, a4, a5
```

测试 `a3` 时，在相同任务环境中执行：

```text
a1, a2, a4, a5
```

整个过程不调用 LLM 重新生成动作，只重新执行 ALFWorld 环境。因此：

- 不需要额外的模型或 reward model；
- 不需要 LLM 判断某个步骤是否正确；
- 不增加测试时开销；
- 训练时会增加一些环境执行次数。

这种实验回答的是一个比较窄的问题：**在保持其他已生成动作不变时，这个
action 能否被删除？** 它不等于“模型遇到新状态后会怎样重新规划”。论文中
不能把它夸大成完全准确的动作价值。

## 7. 为什么不能只测试当前 PTE 最高的 turns

最初的验证优先删除当前 PTE 最高的 8 个 turns，但全 turn 实验发现，这会漏掉
一些很有价值的早期 action。

原因是：早期 action 自身的 PTE 可能不高，但删除它以后，后面所有 turns 的
输入上下文都会变短，因此总节省可能很大。

所以当前代码选择待测试 turns 时，同时考虑：

```text
优先级 = 当前 turn 的成本 + 它可能给后续 turns 增加的成本
```

为了控制训练开销，V2 每条成功轨迹最多只测试 2 个 action：

- 第一个选择“它自己加上它对后续各轮的影响”最大的 action；
- 第二个选择“当前这一轮本身计算量”最大的、且尚未被选择的 action。

第一种排序使用下面这个容易理解的近似值：

```text
影响分 = 当前轮成本 + 后面剩余轮数 × 本轮新增加的文字长度
```

第二种排序只看当前这一轮的 PTE。若两种排序都选中同一个 action，第二种排序
就向下选择下一个尚未测试的 action。轨迹只有一个 action 时只测试一次。

例如一条成功轨迹包含：

```text
1. look                         # 返回了很长的房间描述
2. go to fridge
3. open fridge
4. take apple from fridge
5. move apple to table          # 当前这一轮输入最长
```

假设第 1 个 `look` 自己并不很贵，但它产生了 150 个 token，后面 4 轮每次都要
重新读到这些内容；它的“影响分”就可能最高。第 5 个 action 没有后续轮次，但
它当前读取的上下文最长，单轮 PTE 可能最高。代码会优先测试第 1 和第 5 个，
这样一个覆盖“早期留下长上下文”的风险，一个覆盖“单轮特别贵”的风险。

用一组简化数字表示就是：

| action | 当前轮成本 | 本轮新增长度 | 后面轮数 | 影响分 |
|---|---:|---:|---:|---:|
| `look` | 500 | 150 | 4 | **1,100** |
| `go to fridge` | 620 | 20 | 3 | 680 |
| `open fridge` | 700 | 15 | 2 | 730 |
| `take apple` | 780 | 12 | 1 | 792 |
| `move apple` | **850** | 10 | 0 | 850 |

第一种排序选择影响分最高的 `look`，第二种排序选择当前轮成本最高的
`move apple`。这里的数字只是帮助理解，训练时使用每条真实轨迹的 token 数和
PTE 自动计算。

当前 V2 **没有随机抽 action**。随机候选是后续可以做的对照实验，不应写成
现有实现。最终分数仍来自真实删除和环境重放，上面的影响分只决定先测试谁，
不会直接拿来训练模型。

### 一个完整的删除与打分例子

原来的成功动作是：

```text
1. go to fridge
2. open fridge
3. look
4. take apple from fridge
5. move apple to table
```

先测试第 3 个 `look`。重新开始同一个任务，只执行
`1, 2, 4, 5`。如果仍然成功，而且原轨迹 PTE 是 8,000，删除后是 6,900，
那么这个 `look` 节省了 1,100 PTE。它会得到负分：

```text
-1100 / 8000 = -0.1375
```

负分的意思是提醒模型：以后遇到相似情况，不要轻易生成这个多余的 `look`。

再单独测试第 2 个 `open fridge`。环境重新开始，只执行 `1, 3, 4, 5`。
如果因为冰箱没有打开而失败，第 2 个 action 会得到正分，表示在这条既定动作
序列里它不能被删掉。

同一条轨迹可能同时出现正分和负分。代码把正分和负分分别缩放到容易训练的
范围，再乘以 `0.10` 加到原来的整条轨迹分数上。只有被测试 action 对应的
assistant tokens 会收到这次加分或扣分：例如删除 `look` 得到的负分只写到
`<action>look</action>`，不会写到环境描述，也不会扣到其他四个 action。

这里还有一个重要边界：删掉 action 后，代码只是照原样执行后续动作，不让
模型临时重新想一条新路线。因此它判断的是“这个 action 对当前这条动作序列
是否必要”，而不是“任何情况下是否必要”。

## 8. 完整训练流程

每个训练 step 执行：

1. 像普通 multi-turn GRPO 一样生成一组 ALFWorld 轨迹；
2. 计算每条轨迹的成功奖励和 PTE；
3. 计算整条轨迹的 group-relative advantage；
4. 为每条轨迹选择少量待测试 turns；
5. 删除选中的 action，并在相同环境中重放其他 action；
6. 根据成功变化和 PTE 变化，计算每个被测试 turn 的 credit；
7. 把 turn credit 只分配给对应 assistant turn 的 tokens；
8. 继续使用原来的 GRPO loss 更新模型。

模型实际部署时，仍按照普通方式逐轮生成 action。RECAP-GRPO 不增加新的模型、
搜索模块或特殊推理步骤。

## 9. 当前已经完成的内容

### 已完成：不做逐 action 删除的内部版本

Qwen2.5-1.5B-Instruct 从基础模型训练到 step 200：

- ALFWorld validation 成功率：136/140；
- 与 from-scratch GiGPO step 222 成功率相同；
- 全任务平均 PTE 比 GiGPO 低 6.6%；
- 成功任务平均 PTE 低 8.2%；
- 两边都成功的相同任务上，PTE 低 10.6%；
- 平均 turns 少 4.7%。

这说明训练模型在保持成功率的同时减少 PTE 是可行的，但它还是让整条轨迹的
所有 action 共用同一个分数。

### 已完成：32 条成功轨迹的 action-deletion 验证

- 32/32 条原轨迹可以在 ALFWorld 中重新执行并成功；
- 对全部 186 个 turns 分别做删除实验；
- 56/186，也就是 30.1% 的 turns 删除后仍成功；
- 18/32 个任务至少有一个可删除 turn；
- 逐步删除多余 turns 后，平均 PTE 可以降低 25.4%。

这证明成功轨迹中确实存在大量可识别的多余 action，也证明 turn-level 方法有
继续实验的价值。

### 第一版在线训练已经完成的内容

- turn credit 已接入训练数据；
- assistant turn 对应 token 区间的 advantage 已写入；
- 成功轨迹的环境重放已进入每个训练 batch；
- RECAP-GRPO 已完成 from-scratch step-200 验证。

仍未完成的是：失败轨迹的 action-deletion、多随机种子、第二个 agent 环境，
以及相对内部消融稳定的 turn-level 增益。

### 已完成的 RECAP-GRPO V1

最小版本已经完成 step-200 from-scratch 训练，配置为：

- 基础模型：Qwen2.5-1.5B-Instruct；
- 只对成功轨迹做 deletion replay；
- 每条成功轨迹最多测试 2 个 turns；
- 删除后仍成功且 PTE 降低，才给该 turn 负 credit；
- 删除后失败的必要 turn 暂时不额外加分；
- turn-local weight：`0.10`；
- validation：每 10 steps；
- checkpoint：每 50 steps；
- 测试时的 agent 推理流程不变。

对应入口为：
`RECAP_VERSION=v1 bash lab/qwen2_5_alfworld_recap_grpo.sh`。

V1 相对 GiGPO step 222 在相同 136/140 成功率下，全任务 mean PTE 低 7.9%，
共同成功任务 mean PTE 低 8.7%。但相对不使用 turn-local credit 的内部消融，
共同成功任务 mean PTE 反而高 2.6%。这说明 V1 保住了整体效率训练的收益，
但简单的二值 `removable -> -1` 标签没有证明自身有效。

### 已完成：RECAP-GRPO V2

V2 不再把所有可删除 turn 都写成相同的 `-1`。每个被测试 turn 使用真实删除
结果决定加分还是扣分：

```text
删除后失败：正 credit，说明该 action 对当前执行过程必要
删除后仍成功：- delta_PTE / original_PTE
```

同一条轨迹内，加分和扣分分别按各自最大的数缩放。这样既不会
让“删除后很快失败产生的低 PTE”被误认为高效，也能让节省更多 PTE 的冗余
turn 得到更强惩罚。最后仍只把分数写到对应 action 的模型输出上：

```text
这个 action 的训练分数 = 原轨迹的训练分数 + 0.10 × 删除实验分数
```

V2 保持每条成功轨迹最多 2 次 replay，不先增加环境计算预算。它还会记录：

- 实际 replay 数；
- 可删除 turn 数和比例；
- 删除失败的必要 turn 数；
- 成功删除节省的 PTE；
- replay 墙钟时间。

统一实验入口为：
`RECAP_VERSION=v2 bash lab/qwen2_5_alfworld_recap_grpo.sh`。默认实验标识为：

```text
project: alfworld_recap_cfutility_from_scratch
experiment: qwen2.5_1.5b_replay2_signed_delta_beta0.10
```

V2 已完成 222 steps。最佳 checkpoint 是 step 200：成功 137/140，mean PTE
为 7,232，平均 8.01 turns。相对同为 step 200 的 GiGPO，成功少 2 个任务，
但共同成功任务的 PTE 低 14.3%，turns 低 10.1%；重复 action 低 47.7%。相对
`w/o turn-level counterfactual credit` 内部消融，共同成功任务 PTE 低 5.7%。

最后的 step 222 退化为 135/140、mean PTE 8,147，因此后续使用 step 200。
当前结果足以开始第二个 benchmark 和额外模型规模，但还不能作为最终论文证据：
仍需 `valid_unseen`、多 seeds、第二环境、关键消融和真实模型推理时间。

## 10. Related work 与创新边界

截至 2026-09-08，没有发现已录用论文同时使用下面这套完整设计：

```text
在真实 agent 环境中删除 action 并重新执行
+ 用任务成败和节省的 PTE 给这个 action 单独打分
+ 把分数写回对应 action 的模型输出
+ 测试时仍使用普通的逐轮推理流程
```

下面的“模型数”只统计论文方法真正训练或使用的目标模型，不把 GPT-4 教师、
闭源 API 和普通 baseline 算进去。“核心规模”只统计主要 agent 实验；括号中的
“完整规模”还包括 QA、数学、多模态或附录实验。ALFWorld 的 L0/L1/L2 属于
同一环境的不同难度，不重复计算 benchmark。

### 10.1 专注推理效率的方法

| 方法 | 核心规模 | 完整规模 | 它怎样提高效率 | 与 RECAP 的区别 |
|---|---|---|---|---|
| Beyond Accuracy / PTE，ACL 2026 | 不训练新策略；5 个工具推理 benchmark | 相同 | 提出 PTE 并分析低效模式 | RECAP 使用 PTE 产生训练信号，而不只是测量 |
| DEPO，2025 预印本 | 2 个模型；WebShop、BabyAI | 2 个模型；5 个 benchmark | 用偏好训练同时减少每轮 token 和总步数 | 不删除 action，也没有计算早期 action 对后续输入的影响 |
| RLVMR，ICLR 2026 | 2 个模型；ALFWorld、ScienceWorld | 相同 | 奖励规划、探索和纠错，间接减少重复动作 | 需要特殊思考标签、规则奖励和冷启动 SFT；RECAP 不需要 |
| RECAP 当前结果 | 1 个模型；ALFWorld | 相同 | 删除 action 后直接测量任务变化和 PTE 变化 | 当前优势是分数更具体；不足是实验规模最小 |

DEPO 是效率目标上最接近的方法，但它优化的是 token 数和 step 数。RECAP 使用
PTE，是因为一个早期 action 会被后面多轮重复读入，实际成本不只等于它自己
生成了多少 token。论文不能声称“首次研究 agent 效率”，更准确的贡献是：
**首次尝试用真实环境删除实验，为在线 multi-turn GRPO 产生逐 action 的 PTE
训练信号。** “首次”在投稿前仍需重新检索确认。

### 10.2 专注逐 action 分数的方法

| 方法 | 核心实验规模 | action 分数怎样得到 | 与 RECAP 的区别 |
|---|---|---|---|
| GiGPO，NeurIPS 2025 | 2 个模型、2 个 agent 环境；完整为 4 种模型、11 个数据集 | 比较多条轨迹中到达相同状态后的不同 action | 不主动删除 action、不重新执行环境，也不直接优化 PTE |
| POAD，NeurIPS 2024 | 推理任务 | 从公式上把整段分数分到 action/token | 没有真实 agent 环境中的删除实验 |
| SPO，NeurIPS 2025 | 文本推理任务 | 给不同推理片段分配不同分数 | 面向单次推理文本，不处理环境 action |
| HGPO，ICLR 2026 | ALFWorld、WebShop | 按状态和历史信息对 action 分组比较 | 使用已有样本分组，不执行删除后的环境 |
| SSVPO，ICLR 2026 | 推理任务 | 估计不同推理步骤对答案的贡献 | 不在真实 agent 环境删除 action，也不使用 PTE |
| RECAP | 当前为 1 个模型、1 个环境 | 删除一个 action，重新执行其余 action，看结果变化 | 分数来自真实环境，但训练时增加环境执行成本 |

GiGPO 是主实验必须直接复现的 baseline。两者都把整条轨迹分数和逐 action 分数
结合起来，但局部分数的来源不同：GiGPO 使用自然出现的相同状态，RECAP 主动
做删除实验。论文叙事不能只写“我们也做 turn-level credit”，必须突出
“环境实际执行的删除结果”和“计算量目标”。

### 10.3 使用重放或改变训练探索的方法

| 方法 | 核心规模 | 完整规模 | 怎样使用额外执行或历史数据 | 与 RECAP 的区别 |
|---|---|---|---|---|
| Spark，ACL 2026 | 2 个模型；ALFWorld、ScienceWorld、WebShop | 3 种模型、5 个 benchmark | 在模型不确定的状态生成多个新分支 | 解决训练探索不足，不给原轨迹 action 做删除分数 |
| SPEAR，ICLR 2026 | 2 个模型；ALFWorld、WebShop | 4 种模型、4 个 benchmark | 保存过去的成功轨迹，之后继续模仿 | 使用历史 replay buffer；RECAP 重放当前轨迹的环境动作 |
| Erasable RL，ICLR 2026 | 推理任务 | — | 删除错误推理步骤后让模型重新生成 | 会生成新的后续内容；RECAP 固定使用原来的后续 action |
| C3，2026 预印本 | 多智能体数学和代码任务 | 5 个 benchmark | 固定后续消息，比较某条消息存在与否 | 最接近“固定后续内容”的做法，但不是单 agent 环境效率训练 |
| Causal Agent Replay，2026 预印本 | agent 失败归因 | — | 改动步骤后重新运行策略，寻找失败原因 | 用于解释失败，不把 PTE 分数接入 GRPO 训练 |

Spark 和 SPEAR 与 RECAP 都会增加训练阶段的处理，但解决的问题不同：前两者
主要帮助模型找到更多成功路线，RECAP 主要判断一条已经成功的路线里哪些
action 可以省掉。C3 和 Causal Agent Replay 是需要主动说明的近期工作；它们
不阻止本项目继续，但说明“使用 counterfactual replay”本身已经不能单独作为
全部创新。

因此 RECAP 的完整创新点应当是三部分共同成立：

1. 真实环境中的 action deletion；
2. 任务成功优先、PTE 节省其次的逐 action 分数；
3. 固定少量 replay 时，考虑早期 action 对后续输入成本的候选选择。

第三点目前只有离线支持，还没有通过独立训练消融证明，不能提前写成定论。

## 11. ICLR 2027 潜力与需要达到的证据

当前方法已经具备继续按 ICLR 2027 完整论文推进的创新潜力，但当前实验包还不
够投稿。主要差距不在于与 GiGPO、Spark、RLVMR、SPEAR 或 DEPO 撞车，而在于
只有 1 个模型、1 个环境和 1 个训练 seed，并且还没有证明 PTE 下降会带来真实
推理时间下降。

与现有论文的核心 agent 实验相比：

| 方法 | 目标模型 | agent benchmark |
|---|---:|---:|
| GiGPO | 2 | 2 |
| Spark | 2 | 3 |
| RLVMR | 2 | 2 |
| SPEAR | 2 | 2 |
| DEPO | 2 | 2 |
| RECAP 当前 | **1** | **1** |

RECAP 的最低目标应为“2 × 2 × 3”：

```text
至少 2 个模型规模
× 至少 2 个 agent benchmark
× 每个核心配置至少 3 个训练 seeds
```

推荐使用 Qwen2.5-1.5B 和 Qwen2.5-3B/7B，以及 ALFWorld 和
WebShop/ScienceWorld。达到 2 个模型、2 个环境后，实验广度已经能与 GiGPO、
RLVMR、SPEAR 和 DEPO 的核心 agent 部分对齐；第三个环境是加分项，不是第一
阶段的硬门槛。

论文还必须同时满足：

1. 与 GRPO、GiGPO 和“不逐 action 删除”的内部版本使用相同训练设置；
2. 成功率只小幅下降，同时共同成功任务的 PTE 稳定降低；
3. 在未用于选择 checkpoint 的测试集上报告最终结果；
4. 报告真实模型推理时间，而不只报告 PTE、turns 和 token；
5. 报告训练额外花费的环境时间，并说明需要多少次部署推理才能抵消训练开销；
6. 证明收益不是来自提前失败、格式错误或只在一个 ALFWorld 任务类型上有效。

最重要的主图仍然是成功率与计算量的关系：在成功率接近时，RECAP 应稳定使用
更少 PTE 和更短真实模型时间。

## 12. 方法进度与下一步实验

### 已完成

- 已实现成功轨迹的环境重放；
- 每条成功轨迹最多测试 2 个 action；
- 删除后仍成功就按实际节省的 PTE 扣分；
- 删除后失败就给必要 action 加分；
- 分数只写回被测试 action 的模型输出；
- 已从 Qwen2.5-1.5B-Instruct 完整训练 222 steps；
- 已完成与 GiGPO、V1 和“不逐 action 删除”版本的对比。

最佳 step 200 成功 137/140。相对相同步数 GiGPO，共同成功任务 PTE 低
14.3%；相对“不逐 action 删除”的内部版本低 5.7%。重复 action 低 47.7%。
222 steps 中执行了 37,058 次删除实验，说明逐 action 分数确实参与了训练。

### 候选 action 选择的当前证据

对已有 32 条轨迹的全部 186 个 action 做离线删除后，在“每条轨迹只能选择
两个 action”的相同预算下：

| 选择方法 | 找到的可删除 action | 覆盖的可节省 PTE |
|---|---:|---:|
| 只选当前轮 PTE 最高的两个 | 2/56 | 4.0% |
| 选后续影响分最高的两个 | 15/56 | 30.0% |
| 当前 V2：一个后续影响＋一个当前轮最贵 | 10/56 | 20.1% |
| 知道全部删除结果后的理论上限 | 31/56 | 54.3% |

这证明“只看当前轮成本会漏掉早期 action”，也为考虑后续影响提供了离线支持。
但它还不能证明当前公式最优，也不能证明离线命中更多可删除 action 就一定能
训练出更好的模型。尤其是 `Suffix-only` 离线高于当前 Mixed，应当优先训练
验证。

### 接下来按顺序执行

1. 冻结当前 step-200 checkpoint，在 `valid_unseen` 上只评估一次；
2. 在不改变其他设置的情况下训练 `Local-PTE`、`Mixed`、`Suffix-only` 和
   `Random` 候选消融，保持 `replay_budget=2`；
3. 为最终选出的候选方法补两个 ALFWorld seeds；
4. 在第二个支持稳定 reset/replay 的 agent benchmark 上实现相同方法；
5. 1.5B 跨环境成立后，再训练 Qwen2.5-3B/7B；
6. 分别关闭“必要 action 加分”和“按实际 PTE 节省量扣分”，确认两部分作用；
7. 固定并发度，分别记录模型读取输入、生成 action、环境执行和整条任务时间。

每个实验继续使用单独脚本，并以验证目标命名，防止覆盖日志。例如：

```text
qwen2_5_alfworld_recap_candidate_local.sh
qwen2_5_alfworld_recap_candidate_suffix.sh
qwen2_5_alfworld_recap_candidate_random.sh
```

## 13. 继续、暂停和重新设计的判断标准

满足下面条件就继续扩大实验：

- 至少 2/3 个 seeds 的共同成功任务 PTE 优于内部版本，整体平均方向一致；
- 成功率下降保持在可接受的小范围，而不是依靠大量快速失败降低 PTE；
- 第二个环境也出现 PTE 或真实推理时间改善；
- PTE 与拆开的模型推理时间具有稳定相关性；
- 训练增加的环境计算可以由多次部署推理的节省抵消。

出现下面情况时，不应直接扩大模型，而应先重新设计：

- 换 seed 后 5.7% 的内部增益消失或方向反复；
- 只有 ALFWorld 某一类任务有效；
- PTE 下降但真实模型时间不下降；
- 模型主要通过提前结束、格式错误或放弃困难任务变短；
- 固定后续 action 得到的删除分数与允许模型重新规划后的结果经常相反；
- replay 训练成本很高，但模型部署次数不足以收回这部分成本。

候选优先级公式本身不是必须保留的核心。如果 `Suffix-only`、随机选择或更准确
的后续成本计算更好，应替换当前 Mixed，而不是为了维持原设计忽略实验结果。

## 14. 当前最准确的项目表述

推荐表述：

> RECAP-GRPO 在训练时删除一条成功轨迹中的某个 action，并在真实环境中重新
> 执行其余 action。任务失败说明原 action 对当前路线重要；任务仍成功则按照
> 实际节省的 PTE 对该 action 扣分。这个分数只写回对应 action，部署时仍使用
> 普通的逐轮推理流程。当前 V2 在 Qwen2.5-1.5B 和 ALFWorld 的一个 seed 上，
> 以小幅成功率差异换得了相对 GiGPO 和内部版本更低的 PTE。

目前可以声称：

- 完整的逐 action 删除、环境重放和训练链路已经跑通；
- 当前一个 seed 的结果支持该方法具有继续扩展的价值；
- 相比 GiGPO、Spark、RLVMR、SPEAR 和 DEPO，训练信号的来源和优化目标不同。

目前不能声称：

- 已经稳定优于所有内部消融；
- 已经在多个模型和多个环境上成立；
- 已经实现真实端到端推理加速；
- 当前候选优先级公式已经被证明最优；
- 单独使用 counterfactual replay 就构成全部创新。

论文中的方法贡献应始终作为一个整体描述：真实环境 action deletion、成功优先
的 PTE 分数，以及固定 replay 预算下考虑后续成本的候选选择。PTE 来源于已有
工作，不能把 PTE 指标本身写成我们的贡献。

## 15. 主要参考论文

### 效率与计算量

- [Beyond Accuracy / PTE, ACL 2026](https://aclanthology.org/2026.acl-long.339/)
- [DEPO](https://arxiv.org/abs/2511.15392)
- [RLVMR, ICLR 2026](https://openreview.net/forum?id=cTbAevdwBE)

### 逐 action 或逐步骤分数

- [GiGPO, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html)
- [POAD, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc09efb501c801ed92e181e26a885c2d-Abstract-Conference.html)
- [Segment Policy Optimization, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a6536243037d1e32c20de85137d478da-Abstract-Conference.html)
- [HGPO, ICLR 2026](https://openreview.net/forum?id=T8Dev99qnz)
- [SSVPO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4da4dd613cc9e63932d643b9c6e8f86f-Abstract-Conference.html)

### 重放与训练探索

- [Spark, ACL 2026](https://arxiv.org/abs/2601.20209)
- [SPEAR, ICLR 2026](https://arxiv.org/abs/2509.22601)
- [Erasable RL, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9ffc16880822e689abcf6801e11f7f53-Abstract-Conference.html)
- [C3](https://arxiv.org/abs/2603.06859)
- [Causal Agent Replay](https://arxiv.org/abs/2606.08275)
