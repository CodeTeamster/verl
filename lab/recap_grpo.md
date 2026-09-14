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

后续又完成了两组“按轨迹长度决定测试数量”的 step-200 实验。把比例从
`40%` 降到 `20%` 后，结果明显恢复：成功数从 131 提升到 135，平均 PTE 从
10,764 降到 8,649。这证明 `40%` 的测试和打分强度确实过大。不过，`20%`
版本仍不如固定测试 2 个 action 的 V2：共同成功任务上的 PTE 高 9.1%。因此
当前主方法仍应保留固定 2 次 replay；比例版本作为重要消融，而不是替换 V2。

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

先给一条轨迹计算原有的、按组归一化的 PTE-GRPO advantage，记为
\(A^{\mathrm{macro}}_\tau\)。它仍然以任务成功为第一目标；只在成功轨迹之间偏向
PTE 更低的路线，避免模型通过快速失败获得高分。

下面是当前效果最好的 **RECAP-GRPO V2** 实际使用的 turn-level 公式。对一条原本
成功的轨迹 \(\tau\)，删除第 \(t\) 个 action 并重放原后缀，得到
\(\tau_{-t}\)。令 \(R(\cdot)\) 为任务奖励，\(C(\cdot)\) 为整条轨迹的 PTE：

\[
\Delta R_t=R(\tau)-R(\tau_{-t}),\qquad
\Delta C_t=C(\tau)-C(\tau_{-t}).
\]

原始局部 credit 是：

\[
c_t^{\mathrm{raw}}=
\begin{cases}
\Delta R_t, & |\Delta R_t|>\epsilon,\\[4pt]
-\dfrac{\Delta C_t}{C(\tau)+\epsilon}, & |\Delta R_t|\le\epsilon.
\end{cases}
\]

因此，在当前只测试原本成功轨迹的实现中：

- 删除后失败时，\(\Delta R_t=1-0=+1\)：该 action 对这条固定后缀路线必要；
- 删除后仍成功时，credit 为 \(-\Delta C_t/C(\tau)\)：删掉后节省的 PTE 越多，
  惩罚越强；这个节省同时包含被删 turn 本身和后续 turn 因上下文变短而减少的
  prefill；
- 未被选中测试的 turn 不得到局部 credit。

V2 对同一条轨迹内的正、负值**分别**做 `signed_absmax` 归一化：

\[
\hat c_t=
\begin{cases}
c_t^{\mathrm{raw}} / \max_{j:c_j^{\mathrm{raw}}>0}c_j^{\mathrm{raw}}, & c_t^{\mathrm{raw}}>0,\\[4pt]
c_t^{\mathrm{raw}} / \max_{j:c_j^{\mathrm{raw}}<0}|c_j^{\mathrm{raw}}|, & c_t^{\mathrm{raw}}<0,\\[4pt]
0, & \text{otherwise}.
\end{cases}
\]

最后只在该 turn 的 assistant token 区间写入局部项：

\[
A_{t,\mathrm{token}}=A^{\mathrm{macro}}_\tau+\beta\hat c_t,
\qquad \beta=0.10.
\]

例如，关键 action 删除后失败的原始和归一化 credit 都是 \(+1\)，所以该 turn
每个模型 token 在原 macro advantage 基础上额外得到 \(+0.10\)。若删掉一个
冗余 action 使 PTE 从 10,000 降到 8,000，则其原始 credit 为 \(-0.2\)；若同一
轨迹最大的负 credit 为 \(-0.25\)，该 turn 的局部增量为
\(0.10\times(-0.2/0.25)=-0.08\)。

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

因此主结果中的 V2 在选择待测试 turns 时，同时考虑：

```text
优先级 = 当前 turn 的成本 + 它可能给后续 turns 增加的成本
```

为了控制训练开销，主版本每条成功轨迹最多只测试 2 个 action：

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

代码现在也支持按轨迹长度分配预算：

```text
测试数量 K = 轨迹 action 数量 × 比例，再限制到最大值
```

已经测试过两种设置：`40% / 最大 6 个` 和 `20% / 最大 4 个`。这两个版本还把
75% 的名额分给“后续影响分”，剩余名额分给“当前轮成本”。例如 `K=4` 时，
选择 3 个后续影响大的 action 和 1 个当前轮最贵的 action。`K=2` 时因取整会
选择 2 个后续影响大的 action，而不是主 V2 的“一后续影响＋一当前轮成本”。

所以这两组实验同时改了两件事：测试数量由固定值变为比例值，候选也更偏向
轨迹前段。它们是“比例预算＋偏后续影响选择”的联合实验，不能把结果变化全
归因于比例。后面需要在相同总 replay 数量下单独改变分配方式，才能干净比较。

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

### PTE-GRPO 与 RECAP-GRPO V1：早期对照

PTE-GRPO 和 V1 都已完成 from-scratch step-200 验证。前者只给整条轨迹一个
效率分数；V1 首次接入 deletion replay，但只对“删除后仍成功”的 action 写入
固定 \(-1\) 局部 credit，删除后失败的必要 action 不额外加分。它们的结果统一
放在下面的方法级对比表中。V1 相比不使用 turn-level credit 的 PTE-GRPO 没有
降低共同成功任务的 PTE，因此它只是完整链路的早期可行性验证，不是当前方法。

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

最佳 V2 的独立实验入口为：
`bash lab/qwen2_5_alfworld_recap_grpo.sh`。默认实验标识为：

```text
project: alfworld_recap_cfutility_from_scratch
experiment: qwen2.5_1.5b_replay2_signed_delta_beta0.10
```

最后的 step 222 退化为 135/140、mean PTE 8,147，因此下面均使用最佳的
step 200。当前仍缺 `valid_unseen`、多 seeds、第二环境和真实模型推理时间，
不能把单 seed 结果当作最终论文结论。

### 方法级对比：统一 step-200

下面全部使用同一批 140 个 ALFWorld validation 任务、同一个 PTE 计算脚本和
step 200，避免把不同 checkpoint 混在一起。数值越低表示推理越省。

| 训练方法 | 成功数 / 140 | 全任务平均 PTE | 成功任务平均 PTE | 平均 turns | 平均生成 tokens |
|---|---:|---:|---:|---:|---:|
| 标准 GRPO | **138** | 9,206.7 | 8,342.7 | 8.779 | 216.6 |
| GiGPO | **139** | 7,726.1 | 7,423.0 | 8.521 | 109.1 |
| PTE-GRPO：轨迹级效率分数 | 136 | 8,258.8 | 6,958.2 | 8.757 | 111.7 |
| RECAP-GRPO V1：固定惩罚可删除 action | 136 | 8,142.6 | 6,842.1 | 8.664 | 110.6 |
| **RECAP-GRPO V2：固定测试 2 个** | **137** | **7,232.5** | **6,280.9** | **8.007** | **102.7** |

这张表给出两点直接结论：

1. 固定测试 2 个 action 的 V2 仍是当前最好的 RECAP 配置。它比标准 GRPO 少
   1 个成功任务，但全任务平均 PTE 低 21.4%，平均 turns 低 8.8%；在两者都
   成功的 136 个任务上，PTE 低 25.5%。
2. 比例预算、候选规则和归一化规则的影响放在下一张 V2 组内消融表中；它们都没有
   超过固定测试 2 个 action 的 V2。

### V2 组内消融：预算、候选与局部 credit

所有行均为 Qwen2.5-1.5B、ALFWorld、from-scratch、step 200。`Mixed` 表示一个
候选按后续影响选取、一个候选按当前 local PTE 选取；`Suffix-only` 表示两个都按
后续影响选取。AnchorCap 表示无论预算多大，都先保留 Mixed 的两个候选，再交替
补充候选。除表中明确标出的行外，局部 credit 都使用第 5 节的连续公式与
`signed_absmax`。

| V2 设置 | replay 预算与候选 | 局部 credit 归一化 | 成功数 / 140 | 全任务平均 PTE | 成功任务平均 PTE | 平均 turns | 平均生成 tokens |
|---|---|---|---:|---:|---:|---:|---:|
| **主 V2** | 固定 2，Mixed | signed-absmax | **137** | **7,232.5** | **6,280.9** | **8.007** | **102.7** |
| 比例预算 | 40%，最多 6；75% suffix | signed-absmax | 131 | 10,764.1 | 7,542.4 | 10.107 | 129.0 |
| 比例预算 | 20%，最多 4；75% suffix | signed-absmax | 135 | 8,648.8 | 6,837.7 | 8.886 | 112.4 |
| 候选消融 | 固定 2，Suffix-only | signed-absmax | 135 | 8,712.0 | 6,869.8 | 8.807 | 111.8 |
| credit 消融 | 固定 2，Mixed | signed-sum | 135 | 8,347.7 | 6,712.7 | 8.679 | 110.2 |
| AnchorCap 消融 | 25%，最少 2、最多 4；Mixed anchors 后交替补充 | signed-absmax，同号总量最多 2 | **139** | 7,641.7 | 7,309.0 | 8.450 | 107.8 |

在固定 2 个与 `20% / max4` 都成功的 133 个任务上，比例版本的 PTE **高
9.1%**。它平均多走 0.88 turns，多生成约 9.7 个 tokens；分叉后的后半段也有
更多重复 action 和失败环境反馈。因此，`20%` 版本相对 `40%` 版本是成功的
修复实验，但相对主 V2 仍是负结果。

训练期间的删除实验却呈现了另一面。前 200 个 steps 中，固定 2 个、40% / max6
和 20% / max4 的删除命中率分别为 14.6%、22.8% 和 35.1%；每次测试发现的平均
可节省 PTE 分别为 238.8、418.4 和 610.5。`20% / max4` 每条可重放轨迹平均只
测试 1.67 次，比固定 2 次少 16.5%，但命中率是固定版本的 2.40 倍。这说明按长度、
偏后续影响的选择确实更容易找到“看起来可以删”的 action，训练信号利用率符合预期。

但最终模型没有因此更快。这是一个重要发现：**更容易找到可删除 action，不
代表这些 action 的训练分数一定能教出更短的策略。** 比例版本可能过多集中于
很容易识别、但对学习新路线帮助有限的冗余动作，也可能减少了对“当前轮昂贵
但必要”的 action 的覆盖。正负 credit 的数量和大小也随测试数变化，进一步
改变了训练强度。

训练时间采用 step 151--199 中排除 validation 和保存 checkpoint 的普通 steps。
固定 2 个、40% / max6 和 20% / max4 的“生成轨迹和环境重放”时间分别为
236.1、339.9 和 266.2 秒 / step；完整 step 时间分别为 301.8、408.7 和 414.1 秒。
`20% / max4` 的生成与重放阶段比 `40% / max6` 快 21.7%，只比固定 2 个慢 12.7%。
完整 step 时间没有同步下降，因为该次运行的 `old_log_prob`、参考模型和 actor
更新也更慢；这些阶段不执行 action deletion，不能把这部分波动归因于比例预算。

本次实验最终判断为：

- “40% 比例过大”得到支持；
- “降低比例可以恢复成功率和效率”得到支持；
- “按长度分配能提高删除测试命中率”得到支持；
- “比例预算能超过固定 2 次 V2 的最终推理效率”没有得到支持；
- 因为比例和候选分配同时变化，目前还不能判断退化究竟来自哪一个因素。

对应结果文件为：

```text
固定 2 个：outputs/validation_log/alfworld_recap_cfutility_from_scratch/
  qwen2.5_1.5b_replay2_signed_delta_beta0.10/2026-09-07_18-23-30/200.jsonl
40% / max6：outputs/validation_log/alfworld_recap_proportional_budget_from_scratch/
  qwen2.5_1.5b_ratio0.40_max6_suffix0.75/2026-09-09_20-09-50/200.jsonl
20% / max4：outputs/validation_log/alfworld_recap_ratio020_max4_from_scratch/
  qwen2.5_1.5b_ratio0.20_max4_nearest_suffix0.75/2026-09-10_17-44-05/200.jsonl
```

### 为什么固定 2 次 Mixed 比当前比例版本好

当前结果最合理的解释可以简化成三点：

1. **Mixed 同时寻找“可以删”和“必须留”的 action。** 一个名额看 action 对
   后续输入的影响，容易找到多余观察和绕路；另一个名额看当前轮成本，容易
   覆盖接近任务完成位置的关键 action。前者主要提供负 credit，后者帮助提供
   必要 action 的正 credit。
2. **当前比例版本实际上太偏向第一类候选。** `suffix_fraction=0.75` 使用向上
   取整，所以当 `K=1、2、3` 时，所有名额都在看后续影响。`20% / max4` 平均
   每条轨迹只测试 1.67 次，因此绝大多数轨迹近似 `Suffix-only`，没有 Mixed
   中的“当前轮成本”候选。
3. **测试越多，局部 credit 的总影响也越大。** 当前代码只缩放每个 credit
   的大小，不会除以测试数量。长轨迹得到更多 replay，也就有更多 action token
   收到局部分数。长轨迹通常依赖关系更复杂，而删除实验又固定执行原来的后续
   action，所以这些分数更容易只对当前路线成立。测试过多可能反过来强化绕路。

validation 与这个解释一致：三个版本每个 turn 都只生成约 12--13 个 tokens，
差距不是每轮 `<think>` 更长，而是比例版本走了更多 turns，并且更常出现重复
action、失败环境反馈和 35-turn 长尾。`20%` 版本前期学习更快、后期被 Mixed
反超，也说明只清理明显冗余适合训练前期，但不足以持续改善完整规划。

固定 `K=2` 的 Mixed/Suffix-only 对照也见上面的 V2 组内消融表。它确认了上述
关键判断：只偏向后续影响虽然更容易找到可删除 action，但没有教出更短的成功路线；
保留两种候选来源更合理。

### 已完成：固定 K=2 的候选与归一化消融

这里包含两个单变量实验。Suffix-only 只把候选从 Mixed（一个后续影响最大的
action，加一个当前轮最贵的 action）改成两个都按后续影响选择；Signed-sum
保持 Mixed，只把同符号 credit 从“按最大值缩放”改成“按总和缩放”。其余训练
设置都保持为 V2：每条成功轨迹最多重放 2 次、连续 PTE credit 和 `beta=0.1`。

Suffix-only 在 step 200 的 PTE 高 20.5%，而且多走约 10% 的 turns，因此它没有
在相同训练预算下超过 Mixed。Suffix-only 收敛得更晚，最佳点出现在 step 220：

它在 step 220 达到 139/140、全任务 PTE 7,062，但在双方共同成功的 137 个任务上，
PTE 反而比 Mixed 高 6.3%。全任务均值较低主要来自少了两个很长的失败轨迹，不能
解释为成功路线真正加速。

最直观的现象是：Suffix-only 的可删除 action 命中率为 27.3%，Mixed 为
14.6%，但更高的命中率没有变成更高效的成功策略。Suffix-only 更常找到容易
删除的绕路；Mixed 还会测试当前很贵但可能必要的 action，得到“哪些动作必须
保留”的正向证据。这个实验支持继续保留 Mixed，也说明候选规则不能只追求
删除命中率。

Signed-sum 也没有达到预先设定的门槛。它在 step 200 比 Suffix-only 略好：成功
数相同，全任务 PTE 低 4.2%；但相对原 Mixed 少成功 2 个，全任务 PTE 高 15.4%，
双方共同成功任务 PTE 高 7.6%。它在 step 190、200、210、220、222 的成功数
分别为 132、135、133、132、133，只有 step 200 刚好达到 135/140 的最低门槛，
后期也没有形成持续改善。

直观上，`signed_sum` 确实去掉了“两个同号结果就得到两份完整 credit”的数量
影响，但同时把最常见的必要 action 正 credit 变弱了。删除测试中大部分 action
都是必要 action；把两个 `+1` 改成两个 `+0.5` 后，模型更少收到保持路线完整的
局部提醒。step 200 的成功轨迹中，重复 action 和失败环境反馈都没有优于原
Mixed，说明减少正 credit 没有换来更干净的路线。结论不是“固定总量一定错误”，
而是当前 `beta=0.1` 下的 Signed-sum 没有优于 V2；不能据此继续增加比例预算。

### 已完成但未采用：AnchorCap 动态预算

AnchorCap 是在前面两个负结果后做的折中验证。它不再让比例预算退化成
`Suffix-only`，也不把同号 credit 总量强行压到 1。对一条有 `T` 个 actions 的
轨迹，测试数为：

```text
K(T) = min(T, 4, max(2, round(0.25 * T)))
```

前两个名额始终是一个“后续影响”候选和一个“当前轮成本”候选；如果还有名额，
再从两个排序中交替补充。credit 先按 `signed_absmax` 计算，但每条轨迹的正 credit
之和最多为 2，负 credit 的绝对值之和也最多为 2。这样既保留 Mixed 的两种证据，
又允许长轨迹多测少量 action，同时限制额外测试无限放大更新。

step 200 达到 139/140，说明成功率没有被动态预算破坏；但全任务平均 PTE 为
7,641.7，比主 V2 高 5.7%。在双方都成功的 136 个相同任务上，AnchorCap 的 PTE
高 15.9%，因此不能用“多成功了两个任务”解释为成功路线真正更快。它平均多走
0.443 turns、多生成 5.1 tokens；相同成功任务中，平均重复 action 从 0.581 增至
0.721，失败环境反馈从 0.691 增至 0.941。

验证曲线也没有显示后期反超：step 190、200、210、220、222 分别为 135、139、
137、135、136 个成功；对应全任务平均 PTE 为 8,509.8、7,641.7、7,863.2、
8,267.1、8,124.2。step 200 是当前最好的准确率与效率折中点，但没有任何一个
后期 checkpoint 同时超过主 V2 的成功率和 PTE。

训练中的删除证据本身变多了。前 200 steps，AnchorCap 每条可重放轨迹平均测试
2.384 个 action，可删除命中率为 19.5%，每次测试平均找到 326.5 PTE 的节省；
主 V2 对应为 2.000、14.6% 和 238.8。可是这些额外证据没有转化成更短的最终
策略。这进一步支持当前最重要的机制判断：**训练信号的学习价值比可删除 action
的命中数量更重要。** 多测到一些容易删除的绕路，并不等于更清楚地教会模型应该
走哪条短路线。

使用相同并行 slot 管理时，step 151--199 中排除 validation 和 checkpoint 的
普通 steps 平均为 105.3 秒，其中生成与环境重放为 39.4 秒。最接近的并行固定
`K=2` 对照分别约为 94.2 秒和 28.3 秒；因此 AnchorCap 的完整 step 慢 11.8%，
生成与重放阶段慢 39.4%。这不是不可接受的系统开销，但在最终 PTE 没有改善时，
没有理由用它替换更简单的固定预算。

结果文件为：

```text
outputs/validation_log/alfworld_recap_anchorcap_from_scratch/
  qwen2.5_1.5b_ratio025_min2_max4_anchor_cap2_beta0.10/2026-09-13_20-32-26/
```

本次同时使用了新的并行 slot 管理。对 step 151--199 中排除 validation 和
checkpoint 的普通 steps，生成轨迹和环境重放从 236.1 秒 / step 降到 27.7 秒 / step
（降低 88.3%，约 8.5 倍）；完整训练 step 从 301.8 秒降到 93.8 秒（降低 68.9%，
约 3.2 倍）。222-step 训练进度时间从 19 小时 21 分降到 13 小时 02 分。Signed-sum
使用同一套并行管理，step 151--199 的对应时间为 28.3 秒和 94.2 秒，
与 Suffix-only 的 27.7 秒和 93.8 秒几乎一致。这次复现进一步说明并行加速稳定，
而不是 Suffix-only 偶然产生的计时结果。Signed-sum 的完整训练进度时间为
8 小时 47 分；它比 Suffix-only 的 13 小时 02 分更短，但普通 step 耗时几乎
相同，差值主要来自早期机器负载和 checkpoint 等阶段，不能归因于归一化方法。

每个 slot 都是独立且长期复用的 actor；不同 slot 可以并行，同一条轨迹及其
删除重放仍独占同一个 slot。代价是当前 8 个 agent worker 各有 16 个 slot，
约 128 个常驻环境 actor，CPU 内存最高约 200 GB。上表不是严格的管理方式
A/B，因为两个训练分别使用 Mixed 和 Suffix-only；不过二者均为固定 2 次
replay，后期每 step 实际测试 232.8 和 228.6 个 action，负载十分接近。论文
报告最终加速比前，仍应以相同 Mixed 配置复跑一次。

结果文件为：

```text
outputs/validation_log/alfworld_recap_fixed2_suffix_only_from_scratch/
  qwen2.5_1.5b_replay2_suffix_only_beta0.10/2026-09-11_23-00-42/
outputs/validation_log/alfworld_recap_fixed2_mixed_signed_sum_from_scratch/
  qwen2.5_1.5b_replay2_mixed_signed_sum_beta0.10/2026-09-12_14-02-22/
```

### 设计记录：预算无关的平衡 turn credit

原计划暂称 **RECAP-V3：预算无关的平衡 credit**。它不增加模型、价值网络或
测试时步骤，只改两个简单规则。Signed-sum 和随后更宽松的 AnchorCap 都没有
超过 V2，因此下面保留为设计记录，不再直接扩大动态比例预算训练。

#### 规则一：始终保留两个不同作用的候选

只要预算 `K >= 2`，就先选择：

```text
1 个对后续输入影响最大的 action
+
1 个当前轮成本最大的 action
```

如果比例预算还有剩余名额，再从两个排序中交替补充并去重。这样短轨迹不会因
取整退化成 `Suffix-only`，长轨迹也可以得到更多证据，但不会丢掉 Mixed 的
基本平衡。

#### 规则二：固定每条轨迹的 credit 总量

删除实验仍产生原来的原始分数：

```text
删除后失败：正 credit
删除后仍成功：- 节省的 PTE / 原轨迹 PTE
```

但不再用“同一符号中的最大值”归一化，而是用“同一符号的总和”：

```text
最终正 credit_t = 原始正 credit_t / 所有正 credit 之和
最终负 credit_t = 原始负 credit_t / 所有负 credit 绝对值之和
```

因此无论测试 2 个还是 6 个 action：

```text
所有正 credit 之和最多为 +1
所有负 credit 之和最多为 -1
```

例如一条长轨迹测试出 3 个必要 action 和 3 个可删除 action，它们只是共同
分配固定的正、负 credit，而不是得到 6 份完整强度的局部分数。增加 replay
只会让 credit 分配得更准确，不会让这条长轨迹在训练中自动变得更重要。

最终仍使用：

```text
这个 turn 中每个 token 的局部分数
    = 归一化后的 turn credit
      × batch 内被测试 turn 的平均 token 数
      / 这个 turn 的模型 token 数

token 的训练分数
    = 整条轨迹分数 + beta × token 的局部分数
```

也就是说，一个 turn 的 credit 会平均分给它的 `<think>` 和 `<action>` tokens，
而不是给每个 token 都复制一份完整 credit。乘上 batch 平均长度只是为了让
`beta` 的常用大小与当前实现接近，不引入新的可调参数。否则同一个 action 仅仅
因为文字更长就会获得更大的更新量。这样不仅 action 数量变化时总量不变，turn
的文字长度变化时总量也不变。

这个设计比按 `K` 手工缩放 `beta` 更通用，因为它不依赖固定预算、平均轨迹
长度或具体环境。PTE 也可以替换成真实时间、token、费用或能耗：只要能够删除
action 后重新执行，并测量任务结果和成本，就能使用相同的 turn credit。

它的核心可以概括为：

```text
Mixed 保证证据来源平衡
+
credit 总量固定，保证训练强度不随 replay 数和 turn 长度变化
```

相比继续调 `20%`、`30%` 或 `40%`，这一改动更像一个完整的 credit assignment
原则：**replay 数量决定我们观察多少证据，但不应决定一条轨迹获得多大的训练
权重。** 这个原则简单、可解释，也比某个只适用于 ALFWorld 的固定比例更容易
推广到其他 multi-turn agent 环境。

最小验证顺序为：

1. 固定 `K=2` 的 Mixed/Suffix-only 已完成：Suffix-only 删除命中率更高，
   但共同成功任务 PTE 更差，因此保留 Mixed；
2. 固定 Mixed 的 Signed-sum 已完成：成功率刚到最低门槛，但共同成功任务 PTE
   高 7.6%，所以不替换 `signed_absmax`；
3. 因总量归一化没有效果，不打开比例预算和额外候选；
4. 把 turn credit 平均分给该 turn 的 tokens 暂时后移为消融，因为现有版本每轮
   生成 tokens 接近，当前效率差异主要来自 turns 数，而不是单轮文字长度；
5. 前三项成立后，再把 RECAP-V3 扩展到额外 seeds、模型和 benchmark。

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
现在虽然已有两组比例预算训练，但它们同时改变了测试数量和候选构成，仍不能
作为候选公式的独立证明。

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
- 主版本每条成功轨迹最多测试 2 个 action；
- 删除后仍成功就按实际节省的 PTE 扣分；
- 删除后失败就给必要 action 加分；
- 分数只写回被测试 action 的模型输出；
- 已从 Qwen2.5-1.5B-Instruct 完整训练 222 steps；
- 已完成与 GRPO、GiGPO、V1 和“不逐 action 删除”版本的统一 step-200 对比；
- 已完成 `40% / max6` 与 `20% / max4` 两组比例预算训练；
- 已完成固定 `K=2` 的 Mixed/Suffix-only 候选消融；
- 已完成固定 `K=2` Mixed 的 Signed-sum 归一化消融；
- 已完成 `25% / min2 / max4` AnchorCap 动态预算消融；
- 已验证并行 slot 管理可把后期普通 step 的环境相关阶段缩短约 88%。

当前主 V2 的最佳 step 200 成功 137/140。相对相同步数 GiGPO，共同成功任务 PTE 低
14.3%；相对“不逐 action 删除”的内部版本低 5.7%。重复 action 低 47.7%。
222 steps 中执行了 37,058 次删除实验，说明逐 action 分数确实参与了训练。

比例预算没有替代主版本。`20% / max4` 虽然比 `40% / max6` 明显更好，而且
删除测试命中率最高，但 step-200 的最终 PTE 仍高于固定 2 次版本。因此当前
默认配置和论文主方法仍是固定 2 次 replay；两个比例版本作为预算消融保留。
Signed-sum 同样没有替代主版本：它排除了 credit 数量变化，但削弱局部信号后
没有得到更短的成功路线。因此当前默认归一化仍为 `signed_absmax`。
AnchorCap 把 Mixed anchors、动态预算和较宽松的同号 credit 上限组合起来，
step 200 成功率提高到 139/140，但共同成功任务 PTE 比主 V2 高 15.9%，普通
训练 step 也慢 11.8%。因此它证明“保住成功率”是可以做到的，却没有证明多出的
counterfactual 证据值得使用；当前不再继续扩大动态 replay 预算。

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
在线 Suffix-only 消融进一步说明：离线命中更多可删除 action，并不等于训练后
策略更高效。当前证据支持保留一个后续影响候选和一个当前成本候选，但还不能
证明两个优先级公式本身已经最优。

### 接下来按顺序执行

1. 保留固定 2 次 Mixed + `signed_absmax` 的 V2 作为当前主方法，不继续扩大动态预算；
2. 用并行 slot 原样复跑 V2，得到严格的环境管理 A/B，并检查主结果能否复现；
3. 冻结复现后的主方法，在 `valid_unseen` 上只评估一次；
4. 为 V2 补两个 ALFWorld seeds，先确认当前增益不是单 seed 偶然；
5. 在第二个支持稳定 reset/replay 的 agent benchmark 上实现相同方法；
6. 1.5B 跨环境成立后，再训练 Qwen2.5-3B/7B；
7. 再补 `Local-PTE` 和 `Random` 候选对照，所有版本保持相同 replay 总数；
8. 分别关闭“必要 action 加分”和“按实际 PTE 节省量扣分”，确认两部分作用；
9. 把每个 turn 的 credit 平均分给其模型 tokens，作为长度不变性的次要消融；
10. 固定并发度记录模型读取输入、生成 action、环境执行和整条任务时间。

### 已完成但未采用：固定 K=2 Mixed + credit 总量归一化

本次训练没有改候选、replay 数和 `beta`，只替换归一化方式。原因很
直观：当前 `signed_absmax` 下，如果两个被测 action 都是必要的，它们会各得
`+1`，整条轨迹一共得到 `+2`；如果只测到一个必要 action，则总量只有 `+1`。
训练强度因此会随同号证据数量变化。新规则让两个必要 action 各分 `+0.5`，总量
始终为 `+1`；负 credit 也用相同方式处理。一正一负时仍分别为 `+1/-1`，不会
抹掉“必须留”和“可以删”的方向。

实验入口为：

```text
bash lab/qwen2_5_alfworld_recap_ablations.sh signed-sum
```

设置保持 validation 每 10 steps、checkpoint 每 50 steps，并已训练到 222。
预先设定的继续门槛是成功数不低于 135/140、共同成功任务 PTE 不高于 Mixed，
且最好下降至少 3%。实际最佳 step 200 为 135/140，但共同成功任务 PTE 高 7.6%，
因此按原判断规则停止增加比例预算，保留当前 V2。

所有消融现在使用同一个脚本，并通过第一个参数选择配置。脚本仍为每种配置设置
独立的 project 和 experiment 名称，所以不会覆盖历史日志：

```bash
bash lab/qwen2_5_alfworld_recap_ablations.sh --help
bash lab/qwen2_5_alfworld_recap_ablations.sh v1
bash lab/qwen2_5_alfworld_recap_ablations.sh ratio040-max6
bash lab/qwen2_5_alfworld_recap_ablations.sh ratio020-max4
bash lab/qwen2_5_alfworld_recap_ablations.sh suffix-only
bash lab/qwen2_5_alfworld_recap_ablations.sh signed-sum
bash lab/qwen2_5_alfworld_recap_ablations.sh anchorcap
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
> 以小幅成功率差异换得了相对 GRPO、GiGPO 和内部版本更低的 PTE。比例预算
> 与 AnchorCap 消融表明，更高的删除命中率或更多 counterfactual 证据不会自动
> 变成更高效的最终策略，因此主版本仍使用固定 2 次 replay。

目前可以声称：

- 完整的逐 action 删除、环境重放和训练链路已经跑通；
- 当前一个 seed 的结果支持该方法具有继续扩展的价值；
- 固定 2 次 replay 在所有已完成的预算设置中最好；
- `40%` 预算过强，`20%` 能恢复大部分性能但仍未超过固定预算；AnchorCap 能把
  成功率恢复到 139/140，但共同成功任务仍比固定预算低效；
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
