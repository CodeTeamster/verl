# ELICA-GRPO：用模型内部表示给多轮 Agent 分配训练信用

## 1. 方法名称与一句话说明

**ELICA** 的全称是：

> **Efficiency-aware Latent Intervention for Credit Assignment**

基于 GRPO 的实现称为 **ELICA-GRPO**。

它不改变 Agent 的环境交互、action 空间或推理流程。训练时，ELICA 从 actor 已有前向
计算中取出每个 turn 的 hidden states，用一个很小的预测头估计整条轨迹的
“成功率—计算量”综合效用；再在这个小预测头里逐个遮住 turn，判断每个 turn 对预测
结果的影响，并据此重新分配原有的轨迹优势。

当前实验的轨迹级奖励沿用 PTE-GRPO，计算量用 Prefill Token Equivalents（PTE）衡量。
prediction head 只在训练中使用，验证和部署时完全移除，因此不会增加推理开销。

## 2. ELICA 解决什么问题

普通 GRPO 给一条轨迹中的全部 action token 相同的优势。假设 Agent 最终成功，但中间
先后做了以下动作：

```text
turn 1: go to kitchen
turn 2: go to hallway       # 绕路
turn 3: go to kitchen       # 回到原处
turn 4: take apple
turn 5: put apple in fridge
```

普通 GRPO 会一起奖励五个 turns，绕路 action 也得到正向更新。轨迹失败时情况相反：前面
可能正确的探索 action 会和真正导致失败的 action 一起受罚。

ELICA 的目标不是凭空增加新的奖励，而是回答：

> 在整条轨迹已经获得一个可靠结果奖励后，哪些 turns 更应该承担这条奖励？

## 3. 第一层：成功率与 PTE 共同决定轨迹优势

对同一个 prompt 的第 `i` 条 rollout，先计算任务得分 `R_i` 和整条轨迹 PTE：

```text
C_i = sum_t [prefill_i,t + gamma * prefill_i,t * decode_i,t]
```

当前 `gamma=0.00704`。只在同一 prompt 的成功轨迹之间归一化成本：

```text
C_tilde_i = (C_i - min_success C) / (max_success C - min_success C + epsilon)
```

轨迹效用为：

```text
U_i = R_i * (1 - lambda * C_tilde_i)
```

当前 `lambda=0.10`。因此：

- 失败轨迹的效用为 `0`；
- 成功轨迹效用位于 `[0.9, 1.0]`；
- 任意成功轨迹仍然优于失败轨迹；
- 在都成功的前提下，PTE 更低的轨迹获得更高效用。

然后像 GRPO 一样，在同一个 rollout group 内中心化并按标准差归一化：

```text
A_i = (U_i - mean_group(U)) / (std_group(U) + epsilon)
```

`A_i` 是宏观优势。它同时与任务是否成功和轨迹 PTE 挂钩，而且成功率的优先级更高。

## 4. 第二层：从 hidden states 得到 turn 表示

设 actor 最后一层输出为 `H_i,k`。对第 `t` 个 turn 的所有输出 token 做平均：

```text
h_i,t = mean(stop_gradient(H_i,k)),  k 属于 turn t
```

这里有两个重要细节：

1. observation token 不参与池化和策略更新，只处理模型自己生成的 response token；
2. `stop_gradient` 切断 prediction head 到 actor hidden states 的梯度。head 学着读取已有
   表示，但不会迫使语言模型为了方便预测奖励而改变表示。

实现复用了 actor 计算 log-prob 的前向过程，没有为 hidden states 单独再跑一次大模型。
完整 token hidden states 在当前 micro-batch 内立即压缩成
`batch × max_turns × hidden_size`，不会跨 step 保存。

## 5. 第三层：训练期效用预测头

Qwen2.5-1.5B 的 hidden size 是 `1536`。当前预测头为：

```text
LayerNorm(1536)
-> Linear(1536, 128)
-> GELU
-> Linear(128, 128)
-> GELU
-> gate:  Linear(128, 1)
-> value: Linear(128, 1)
```

对每个 turn，head 分别产生权重和值：

```text
z_i,t = Encoder(h_i,t)
w_i,t = softmax_t(Gate(z_i,t))
v_i,t = Value(z_i,t)
A_hat_i = sum_t w_i,t * v_i,t
```

当前代码直接回归宏观优势，而不是早期草案中的 pairwise ranking：

```text
L_head = MSE(A_hat_i, stop_gradient(A_i))
```

head optimizer 使用 `lr=1e-3`，loss coefficient 为 `0.1`。head 本身使用 fp32；actor
FSDP 参数仍按现有配置保存为 fp32，并在 bfloat16 autocast 下计算，没有为了 ELICA
改变模型精度。

## 6. 第四层：在潜在表示中逐 turn 遮挡

先用完整 turn 序列预测一次，再逐个遮掉一个 turn，只重新执行小预测头：

```text
d_i,t = F(h_i,1, ..., h_i,T)
        - F(h_i,1, ..., 0, ..., h_i,T)
```

如果遮掉某个 turn 后预测效用明显下降，说明在 prediction head 看来它更重要；如果几乎
不变，说明它对预测结果贡献较小。这个过程不调用环境，也不重新生成后续 action。

因此它是 **latent intervention**，不是 RECAP 的真实 action deletion replay。它衡量的
是 prediction head 内部的边际贡献，不能未经验证就称为真实环境因果效应。

将边际贡献在每条轨迹内部中心化：

```text
q_i,t = d_i,t - mean_t(d_i,t)
```

当前实现再把 turn credit 映射给该 turn 的每个 action token：

```text
A_final_i,k = A_i + beta * q_i,t,  k 属于 turn t
```

主实验使用 `beta=0.10`。没有合法 turn 的异常轨迹仍保留宏观 PTE-GRPO 优势，局部项为
零，不会因为解析失败而丢掉训练信号。

### 当前“守恒”的准确含义

代码保证：

```text
sum_t q_i,t = 0
```

当前代码使用 `elica/conservation_error=0` 检查这个 turn 数量口径。已完成的历史运行在
重命名前记录为 `hidden_credit/conservation_error`，但含义相同。由于
一个 turn 的 `q_i,t` 会复制到它的所有 token，不同 turn 长度不同时，并不严格保证：

```text
sum_t L_i,t * q_i,t = 0
```

所以当前版本是 **turn-mean centered**，不是严格的 token-weighted credit conservation。
后续正式版本应改成：

```text
q'_i,t = q_i,t - sum_j(L_i,j * q_i,j) / sum_j L_i,j
```

或者先将每个 turn 的 credit 除以其 token 数。这个问题需要作为后续高价值消融，不能把
当前实验写成已经证明了严格 token 守恒。

## 7. 与已有方法的边界

| 方法 | turn credit 从哪里来 | 是否额外环境重放 | 是否改变 rollout/inference |
|---|---|---:|---:|
| GiGPO | 相同环境状态下的组内 action 比较 | 否 | 否 |
| RECAP V2 | 删除 action 后真实执行后续 action | 是 | 训练时是 |
| SHADOW | 状态转移分组和局部动态优势 | 否 | 否 |
| AT²PO | rollout tree 的节点价值回传 | 需要树分支 | 是 |
| ELICA | actor hidden states 的效用预测与表示遮挡 | 否 | 否 |

ELICA 的核心并不是“首次做 turn-level credit”，而是以下组合：

> actor latent state + training-only utility head + turn masking marginal +
> accuracy/PTE joint utility + unchanged agent rollout。

截至 2026-09-15 的调研中，尚未发现已接收顶会论文采用这一完整组合。需要关注的近邻
预印本包括使用 actor hidden geometry 聚类的 BiPACE、使用冻结参考模型前缀概率 TD 的
TRACE，以及强调 action-to-token 严格守恒的 FACTOR。论文中应使用
“to our knowledge”，不能声称所有形式的 turn credit 都是首次提出。

## 8. 正式 from-scratch 实验设置

主运行：

```text
模型：Qwen2.5-1.5B-Instruct
环境：ALFWorld valid_seen，140 个任务
训练：from scratch，222 steps
train batch size：16
rollout n：8
validation frequency：10
save frequency：50
PTE cost coefficient：0.10
hidden credit beta：0.10
head width：128
head learning rate：1e-3
use_remove_padding：True
```

正式运行目录：

```text
outputs/hydra/alfworld_hidden_credit_pte_from_scratch/
  qwen2.5_1.5b_beta0.10_rmpad/2026-09-14_19-46-53/

outputs/validation_log/alfworld_hidden_credit_pte_from_scratch/
  qwen2.5_1.5b_beta0.10_rmpad/2026-09-14_19-46-53/
```

训练完整到达 step 222，并写出了最终 checkpoint 和 validation 文件。

## 9. 实验分析口径

### 9.1 推理效率

使用 `lab/analyze_alfworld_pte.py` 和同一个 Qwen2.5 tokenizer 重建每轮上下文。ALFWorld
每轮都会插入新的环境 observation，因此按保守口径把每轮视为一次 cache miss：

```text
PTE = sum_t(prefill_t + gamma * context_t * decode_t)
```

比较过程为：

1. 对齐相同顺序的 140 个 validation tasks；
2. 分别计算全任务 PTE、成功任务 PTE、turns 和 decoded tokens；
3. 对 ELICA 和 RECAP V2 都成功的 136 个任务再做配对比较；
4. 同时查看中位数与逐任务胜负，避免少量超长失败轨迹扭曲均值。

最强 RECAP V2 使用 step 200，而不是其退化后的 step 222。ELICA 同时报告同训练进度的
step 200 和最终 step 222。

### 9.2 训练效率

从两个训练日志中抽取 step 151–199，排除带 validation 或 checkpoint 保存的 steps，
最终各得到 45 个普通训练 step。这个区间已经越过早期长轨迹阶段，更适合比较稳定训练
成本。

需要注意：ELICA 运行同时使用了新的并行 slot 环境管理和 `use_remove_padding=True`，
而旧 RECAP V2 是旧环境管理。因此 wall-time 是当前系统的端到端结果，不能全部归因于
credit assignment 算法。

## 10. 完整实验结果

### 10.1 不同方法与 checkpoint

所有结果均为 Qwen2.5-1.5B、ALFWorld 的 140 个相同 validation tasks。

| 方法 | checkpoint | 成功数 | 全任务平均 PTE | 成功任务平均 PTE | 平均 turns | 平均生成 token |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla GRPO | 200 | 138 | 9,206.7 | 8,342.7 | 8.779 | 216.6 |
| PTE-GRPO | 200 | 136 | 8,258.8 | 6,958.2 | 8.757 | 111.7 |
| RECAP V2（最强） | 200 | 137 | 7,232.5 | **6,280.9** | 8.007 | 102.7 |
| ELICA-GRPO | 200 | **139** | 7,261.1 | 6,898.0 | 8.179 | 104.7 |
| ELICA-GRPO | 222 | **139** | **6,900.5** | 6,546.8 | **7.829** | **100.5** |

从 step 200 到 step 222，ELICA 保持 `139/140` 成功，同时：

- 全任务 PTE 从 `7,261.1` 降到 `6,900.5`，下降 `4.97%`；
- 平均 turns 从 `8.179` 降到 `7.829`，下降 `4.28%`；
- 平均生成 token 从 `104.7` 降到 `100.5`，下降 `3.92%`。

### 10.2 相对最强 RECAP V2 的严格比较

| 指标 | RECAP V2 step 200 | ELICA step 222 | 变化 |
|---|---:|---:|---:|
| 成功率 | 137/140，97.86% | **139/140，99.29%** | **+1.43 个百分点** |
| 全任务平均 PTE | 7,232.5 | **6,900.5** | **下降 4.59%** |
| PTE 中位数 | **5,802.1** | 5,887.6 | 上升 1.47% |
| 成功任务平均 PTE | **6,280.9** | 6,546.8 | 上升 4.23% |
| 平均 turns | 8.007 | **7.829** | **下降 2.23%** |
| 平均生成 token | 102.7 | **100.5** | **下降 2.10%** |

逐任务配对结果：

```text
ELICA PTE 更低：54 / 140
两者相同：       29 / 140
RECAP V2 更低：  57 / 140
双方都成功：    136 / 140
```

在双方都成功的 136 个任务上，ELICA 的平均 PTE 比 RECAP V2 **高 2.46%**。因此当前
结果支持两种不同强度的结论：

- **可以说**：ELICA 在成功率约束下获得了更低的全任务期望 PTE，是 headline 指标上的
  Pareto 改进；
- **不能说**：ELICA 已经全面支配 RECAP V2 的成功轨迹效率。

出现这个差异的主要原因是失败轨迹非常昂贵。RECAP V2 有 3 条失败，失败轨迹平均 PTE
约 `50,689`；ELICA 只有 1 条失败。全任务平均 PTE 的优势很大一部分来自少了两条昂贵
失败，而不是每条成功路径都比 RECAP V2 更短。

### 10.3 学习过程

| step | 成功数 | 成功率 | validation 平均 turns |
|---:|---:|---:|---:|
| 10 | 5 | 3.57% | 30.400 |
| 50 | 89 | 63.57% | 18.371 |
| 100 | 132 | 94.29% | 9.814 |
| 150 | 137 | 97.86% | 9.300 |
| 190 | 137 | 97.86% | 8.786 |
| 200 | 139 | 99.29% | 8.179 |
| 210 | 138 | 98.57% | 8.486 |
| 220 | 137 | 97.86% | 8.557 |
| 222 | 139 | 99.29% | **7.829** |

训练没有表现出持续成功率崩溃。step 222 与 step 200 并列最高成功数，同时拥有最低
validation turns，因此选择 step 222 不是用低成功率换取低成本。

### 10.4 prediction head 是否正常学习

后期日志中的典型指标如下：

| step | head loss | head grad norm | turn 中心化误差 | 空 turn 轨迹 |
|---:|---:|---:|---:|---:|
| 151 | 0.2705 | 0.1300 | 0.0 | 0 |
| 199 | 0.6155 | 0.1982 | 0.0 | 0 |
| 200 | 0.5566 | 0.1839 | 0.0 | 0 |
| 222 | 0.5331 | 0.1683 | 0.0 | 0 |

loss 没有单调下降，因为 actor、rollout 分布和宏观 advantage 都在持续变化；但 head
梯度始终有限，没有 NaN、梯度爆炸或全程零梯度。这个结果只能证明 head 训练稳定，不能
单独证明它学到了真实 action 因果贡献。

## 11. 训练速度结果

step 151–199 的 45 个普通训练 step：

| 平均耗时 | RECAP V2 | ELICA | 变化 |
|---|---:|---:|---:|
| 完整普通 step | 301.8 s | **81.3 s** | **3.71×；下降 73.1%** |
| 轨迹生成相关部分 | 236.1 s | **14.8 s** | **下降 93.7%** |
| old log-prob | 10.04 s | 10.07 s | 基本相同 |
| reference forward | 13.57 s | 13.55 s | 基本相同 |
| actor update | **39.65 s** | 40.50 s | ELICA 慢约 2.2% |

这组分解说明：

1. 小 head 和 hidden pooling 没有造成严重 GPU 负担，actor update 只慢约 `0.85 s`；
2. 相对 RECAP V2 的主要训练优势来自不做 action deletion replay；
3. 环境 slot 并行和 remove-padding 也贡献了加速，所以不能把完整 `3.71×` 都写成 ELICA
   算法本身的收益；
4. checkpoint 保存约需 30 分钟，会严重扭曲含保存 step 的 wall time，因此没有把
   step 200/222 的总耗时拿来比较普通训练速度。

## 12. attention 与 latent intervention 的机制解释

### 12.1 适合论文的直观叙事

一条长轨迹并不是若干相互独立的 action。每个 turn 的 actor hidden state 已经读取了
此前的 action 和环境 observation，因此它不只表示“这一轮说了什么”，也隐含了：

- Agent 当前认为任务进行到了哪里；
- 这一轮是否重复了已有信息或路线；
- 这一轮是否带来了后续 action 所依赖的新状态；
- 当前行为模式通常对应成功、失败、短路径还是高 PTE 绕路。

ELICA 的 prediction head 可以理解为一个训练期的“轨迹阅读器”。它先让不同 turns 竞争
有限的注意力，再判断这些被关注的 turn 对成功率—PTE 综合效用是正向还是负向的：

```text
w_i,t = softmax_t(Gate(Encoder(h_i,t)))
v_i,t = Value(Encoder(h_i,t))
A_hat_i = sum_t w_i,t * v_i,t
```

这里 `w_i,t` 回答“预测整条轨迹时主要看哪些 turns”，`v_i,t` 回答“这个 turn 的表示
更支持高效成功还是低效/失败”。二者必须结合：高 attention 的 turn 也可能带来负向值，
低 attention 的 turn 也不能直接判定为冗余。

这种设计对冗余 action 有一个自然解释。假设 Agent 两次进入同一个房间并获得近似信息，
两个 turn 的 hidden representations 会包含大量重复内容。它们在 softmax gate 中相互
替代：遮掉其中一个后，另一个仍能向 head 提供类似证据，预测效用变化很小。相反，拿起
目标物体或完成最终放置的 turn 往往提供不可替代的状态变化；遮掉后，其余 turns 难以
补偿，预测效用会明显变化。

可以把这个机制概括为：

> **attention 负责发现轨迹中的效用证据，latent intervention 负责判断这份证据是否可被
> 其他 turns 替代。**

最终分配 credit 使用的不是 raw attention，而是遮挡前后的预测差：

```text
d_i,t = A_hat_i - A_hat_i_without_turn_t
q_i,t = d_i,t - mean_t(d_i,t)
```

因此：

- `d_i,t > 0`：遮掉该 turn 会降低预测优势，它更支持高效完成任务；
- `d_i,t < 0`：遮掉该 turn 会提高预测优势，它更像低效或有害行为；
- `d_i,t` 接近 0：在 head 看来，该 turn 提供的信息容易被其他 turns 替代；
- `q_i,t` 表示同一条轨迹内部的相对责任，而不是跨任务的绝对质量分数。

PTE 并没有直接输入 prediction head。它只通过宏观训练目标 `A_i` 提供监督。这一点对
论文叙事很重要：head 不能简单地看到“这个 turn 很长，所以惩罚它”，而需要从 actor
表示中学习哪些行为模式通常产生高 PTE，例如重复搜索、回到旧位置或在已经足够的信息上
继续推理。同时，由于成功奖励在宏观效用中占主导，模型不能仅靠缩短轨迹获得高 credit。

整体机制可以画成：

```text
actions + observations
          |
          v
actor contextual hidden states h_1, ..., h_T
          |
          v
attention-like gate: 找到与轨迹效用相关的 turns
          |
          v
latent leave-one-turn-out: 检查该证据能否被其他 turns 替代
          |
          v
centered q_t: 在同一轨迹内部重新分配 PTE-GRPO advantage
```

### 12.2 为什么不能只用 attention 权重

raw gate weight 只是模型内部相关性，不能直接证明：

```text
低 attention = 冗余 action
高 attention = 关键 action
```

例如两个几乎相同的必要 turns 可能平分 attention；一个明显的错误 turn 也可能因为很能
解释失败而得到高 attention。另一方面，遮掉一个 turn 后，所有剩余 `w_i,t` 都会重新
归一化，`d_i,t` 同时反映被遮 turn 的 value 和其他 turns 的可替代能力。因此它比单看
`w_i,t` 更接近 ELICA 想回答的“这一轮对整条轨迹有多大边际影响”。

还需要准确说明：当前 head 是每个 turn 独立经过 MLP 后再做 softmax gate，并没有显式的
`QK^T` 跨轮 self-attention 矩阵。跨轮信息有两个来源：

1. 每个 actor hidden state 本身已经编码之前的交互历史；
2. 所有 turns 在 gate 的 softmax 和完整轨迹预测中共同竞争。

因此论文中可以称为 **attention-like turn aggregation** 或 **gated turn attention**，不应
写成已经显式学习了一张 turn-to-turn attention graph。若未来加入轻量 turn Transformer，
才可以直接分析成对的跨轮 attention；这属于可能的扩展，不是当前实验已经验证的组件。

### 12.3 可检验的机制假设

机制解释需要变成能被实验否证的假设，而不能只展示几条好看的轨迹：

| 假设 | 预期现象 | 验证方式 |
|---|---|---|
| 重复 turn 可被替代 | hidden 表示相似、`|d_i,t|` 较小、删除后仍成功 | 与真实 action deletion 标签比较 |
| 关键状态变化不可替代 | `d_i,t` 明显为正，删除后更容易失败 | 关键/可删除 turn 的 AUC |
| 有害绕路应承担负 credit | `q_i,t < 0`，常伴随重复位置或失败 observation | 按 action/observation 类型分组统计 |
| head 不只学习位置和长度 | 控制 turn index、token 数后，`q_i,t` 仍有预测力 | 位置长度基线与残差/分层分析 |
| 效率改善不靠牺牲成功 | 高 credit 集中在任务必要 turns，而不是简单偏爱短 turn | 同时报告成功率、PTE 和删除结果 |

已有 action deletion replay 可以作为 **只用于评估的外部标签**，不参与 ELICA 训练。
在未参与训练的任务上，比较以下信号区分“删除后仍成功”和“删除后失败”的能力：

```text
ELICA centered credit q_i,t
latent marginal d_i,t
raw gate weight w_i,t
hidden cosine similarity
单轮 PTE
token 长度
turn 位置
policy entropy
```

最关键的证据不是 `q_i,t` 与 PTE 有普通相关性，而是：控制位置、长度和单轮 PTE 后，
`q_i,t` 仍然更能预测真实删除结果。如果成立，就能支持如下机制结论：

> actor 的 contextual hidden states 含有跨轮依赖和可替代性信息；ELICA 通过 gated
> aggregation 找到与轨迹效用相关的证据，再通过 latent intervention 将轨迹级结果归因
> 给不同 turns，而不需要在训练中执行环境反事实重放。

### 12.4 论文中建议展示的图

建议选择一条同时含关键 action 和冗余绕路的完整轨迹，按 turn 展示：

```text
action | observation 摘要 | turn PTE | gate weight | value | d_t | q_t | 删除后是否成功
```

再用全量轨迹给出三项统计，而不是只依赖案例：

1. `q_i,t` 识别真实关键/可删除 turns 的 AUC，并与全部简单基线比较；
2. 控制位置、长度和 PTE 后的相关性或分层 AUC；
3. 随训练 step 变化的 PTE、成功率、gate concentration 和 credit 分布。

案例负责让机制直观，全量统计负责证明案例不是挑选出来的。attention 图只作为辅助说明，
真实删除标签和配对统计才是主要机制证据。

## 13. 当前已经证明与尚未证明的内容

### 已经证明

- ELICA 已在真实 verl/FSDP/ALFWorld 流程中从头完整训练 222 steps；
- prediction head、hidden pooling、turn masking 和 token credit 注入能稳定运行；
- validation/inference 不调用 prediction head，推理流程和模型大小不变；
- 一个 seed 上达到 `139/140`，全任务 PTE 为 `6,900.5`；
- headline 成功率和全任务期望 PTE 同时超过当前最强 RECAP V2；
- 当前系统训练普通 step 明显快于需要环境重放的 RECAP V2。

### 尚未证明

- 不能仅凭这一次训练证明增益一定来自 hidden-state credit，而不是随机种子或 PTE-GRPO
  本身；还缺同配置 `beta=0`；
- 没有证明 head 超过位置、长度、entropy 等简单捷径；
- 没有证明 attention 或 latent masking 等价于真实环境因果贡献；
- 成功轨迹条件 PTE 尚未超过 RECAP V2；
- 当前只覆盖 ALFWorld、Qwen2.5-1.5B 和一个 seed；
- 当前代码尚未做到严格 token-weighted credit conservation。

## 14. 下一步最有价值的实验

按信息价值排序：

1. **同配置 `beta=0` 对照**：复用当前并行环境和 remove-padding，隔离 hidden credit 的
   净贡献。这是主张 ELICA 有效所必需的实验。
2. **严格 token 守恒版本**：比较当前 turn-mean centered 与 token-weighted centered，
   同时消除 turn 长度带来的隐式更新强度差异。
3. **真实删除标签机制审计**：不把 replay 用于训练，只用于测试 `q_i,t` 对关键/冗余
   action 的 AUC，以及控制位置和长度后的相关性。
4. **低成本 head 消融**：hidden head、只用位置/长度的 head、raw gate、均分 credit。
5. **扩展主实验**：至少三个 seeds，并增加 WebShop/ScienceWorld 中的两个 benchmark，
   再验证更大模型。

在完成第 1–3 项前，ELICA 应被表述为“有潜力的新方法和一个强单次实验结果”，而不是
已经充分验证的顶会结论。

## 15. 复现入口

正式训练脚本：

```text
lab/qwen2_5_alfworld_elica_grpo.sh
```

默认启动方式：

```bash
conda run -n verl-vllm bash lab/qwen2_5_alfworld_elica_grpo.sh
```

通用 validation PTE 对比工具：

```text
lab/analyze_alfworld_pte.py
```

训练所需的正式实现位于：

```text
verl/experimental/pte_grpo/core.py
verl/experimental/pte_grpo/elica.py
verl/experimental/pte_grpo/agent_loop.py
verl/workers/actor/dp_actor.py
verl/trainer/ppo/ray_trainer.py
```
