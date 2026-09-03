
```py
for batch_dict in self.train_dataloader:
    for key, value in batch_dict.items():
        print(f"batch_dict[{key}] = {value.shape} {value.dtype}")

# 任务能力/任务领域
batch_dict[ability] = (16,) object
# 提供奖励计算所需信息，比如来源 ground truth
batch_dict[reward_model] = (16,) object
# parquet中额外信息
batch_dict[extra_info] = (16,) object
# 原始的prompt，包含system和user的内容
batch_dict[raw_prompt] = (16,) object
# 样本在数据集中的索引
batch_dict[index] = (16,) object
# 工具调用相关
batch_dict[tools_kwargs] = (16,) object
# 多轮交互的额外配置
batch_dict[interaction_kwargs] = (16,) object
```

```py
batch: DataProto = DataProto.from_single_dict(batch_dict)

batch.non_tensor_batch[target] = ['' ... '']
batch.non_tensor_batch[agent_name] = ['alfworld_agent' ... 'alfworld_agent']
batch.non_tensor_batch[prompt] = [list([{'content': '', 'role': 'system'}, {'content': '', 'role': 'user'}])
    ...
    list([{'content': '', 'role': 'system'}, {'content': '', 'role': 'user'}])]
batch.non_tensor_batch[data_source] = ['alfworld_train' ... 'alfworld_train']
batch.non_tensor_batch[ability] = ['alfworld' ... 'alfworld']
batch.non_tensor_batch[reward_model] = [{'ground_truth': 1, 'style': 'rule'} ... {'ground_truth': 1, 'style': 'rule'}]
batch.non_tensor_batch[extra_info]
batch.non_tensor_batch[raw_prompt] = [list([{'content': '', 'role': 'system'}, {'content': '', 'role': 'user'}])
    ...
    list([{'content': '', 'role': 'system'}, {'content': '', 'role': 'user'}])]
batch.non_tensor_batch[index] = [1780 3423 2557 2868 720 207 516 3250 3394 2228 794 1423 1140 566 2658 2957]
batch.non_tensor_batch[tools_kwargs] = [{} ... {}]
batch.non_tensor_batch[interaction_kwargs] = [{} ... {}]
batch.non_tensor_batch[uid] = ['b1f4cc8a-28bb-4c72-9cca-402353c25907'
    ...
    '7becb38f-42fd-4a10-b57f-7718168f26dd']
```

```py
for key, value in batch.batch.items():
    print(f"batch.batch[{key}] = {value.shape} {value.dtype}")

batch.batch[dummy_tensor] = torch.Size([16, 1]) torch.uint8

batch.meta_info[temperature] = 1.0

gen_batch.non_tensor_batch[ability] = batch.non_tensor_batch[ability]
gen_batch.non_tensor_batch[raw_prompt] = batch.non_tensor_batch[raw_prompt]
gen_batch.non_tensor_batch[index] =
gen_batch.non_tensor_batch[prompt]
gen_batch.non_tensor_batch[agent_name]
gen_batch.non_tensor_batch[interaction_kwargs]
gen_batch.non_tensor_batch[tools_kwargs]
gen_batch.non_tensor_batch[target]
gen_batch.non_tensor_batch[data_source]
gen_batch.non_tensor_batch[reward_model]
gen_batch.non_tensor_batch[extra_info]
gen_batch.non_tensor_batch[uid]

```