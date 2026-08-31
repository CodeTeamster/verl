"""Evaluation without PPO training workers or training dataloaders."""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import ray
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.config import HFModelConfig, RolloutConfig


class StandaloneEvaluationVLLMReplica:
    """Launch vLLM for evaluation without checkpoint-engine workers.

    CheckpointEngineWorker exists to receive training weights.  A standalone
    evaluator loads a fixed Hugging Face model, so starting that layer only
    adds an unnecessary process group and startup cost.
    """

    def __new__(cls, *args, **kwargs):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

        class _Replica(vLLMReplica):
            async def init_standalone(self):
                from verl.workers.rollout.replica import RolloutMode
                from verl.utils.device import get_visible_devices_keyword

                if self.nnodes != 1:
                    raise NotImplementedError("Standalone evaluator currently supports one vLLM node.")

                self.rollout_mode = RolloutMode.STANDALONE
                visible_devices_key = get_visible_devices_keyword()
                visible_devices = os.environ.get(visible_devices_key)
                if not visible_devices:
                    visible_devices = ",".join(str(index) for index in range(self.gpus_per_replica_node))

                node_id = ray.get_runtime_context().get_node_id()
                self.workers = []
                self.servers = [
                    self.server_class.options(
                        name=f"evaluator_vllm_server_{self.replica_rank}",
                        num_cpus=0,
                        max_concurrency=self.max_concurrency,
                        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                            node_id=node_id, soft=False
                        ),
                        runtime_env={
                            "env_vars": {
                                visible_devices_key: visible_devices,
                                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                                "NCCL_CUMEM_ENABLE": "0",
                            }
                        },
                    ).remote(
                        config=self.config,
                        model_config=self.model_config,
                        rollout_mode=self.rollout_mode,
                        workers=self.workers,
                        replica_rank=self.replica_rank,
                        node_rank=0,
                        gpus_per_node=self.gpus_per_replica_node,
                        nnodes=1,
                        cuda_visible_devices=visible_devices,
                    )
                ]
                server = self.servers[0]
                await server.launch_server.remote()
                server_address, server_port = await server.get_server_address.remote()
                self._server_handle = server
                self._server_address = f"{server_address}:{server_port}"

        return _Replica(*args, **kwargs)


class StandaloneEvaluationAgentLoopManager:
    """Use the lightweight vLLM replica while retaining the normal AgentLoop."""

    @classmethod
    def create(cls, config):
        from verl.experimental.agent_loop import AgentLoopManager

        if config.actor_rollout_ref.rollout.name != "vllm":
            return AgentLoopManager.create(config=config)

        class _Manager(AgentLoopManager):
            def __init__(self, *args, **kwargs):
                self.rollout_replica_class = StandaloneEvaluationVLLMReplica
                super().__init__(*args, **kwargs)

        return _Manager.create(config=config)


def prepare_evaluator_config(config):
    """Merge evaluator input with model and rollout defaults used by servers."""
    defaults = OmegaConf.create(
        {
            "data": {
                "val_max_samples": -1,
                "val_sample_start": None,
                "val_sample_end": None,
                "val_batch_size": None,
                "validation_shuffle": False,
                "dataloader_num_workers": 0,
                "filter_overlong_prompts": False,
                "filter_overlong_prompts_workers": 1,
                "max_prompt_length": 1024,
                "max_response_length": 1024,
                "prompt_key": "prompt",
                "truncation": "error",
                "trust_remote_code": False,
            },
            "evaluator": {
                "output_dir": "outputs/evaluation",
                "save_generations": True,
                "log_generations": True,
            },
            "ray_kwargs": {"ray_init": {}},
        }
    )
    structured = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": OmegaConf.structured(HFModelConfig),
                "rollout": OmegaConf.structured(RolloutConfig),
            }
        }
    )
    config = OmegaConf.merge(defaults, structured, config)
    OmegaConf.resolve(config)

    rollout = config.actor_rollout_ref.rollout
    world_size = rollout.nnodes * rollout.n_gpus_per_node
    replica_size = (
        rollout.tensor_model_parallel_size
        * rollout.pipeline_model_parallel_size
        * rollout.data_parallel_size
    )
    if rollout.nnodes <= 0 or rollout.n_gpus_per_node <= 0:
        raise ValueError("Standalone evaluator requires rollout.nnodes and rollout.n_gpus_per_node to be positive.")
    if world_size % replica_size != 0:
        raise ValueError(
            f"rollout world size ({world_size}) must be divisible by TP * PP * DP ({replica_size})."
        )
    if not config.data.val_files:
        raise ValueError("data.val_files must not be empty.")
    sample_start = config.data.val_sample_start
    sample_end = config.data.val_sample_end
    if (sample_start is None) != (sample_end is None):
        raise ValueError("data.val_sample_start and data.val_sample_end must be set together.")
    if sample_start is not None and (sample_start < 0 or sample_end <= sample_start):
        raise ValueError("data.val_sample_range must satisfy 0 <= start < end.")
    return config


class RolloutEvaluator:
    """Run validation rollouts with standalone inference servers and AgentLoop workers."""

    def __init__(self, config):
        self.config = config
        self.model_config = omega_conf_to_dataclass(
            config.actor_rollout_ref.model,
            dataclass_type=HFModelConfig,
        )
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.rollout_config = config.actor_rollout_ref.rollout

    def _build_dataloader(self) -> DataLoader:
        sample_start = self.config.data.val_sample_start
        sample_end = self.config.data.val_sample_end
        # Let the evaluator select a deterministic contiguous interval below.
        # Passing val_max_samples here would select before the requested range.
        max_samples = -1 if sample_start is not None else self.config.data.val_max_samples
        dataset = create_rl_dataset(
            self.config.data.val_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=False,
            max_samples=max_samples,
        )
        if sample_start is not None:
            total = len(dataset)
            if sample_end > total:
                raise ValueError(
                    f"data.val_sample_end ({sample_end}) exceeds the available validation samples ({total})."
                )
            dataset.dataframe = dataset.dataframe.select(range(sample_start, sample_end))
            print(f"selected fixed validation range [{sample_start}, {sample_end}) out of {total} samples")
        batch_size = self.config.data.val_batch_size or len(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.config.data.dataloader_num_workers,
            shuffle=self.config.data.validation_shuffle,
            drop_last=False,
            collate_fn=collate_fn,
        )

    @staticmethod
    def _ensure_uids(batch: DataProto) -> None:
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch))],
                dtype=object,
            )

    @staticmethod
    def _non_tensor_values(batch: DataProto, key: str, default: Any) -> list[Any]:
        values = batch.non_tensor_batch.get(key)
        if values is None:
            return [default] * len(batch)
        return values.tolist() if hasattr(values, "tolist") else list(values)

    def _prepare_generation_batch(self, batch: DataProto) -> DataProto:
        self._ensure_uids(batch)
        batch = batch.repeat(
            repeat_times=self.rollout_config.val_kwargs.n,
            interleave=True,
        )
        batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.rollout_config.val_kwargs.do_sample,
            "validate": True,
            "global_steps": 0,
        }
        return batch

    @staticmethod
    def _flatten_metrics(
        data_sources: list[str],
        sample_uids: list[str],
        reward_extra_infos: dict[str, list[Any]],
        sample_turns: list[int],
    ) -> dict[str, float]:
        grouped = process_validation_metrics(data_sources, sample_uids, reward_extra_infos)
        metrics: dict[str, float] = {}
        for data_source, var2metric2value in grouped.items():
            core_var = "acc" if "acc" in var2metric2value else "reward"
            for var_name, metric2value in var2metric2value.items():
                max_n = max(int(name.split("@")[-1].split("/")[0]) for name in metric2value)
                for name, value in metric2value.items():
                    section = "val-core" if (
                        var_name == core_var
                        and name.startswith(("mean", "maj", "best"))
                        and f"@{max_n}" in name
                    ) else "val-aux"
                    metrics[f"{section}/{data_source}/{var_name}/{name}"] = float(value)
        if sample_turns:
            turns = np.asarray(sample_turns)
            metrics["val-aux/num_turns/min"] = float(turns.min())
            metrics["val-aux/num_turns/max"] = float(turns.max())
            metrics["val-aux/num_turns/mean"] = float(turns.mean())
        return metrics

    @staticmethod
    def _dump_generations(
        output_dir: Path,
        inputs: list[str],
        outputs: list[str],
        ground_truths: list[Any],
        scores: list[float],
        reward_extra_infos: dict[str, list[Any]],
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "generations.jsonl"
        with output_file.open("w", encoding="utf-8") as file:
            for index, (prompt, response, ground_truth, score) in enumerate(
                zip(inputs, outputs, ground_truths, scores, strict=True)
            ):
                row = {
                    "input": prompt,
                    "output": response,
                    "gts": ground_truth,
                    "score": score,
                    "step": 0,
                }
                row.update(
                    {
                        key: values[index]
                        for key, values in reward_extra_infos.items()
                        if len(values) == len(scores)
                    }
                )
                file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return output_file

    def run(self) -> dict[str, float]:
        dataloader = self._build_dataloader()
        # Importing the manager registers ALFWorldAgentLoop and other built-in loops.
        manager = StandaloneEvaluationAgentLoopManager.create(config=self.config)

        data_sources: list[str] = []
        sample_uids: list[str] = []
        sample_turns: list[int] = []
        reward_extra_infos: dict[str, list[Any]] = defaultdict(list)
        inputs: list[str] = []
        outputs: list[str] = []
        ground_truths: list[Any] = []
        scores: list[float] = []

        for batch_dict in tqdm(dataloader, desc="Validation"):
            batch = self._prepare_generation_batch(DataProto.from_single_dict(batch_dict))
            padded_batch, pad_size = pad_dataproto_to_divisor(batch, self.rollout_config.agent.num_workers)
            generated = unpad_dataproto(manager.generate_sequences(padded_batch), pad_size)

            if "rm_scores" not in generated.batch:
                raise RuntimeError(
                    "Standalone evaluator requires AgentLoop or a reward worker to return rm_scores."
                )

            batch_scores = generated.batch["rm_scores"].sum(-1).cpu().tolist()
            batch_inputs = [
                self.tokenizer.decode(token_ids, skip_special_tokens=True)
                for token_ids in generated.batch["prompts"]
            ]
            batch_outputs = [
                self.tokenizer.decode(token_ids, skip_special_tokens=True)
                for token_ids in generated.batch["responses"]
            ]

            inputs.extend(batch_inputs)
            outputs.extend(batch_outputs)
            scores.extend(batch_scores)
            sample_uids.extend(generated.non_tensor_batch["uid"].tolist())
            data_sources.extend(self._non_tensor_values(generated, "data_source", "unknown"))
            sample_turns.extend(self._non_tensor_values(generated, "__num_turns__", 0))
            ground_truths.extend(
                item.get("ground_truth") if isinstance(item, dict) else None
                for item in self._non_tensor_values(generated, "reward_model", {})
            )

            reward_extra_infos["reward"].extend(batch_scores)
            for key in generated.meta_info.get("reward_extra_keys", []):
                reward_extra_infos[key].extend(self._non_tensor_values(generated, key, None))

        metrics = self._flatten_metrics(data_sources, sample_uids, reward_extra_infos, sample_turns)
        output_dir = Path(self.config.evaluator.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.config.evaluator.save_generations:
            generations_path = self._dump_generations(
                output_dir,
                inputs,
                outputs,
                ground_truths,
                scores,
                reward_extra_infos,
            )
            print(f"Saved generations to {generations_path}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics
