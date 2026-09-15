"""ALFWorld rollout metadata collection for PTE-GRPO.

This loop preserves ALFWorldAgentLoop's action generation and environment
interaction exactly.  It only emits per-request token counts for the trainer.
"""

import time
from typing import Any
from uuid import uuid4

import hydra
import ray

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _agent_loop_registry,
    register,
    rollout_trace_attr,
)
from verl.experimental.agent_loop.alfworld_agent_loop import ALFWorldAgentLoop, AlfWORLD_SYSTEM_PROMPT, logger
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from . import core  # noqa: F401
from .core import proportional_replay_budget, select_anchored_replay_candidates, select_replay_candidates


@register("pte_grpo_alfworld_agent")
class PTEGRPOALFWorldAgentLoop(ALFWorldAgentLoop):
    """Standard ALFWorld rollout with PTE accounting metadata."""

    async def _replay_deleted_turn(
        self,
        env,
        turns: list[dict[str, Any]],
        deleted_turn: int,
        gamma: float,
    ) -> tuple[float, float]:
        """Replay logged actions after deleting one turn, without an LLM call."""
        reset, replay_results = await env.replay.remote(
            [turn["action"] for turn in turns],
            deleted_turn,
        )
        context = [
            {"role": "system", "content": AlfWORLD_SYSTEM_PROMPT},
            {"role": "user", "content": self._format_observation(reset.observation)},
        ]
        context_ids = (await self._tokenizer_encode(context))[: self.prompt_length]
        replay_pte = 0.0
        result_index = 0
        for turn_index, turn in enumerate(turns):
            if turn_index == deleted_turn:
                continue
            generated_ids = turn["generated_ids"]
            prefill_tokens = len(context_ids)
            replay_pte += prefill_tokens + gamma * prefill_tokens * len(generated_ids)
            context_ids += generated_ids
            if result_index >= len(replay_results):
                raise RuntimeError("ALFWorld replay returned fewer environment results than replayed actions.")
            result = replay_results[result_index]
            result_index += 1
            if result.done:
                return float(result.reward), replay_pte
            observation_ids = await self.apply_chat_template(
                [{"role": "user", "content": self._format_observation(result.observation)}],
                remove_system_prompt=True,
            )
            context_ids += observation_ids
        return 0.0, replay_pte

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        if self.environment_manager is None:
            raise RuntimeError("ALFWorldEnvironmentManager must be injected by PTEGRPOAgentLoopWorker.")
        async with self.environment_manager.episode(kwargs["extra_info"]) as (env, reset):
            observation = reset.observation
            context = [
                {"role": "system", "content": AlfWORLD_SYSTEM_PROMPT},
                {"role": "user", "content": self._format_observation(observation)},
            ]
            context_ids = (await self._tokenizer_encode(context))[: self.prompt_length]
            response_mask: list[int] = []
            response_logprobs: list[float] = []
            pte_turns: list[dict[str, int]] = []
            replay_turns: list[dict[str, Any]] = []
            recap_stats = {
                "eligible": 0,
                "tested": 0,
                "removable": 0,
                "necessary": 0,
                "pte_saving": 0.0,
                "replay_seconds": 0.0,
            }
            assistant_turns, final_reward = 0, 0.0
            metrics: dict[str, Any] = {}
            request_id = uuid4().hex
            while True:
                remaining_response_tokens = self.response_length - len(response_mask)
                if remaining_response_tokens <= 0:
                    break
                prefill_tokens = len(context_ids)
                with simple_timer("generate_sequences", metrics):
                    step_output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=context_ids,
                        sampling_params={**sampling_params, "max_tokens": remaining_response_tokens},
                    )
                generated_ids = step_output.token_ids
                if not generated_ids:
                    logger.warning("ALFWorld rollout stopped because the model returned an empty response.")
                    break
                turn_start = len(response_mask)
                context_ids += generated_ids
                response_mask += [1] * len(generated_ids)
                if step_output.log_probs is not None:
                    if len(step_output.log_probs) != len(generated_ids):
                        raise ValueError("ALFWorld rollout received mismatched token and logprob lengths.")
                    response_logprobs += step_output.log_probs
                assistant_turns += 1
                action = self._parse_action(await self._tokenizer_decode(generated_ids))
                if action is None:
                    metrics["invalid_action_format"] = metrics.get("invalid_action_format", 0) + 1
                    break
                try:
                    results = await env.step.remote(action)
                except Exception:
                    logger.exception("ALFWorld environment step failed for action %r", action)
                    break
                pte_turns.append(
                    {
                        "prefill_tokens": prefill_tokens,
                        "decode_tokens": len(generated_ids),
                        "response_start": turn_start,
                        "response_end": len(response_mask),
                    }
                )
                replay_turns.append(
                    {
                        "action": action,
                        "generated_ids": list(generated_ids),
                        "start": turn_start,
                        "end": len(response_mask),
                        "prefill_tokens": prefill_tokens,
                        "decode_tokens": len(generated_ids),
                        "observation_tokens": 0,
                        "tested": False,
                        "removable": False,
                    }
                )
                if results.done:
                    final_reward = results.reward
                    break
                if len(response_mask) >= self.response_length or (
                    self.max_assistant_turns and assistant_turns >= self.max_assistant_turns
                ):
                    break
                observation = results.observation
                observation_ids = await self.apply_chat_template(
                    [{"role": "user", "content": self._format_observation(observation)}], remove_system_prompt=True
                )
                if len(response_mask) + len(observation_ids) >= self.response_length:
                    break
                context_ids += observation_ids
                response_mask += [0] * len(observation_ids)
                replay_turns[-1]["observation_tokens"] = len(observation_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(observation_ids)

            recap_config = self.config.algorithm.get("recap_grpo", {})
            fixed_replay_budget = int(recap_config.get("replay_budget", 0))
            replay_ratio = recap_config.get("replay_ratio")
            max_replay_budget = int(recap_config.get("max_replay_budget", fixed_replay_budget))
            min_replay_budget = int(recap_config.get("min_replay_budget", 1))
            replay_budget_rounding = str(recap_config.get("replay_budget_rounding", "ceil"))
            candidate_selection = str(recap_config.get("candidate_selection", "quota"))
            configured_suffix_fraction = recap_config.get("suffix_fraction")
            suffix_fraction = (
                float(configured_suffix_fraction) if configured_suffix_fraction is not None else None
            )
            replay_budget = (
                proportional_replay_budget(
                    len(replay_turns),
                    float(replay_ratio),
                    max_replay_budget,
                    replay_budget_rounding,
                    min_replay_budget,
                )
                if replay_ratio is not None
                else min(len(replay_turns), fixed_replay_budget)
            )
            gamma = float(recap_config.get("gamma", 0.00704))
            is_validation = bool(kwargs.get("_recap_validate", False))
            if final_reward > 0 and replay_budget > 0 and not is_validation and replay_turns:
                recap_stats["eligible"] = 1
                original_pte = sum(
                    turn["prefill_tokens"] + gamma * turn["prefill_tokens"] * turn["decode_tokens"]
                    for turn in replay_turns
                )
                local_cost = [
                    turn["prefill_tokens"] + gamma * turn["prefill_tokens"] * turn["decode_tokens"]
                    for turn in replay_turns
                ]
                suffix_priority = [
                    cost
                    + (len(replay_turns) - turn_index - 1)
                    * (turn["decode_tokens"] + turn["observation_tokens"])
                    for turn_index, (turn, cost) in enumerate(zip(replay_turns, local_cost, strict=True))
                ]
                if candidate_selection == "anchored_alternating":
                    candidates = select_anchored_replay_candidates(local_cost, suffix_priority, replay_budget)
                elif candidate_selection == "quota":
                    candidates = select_replay_candidates(
                        local_cost, suffix_priority, replay_budget, suffix_fraction
                    )
                else:
                    raise ValueError(f"Unknown recap_grpo.candidate_selection: {candidate_selection}.")
                for turn_index in candidates:
                    replay_turns[turn_index]["tested"] = True
                    replay_turns[turn_index]["original_reward"] = float(final_reward)
                    replay_turns[turn_index]["original_pte"] = float(original_pte)
                    replay_started = time.perf_counter()
                    try:
                        reward, counterfactual_pte = await self._replay_deleted_turn(
                            env, replay_turns, turn_index, gamma
                        )
                    except Exception:
                        logger.exception("RECAP replay failed for deleted turn %d", turn_index)
                        continue
                    finally:
                        recap_stats["replay_seconds"] += time.perf_counter() - replay_started
                    recap_stats["tested"] += 1
                    pte_saving = original_pte - counterfactual_pte
                    replay_turns[turn_index]["counterfactual_reward"] = reward
                    replay_turns[turn_index]["pte_saving"] = pte_saving
                    replay_turns[turn_index]["removable"] = reward > 0 and pte_saving > 0
                    if replay_turns[turn_index]["removable"]:
                        recap_stats["removable"] += 1
                        recap_stats["pte_saving"] += float(pte_saving)
                    elif reward <= 0:
                        recap_stats["necessary"] += 1
            if response_mask:
                response_ids = context_ids[-len(response_mask) :]
                context_ids = context_ids[: -len(response_mask)]
            else:
                response_ids = []
            return AgentLoopOutput(
                prompt_ids=context_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                multi_modal_data={},
                response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
                num_turns=assistant_turns,
                reward_score=final_reward,
                metrics=metrics,
                extra_fields={"pte_turns": pte_turns, "recap_turns": replay_turns, "recap_stats": recap_stats},
            )


class PTEGRPOAgentLoopWorker(AgentLoopWorker):
    async def _run_agent_loop(self, sampling_params: dict[str, Any], trajectory: dict[str, Any], *, agent_name: str, trace: bool = True, **kwargs):
        if agent_name == "alfworld_agent":
            agent_name = "pte_grpo_alfworld_agent"
        if agent_name != "pte_grpo_alfworld_agent":
            raise ValueError("PTEGRPOAgentLoopManager only supports ALFWorld agent rows.")
        with rollout_trace_attr(
            step=trajectory["step"], sample_index=trajectory["sample_index"], rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"], name="agent_loop", trace=trace,
        ):
            agent_loop = hydra.utils.instantiate(
                config=_agent_loop_registry[agent_name], trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager, tokenizer=self.tokenizer, processor=self.processor, dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data), environment_manager=self.alfworld_environment_manager,
            )
            output = await agent_loop.run(
                sampling_params, _recap_validate=trajectory["validate"], **kwargs
            )
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)


class PTEGRPOAgentLoopManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(PTEGRPOAgentLoopWorker)
        super().__init__(*args, **kwargs)
