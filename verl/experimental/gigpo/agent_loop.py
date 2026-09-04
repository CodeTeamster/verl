"""GiGPO's ALFWorld integration without changing the built-in agent loop.

The built-in ``alfworld_agent`` remains the default.  Selecting
``GiGPOAgentLoopManager`` maps that dataset agent name to the separately
registered ``gigpo_alfworld_agent`` and uses the same environment pool.
"""

from typing import Any
from uuid import uuid4

import hydra
import ray

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _agent_loop_registry,
    rollout_trace_attr,
)
from verl.experimental.agent_loop.alfworld_agent_loop import ALFWorldAgentLoop, AlfWORLD_SYSTEM_PROMPT, logger
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

# Register the custom estimator when the opt-in manager is imported on the
# trainer driver through ``agent_loop_manager_class``.
from . import core  # noqa: F401


@register("gigpo_alfworld_agent")
class GiGPOALFWorldAgentLoop(ALFWorldAgentLoop):
    """ALFWorld loop that additionally records one GiGPO record per action."""

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        if self.environment_manager is None:
            raise RuntimeError("ALFWorldEnvironmentManager must be injected by GiGPOAgentLoopWorker.")

        async with self.environment_manager.episode(kwargs["extra_info"]) as env:
            observation = env.reset.observation
            context = [
                {"role": "system", "content": AlfWORLD_SYSTEM_PROMPT},
                {"role": "user", "content": self._format_observation(observation)},
            ]
            context_ids = (await self._tokenizer_encode(context))[: self.prompt_length]

            response_mask: list[int] = []
            response_logprobs: list[float] = []
            # Each record is local to exactly one sampled trajectory.  ``start``
            # and ``end`` are positions in the flattened response portion, so
            # the estimator can scatter the local advantage only to its action.
            gigpo_turns: list[dict[str, Any]] = []
            assistant_turns, final_reward = 0, 0.0
            metrics: dict[str, Any] = {}
            request_id = uuid4().hex

            while True:
                remaining_response_tokens = self.response_length - len(response_mask)
                if remaining_response_tokens <= 0:
                    break

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
                        raise ValueError(
                            "ALFWorld rollout received mismatched token and logprob lengths: "
                            f"{len(generated_ids)} tokens, {len(step_output.log_probs)} logprobs."
                        )
                    response_logprobs += step_output.log_probs
                assistant_turns += 1

                action = self._parse_action(await self._tokenizer_decode(generated_ids))
                if action is None:
                    metrics["invalid_action_format"] = metrics.get("invalid_action_format", 0) + 1
                    break
                try:
                    results = await env.step(action)
                except Exception:
                    logger.exception("ALFWorld environment step failed for action %r", action)
                    break

                # Use the raw observation (before this action) as GiGPO's anchor,
                # matching the source implementation rather than tokenized history.
                gigpo_turns.append(
                    {
                        "anchor": observation,
                        "reward": float(results.reward),
                        "start": turn_start,
                        "end": len(response_mask),
                    }
                )
                if results.done:
                    final_reward = results.reward
                    break
                if len(response_mask) >= self.response_length:
                    break
                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    break

                observation = results.observation
                observation_ids = await self.apply_chat_template(
                    [{"role": "user", "content": self._format_observation(observation)}],
                    remove_system_prompt=True,
                )
                if len(response_mask) + len(observation_ids) >= self.response_length:
                    break
                context_ids += observation_ids
                response_mask += [0] * len(observation_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(observation_ids)

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
                extra_fields={"gigpo_turns": gigpo_turns},
            )


class GiGPOAgentLoopWorker(AgentLoopWorker):
    """Worker adapter that injects ALFWorld state into the registered GiGPO loop."""

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ):
        if agent_name == "alfworld_agent":
            agent_name = "gigpo_alfworld_agent"
        if agent_name != "gigpo_alfworld_agent":
            raise ValueError(
                "GiGPOAgentLoopManager only supports ALFWorld rows with "
                "agent_name='alfworld_agent' or 'gigpo_alfworld_agent'."
            )
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, f"Agent loop {agent_name} is not registered."
            agent_loop = hydra.utils.instantiate(
                config=_agent_loop_registry[agent_name],
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                environment_manager=self.alfworld_environment_manager,
            )
            output = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, trajectory["validate"], **kwargs)


class GiGPOAgentLoopManager(AgentLoopManager):
    """Opt-in manager; standard ``AgentLoopManager`` is untouched."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(GiGPOAgentLoopWorker)
        super().__init__(*args, **kwargs)
