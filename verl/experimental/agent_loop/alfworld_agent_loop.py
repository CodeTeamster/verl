# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import asyncio
from typing import Any
from uuid import uuid4
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

AlfWORLD_SYSTEM_PROMPT = """You are an agent interacting with a virtual text-based environment.

## Notes:
At each step, you should first think then perform action to fulfill the instruction. You should ALWAYS wrap your thinking with the <think> </think> tag and wrap your action with the <action> </action> tag.
You should ALWAYS take one action each step.
DO NOT try to interact with the user at anytime. Finish the task by yourself.

## Action Format:
Below are the available commands you can use:
  look:                             look around your current location
  inventory:                        check the object you are currently holding (you can only hold one)
  go to (receptacle):               move to a receptacle
  open (receptacle):                open a receptacle
  close (receptacle):               close a receptacle
  take (object) from (receptacle):  take an object from a receptacle
  move (object) to (receptacle):    place an object that you are holding in or on a receptacle
  examine (something):              examine a receptacle or an object to learn its properties
  use (object):                     use an object
  heat (object) with (receptacle):  heat an object using a receptacle
  clean (object) with (receptacle): clean an object using a receptacle
  cool (object) with (receptacle):  cool an object using a receptacle
  slice (object) with (object):     slice an object using a sharp object

For example your output should be like this:
<think> To solve the task, I need first to ... </think><action>go to cabinet 1</action>
"""


@register("alfworld_agent")
class ALFWorldAgentLoop(AgentLoopBase):
    def __init__(self, *args, environment_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment_manager = environment_manager

        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def _format_observation(self, observation: str) -> str:
        if "Nothing happens" in observation:
            observation += f" Please check if the action you take is valid or you have carefully followed the action format."
        return "Observation: " + observation

    def _parse_action(self, response: str) -> str | None:
        start_tag, end_tag = "<action>", "</action>"
        start = response.find(start_tag)
        end = response.find(end_tag, start + len(start_tag)) if start >= 0 else -1
        if start >= 0 and end >= 0:
            action = response[start + len(start_tag) : end].strip()
            if action:
                return action

        return None

    async def _tokenizer_encode(self, requests_text: list[dict]) -> str:
        requests_ids = await asyncio.to_thread(
            self.tokenizer.apply_chat_template,
            requests_text,
            add_generation_prompt=True,
            tokenize=True,
        )
        return requests_ids

    async def _tokenizer_decode(self, responses_ids: list[int]) -> str:
        responses_text = await asyncio.to_thread(self.tokenizer.decode, responses_ids)
        return responses_text

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # kwargs example: {
        #     'uid': '',
        #     "target":"",
        #     "agent_name":"alfworld_agent",
        #     "prompt":[{"content":"","role":"system"},{"content":"","role":"user"}],
        #     "raw_prompt":[{"content":"","role":"system"},{"content":"","role":"user"}],
        #     "data_source":"alfworld",
        #     "ability":"alfworld",
        #     "reward_model":{"ground_truth":1,"style":"rule"},
        #     'tools_kwargs': {},
        #     "extra_info":{
        #         "data_root_env":"ALFWORLD_DATA",
        #         "game_file":"/path/to/xxx.tw-pddl",
        #     }
        # }
        if self.environment_manager is None:
            raise RuntimeError("ALFWorldEnvironmentManager must be injected by AgentLoopWorker.")

        async with self.environment_manager.episode(kwargs["extra_info"]) as env:
            observation = env.reset.observation
            context = [
                {"role": "system", "content": AlfWORLD_SYSTEM_PROMPT},
                {"role": "user", "content": self._format_observation(observation)},
            ]

            context_ids = await self._tokenizer_encode(context)
            context_ids = context_ids[:self.prompt_length]

            response_mask = []
            response_logprobs = []
            assistant_turns, final_reward = 0, 0.0
            metrics = {}
            request_id = uuid4().hex
            while True:
                remaining_response_tokens = self.response_length - len(response_mask)
                if remaining_response_tokens <= 0:
                    break

                turn_sampling_params = {
                    **sampling_params,
                    "max_tokens": remaining_response_tokens,
                }
                with simple_timer("generate_sequences", metrics):
                    step_output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=context_ids,
                        sampling_params=turn_sampling_params,
                    )
                response_ids = step_output.token_ids
                if not response_ids:
                    logger.warning("ALFWorld rollout stopped because the model returned an empty response.")
                    break

                context_ids += response_ids
                response_mask += [1] * len(response_ids)
                if step_output.log_probs is not None:
                    if len(step_output.log_probs) != len(response_ids):
                        raise ValueError(
                            "ALFWorld rollout received mismatched token and logprob lengths: "
                            f"{len(response_ids)} tokens, {len(step_output.log_probs)} logprobs."
                        )
                    response_logprobs += step_output.log_probs
                assistant_turns += 1

                response = await self._tokenizer_decode(response_ids)
                action = self._parse_action(response)
                if action is None:
                    metrics["invalid_action_format"] = metrics.get("invalid_action_format", 0) + 1
                    break
                try:
                    results = await env.step(action)
                except Exception:
                    logger.exception("ALFWorld environment step failed for action %r", action)
                    break
                # last obeservation is EOS, no need to add to context
                if results.done:
                    final_reward = results.reward
                    break

                if len(response_mask) >= self.response_length:
                    break

                if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                    break

                observation = [{
                    "role": "user",
                    "content": self._format_observation(results.observation),
                }]
                observation_ids = await self.apply_chat_template(
                    observation,
                    remove_system_prompt=True,
                )

                # NOTE: last turn should not be user turn, or the EOS token reward
                # can't be propagated to previous token in GAE.
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

            output = AgentLoopOutput(
                prompt_ids=context_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                multi_modal_data={},
                response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
                num_turns=assistant_turns,
                reward_score=final_reward,
                metrics=metrics,
                extra_fields={},
            )

        return output
