"""Step-local GiGPO advantage added on top of the standard GRPO advantage."""

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo import core_algos


@core_algos.register_adv_est("gigpo")
def compute_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gigpo_turns: np.ndarray,
    config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep GRPO's trajectory advantage and add GiGPO's action-local signal.

    ``gigpo_turns`` has one list per trajectory.  A list item stores a raw
    anchor observation, the immediate environment reward, and the generated
    token span for that action.  Discounted step returns are normalized only
    among identical anchors from the same original prompt group.
    """
    algo_config = config or {}
    gigpo_config = algo_config.get("gigpo", {})
    # Match verl-agent: discounting is controlled by the shared algorithm.gamma.
    gamma = float(algo_config.get("gamma", 1.0))
    step_advantage_w = float(gigpo_config.get("step_advantage_w", 1.0))
    mode = gigpo_config.get("mode", "mean_norm")
    if mode == "mean_norm":
        norm_by_std = False
    elif mode == "mean_std_norm":
        norm_by_std = True
    else:
        raise ValueError(f"Unknown GiGPO mode: {mode!r}.")
    enable_similarity = bool(gigpo_config.get("enable_similarity", False))
    similarity_thresh = float(gigpo_config.get("similarity_thresh", 0.95))
    if enable_similarity and not 0.0 < similarity_thresh < 1.0:
        raise ValueError("GiGPO similarity_thresh must be in (0, 1) when similarity grouping is enabled.")
    epsilon = float(gigpo_config.get("epsilon", 1e-6))

    macro_advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=bool(algo_config.get("norm_adv_by_std_in_grpo", True)),
    )
    local_advantages = torch.zeros_like(macro_advantages)
    returns_by_row: list[list[tuple[int, int, float, str]]] = []

    for row, turns in enumerate(gigpo_turns):
        turns = turns or []
        discounted_return = 0.0
        reversed_turns: list[tuple[int, int, float, str]] = []
        for turn in reversed(turns):
            discounted_return = float(turn["reward"]) + gamma * discounted_return
            start, end = int(turn["start"]), int(turn["end"])
            if not (0 <= start < end <= response_mask.shape[1]):
                raise ValueError(f"Invalid GiGPO response span [{start}, {end}) for width {response_mask.shape[1]}.")
            reversed_turns.append((start, end, discounted_return, str(turn["anchor"])))
        returns_by_row.append(list(reversed(reversed_turns)))

    # Flatten in environment step order (step 0 of every rollout, then step
    # 1, ...), matching verl-agent's trajectory collector.  This matters only
    # for its intentionally greedy similarity-based anchor clustering.
    step_records: list[tuple[int, int, int, float, Any, str]] = []
    for turn_index in range(max((len(turns) for turns in returns_by_row), default=0)):
        for row, turns in enumerate(returns_by_row):
            if turn_index < len(turns):
                start, end, discounted_return, anchor = turns[turn_index]
                step_records.append((row, start, end, discounted_return, index[row], anchor))

    grouped_returns: list[list[tuple[int, int, int, float]]] = []
    by_prompt: dict[Any, list[tuple[int, int, int, float, Any, str]]] = defaultdict(list)
    for record in step_records:
        by_prompt[record[4]].append(record)
    for prompt_records in by_prompt.values():
        if not enable_similarity:
            exact_groups: dict[str, list[tuple[int, int, int, float, Any, str]]] = defaultdict(list)
            for record in prompt_records:
                exact_groups[record[5]].append(record)
            prompt_groups = exact_groups.values()
        else:
            clusters: list[list[tuple[int, int, int, float, Any, str]]] = []
            for record in prompt_records:
                for cluster in clusters:
                    if SequenceMatcher(None, record[5], cluster[0][5]).ratio() >= similarity_thresh:
                        cluster.append(record)
                        break
                else:
                    clusters.append([record])
            prompt_groups = clusters
        for group in prompt_groups:
            grouped_returns.append([(row, start, end, value) for row, start, end, value, _, _ in group])

    with torch.no_grad():
        for values in grouped_returns:
            returns = torch.tensor([item[3] for item in values], device=response_mask.device, dtype=torch.float32)
            centered = returns - returns.mean()
            if norm_by_std and len(values) > 1:
                centered = centered / (returns.std() + epsilon)
            for (row, start, end, _), advantage in zip(values, centered, strict=True):
                local_advantages[row, start:end] = advantage

    advantages = (macro_advantages + step_advantage_w * local_advantages) * response_mask
    return advantages, advantages
