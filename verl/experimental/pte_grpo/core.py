"""PTE-regularized outcome advantage without changing the agent policy flow."""

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo import core_algos


def proportional_replay_budget(
    num_turns: int,
    replay_ratio: float,
    max_replay_budget: int,
    rounding: str = "ceil",
) -> int:
    """Allocate a capped replay budget proportional to trajectory length."""
    if num_turns < 0:
        raise ValueError("num_turns must be non-negative.")
    if not 0.0 <= replay_ratio <= 1.0:
        raise ValueError("recap_grpo.replay_ratio must be in [0, 1].")
    if max_replay_budget < 0:
        raise ValueError("recap_grpo.max_replay_budget must be non-negative.")
    if rounding not in {"ceil", "nearest"}:
        raise ValueError("recap_grpo.replay_budget_rounding must be 'ceil' or 'nearest'.")
    raw_budget = replay_ratio * num_turns
    replay_budget = math.ceil(raw_budget) if rounding == "ceil" else math.floor(raw_budget + 0.5)
    if num_turns > 0 and replay_ratio > 0.0 and max_replay_budget > 0:
        replay_budget = max(1, replay_budget)
    return min(num_turns, max_replay_budget, replay_budget)


def select_replay_candidates(
    local_cost: list[float],
    suffix_priority: list[float],
    replay_budget: int,
    suffix_fraction: float | None,
) -> list[int]:
    """Select a suffix-heavy, de-duplicated mix of replay candidates."""
    if len(local_cost) != len(suffix_priority):
        raise ValueError("local_cost and suffix_priority must have the same length.")
    if replay_budget < 0:
        raise ValueError("replay_budget must be non-negative.")
    if suffix_fraction is not None and not 0.0 <= suffix_fraction <= 1.0:
        raise ValueError("recap_grpo.suffix_fraction must be in [0, 1].")

    budget = min(replay_budget, len(local_cost))
    if suffix_fraction is None:
        candidates: list[int] = []
        for ranking in (suffix_priority, local_cost):
            for turn_index in sorted(range(len(ranking)), key=ranking.__getitem__, reverse=True):
                if turn_index not in candidates:
                    candidates.append(turn_index)
                    break
        if len(candidates) < budget:
            for turn_index in sorted(
                range(len(suffix_priority)), key=suffix_priority.__getitem__, reverse=True
            ):
                if turn_index not in candidates:
                    candidates.append(turn_index)
                if len(candidates) >= budget:
                    break
        return candidates[:budget]

    suffix_budget = min(budget, math.ceil(suffix_fraction * budget))
    candidates = sorted(
        range(len(suffix_priority)), key=suffix_priority.__getitem__, reverse=True
    )[:suffix_budget]
    if len(candidates) < budget:
        for turn_index in sorted(range(len(local_cost)), key=local_cost.__getitem__, reverse=True):
            if turn_index not in candidates:
                candidates.append(turn_index)
            if len(candidates) >= budget:
                break
    return candidates


def _compute_trajectory_pte_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    pte_turns: np.ndarray,
    algo_config: Any,
    section: str,
) -> torch.Tensor:
    pte_config = algo_config.get(section, {})
    cost_coef = float(pte_config.get("cost_coef", 0.1))
    gamma = float(pte_config.get("gamma", 0.00704))
    if not 0.0 <= cost_coef < 1.0:
        raise ValueError(f"{section}.cost_coef must be in [0, 1).")
    if gamma < 0.0:
        raise ValueError(f"{section}.gamma must be non-negative.")

    scores = token_level_rewards.sum(dim=-1)
    costs = torch.zeros_like(scores, dtype=torch.float32)
    for row, turns in enumerate(pte_turns):
        for turn in (turns if turns is not None else []):
            prefill = float(turn["prefill_tokens"])
            decode = float(turn["decode_tokens"])
            if prefill < 0 or decode < 0:
                raise ValueError("PTE turn token counts must be non-negative.")
            costs[row] += prefill + gamma * prefill * decode

    by_prompt: dict[Any, list[int]] = defaultdict(list)
    for row, prompt_id in enumerate(index):
        by_prompt[prompt_id].append(row)

    advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    normalize_std = bool(algo_config.get("norm_adv_by_std_in_grpo", True))
    epsilon = float(pte_config.get("epsilon", 1e-6))
    with torch.no_grad():
        for rows in by_prompt.values():
            group_scores = scores[rows]
            group_costs = costs[rows]
            success = group_scores > 0
            relative_cost = torch.zeros_like(group_costs)
            if success.any():
                successful_costs = group_costs[success]
                minimum, maximum = successful_costs.min(), successful_costs.max()
                if maximum > minimum:
                    relative_cost[success] = (successful_costs - minimum) / (maximum - minimum + epsilon)
            utilities = group_scores * (1.0 - cost_coef * relative_cost)
            if len(rows) == 1:
                centered = torch.zeros_like(utilities)
            else:
                centered = utilities - utilities.mean()
                if normalize_std:
                    centered = centered / (utilities.std() + epsilon)
            for row, value in zip(rows, centered, strict=True):
                advantages[row] = value * response_mask[row]
    return advantages


@core_algos.register_adv_est("pte_grpo")
def compute_pte_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    pte_turns: np.ndarray,
    config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO advantage from a success-preserving PTE utility.

    Rollouts and the inference-time action space are exactly the same as GRPO.
    A successful trajectory receives a utility in ``[1 - cost_coef, 1]``;
    failed trajectories retain utility zero.  Thus, for ``cost_coef < 1``, a
    success always outranks a failure while cheaper successful trajectories get
    larger group-relative credit.
    """
    algo_config = config or {}
    advantages = _compute_trajectory_pte_advantage(
        token_level_rewards, response_mask, index, pte_turns, algo_config, "pte_grpo"
    )
    return advantages, advantages


@core_algos.register_adv_est("recap_grpo")
def compute_recap_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    recap_turns: np.ndarray,
    config: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine trajectory credit with action-deletion counterfactual credit."""
    algo_config = config or {}
    recap_config = algo_config.get("recap_grpo", {})
    local_weight = float(recap_config.get("local_weight", 0.1))
    local_credit_mode = str(recap_config.get("local_credit_mode", "binary_removable"))
    local_normalization = str(recap_config.get("local_normalization", "none"))
    epsilon = float(recap_config.get("epsilon", 1e-6))
    if local_weight < 0.0:
        raise ValueError("recap_grpo.local_weight must be non-negative.")
    if local_credit_mode not in {"binary_removable", "counterfactual_utility"}:
        raise ValueError(f"Unknown recap_grpo.local_credit_mode: {local_credit_mode}.")
    if local_normalization not in {"none", "signed_absmax"}:
        raise ValueError(f"Unknown recap_grpo.local_normalization: {local_normalization}.")

    macro = _compute_trajectory_pte_advantage(
        token_level_rewards, response_mask, index, recap_turns, algo_config, "recap_grpo"
    )
    local = torch.zeros_like(macro)
    width = response_mask.shape[1]
    with torch.no_grad():
        for row, turns in enumerate(recap_turns):
            row_turns = turns if turns is not None else []
            credits: list[tuple[dict[str, Any], float]] = []
            for turn in row_turns:
                if local_credit_mode == "binary_removable":
                    if bool(turn.get("removable", False)):
                        credits.append((turn, -1.0))
                    continue

                if not bool(turn.get("tested", False)) or "counterfactual_reward" not in turn:
                    continue
                original_reward = float(turn.get("original_reward", token_level_rewards[row].sum().item()))
                counterfactual_reward = float(turn["counterfactual_reward"])
                reward_gap = original_reward - counterfactual_reward
                if abs(reward_gap) > epsilon:
                    # Task success dominates compute: deleting a necessary action
                    # receives positive credit even if the shorter failed replay is cheap.
                    credit = reward_gap
                else:
                    original_pte = float(turn.get("original_pte", 0.0))
                    if original_pte <= 0.0:
                        raise ValueError("Tested RECAP turns require a positive original_pte.")
                    # If both trajectories have the same outcome, retaining a turn
                    # is penalized in proportion to the compute its deletion saves.
                    credit = -float(turn.get("pte_saving", 0.0)) / (original_pte + epsilon)
                credits.append((turn, credit))

            if local_normalization == "signed_absmax" and credits:
                positive_scale = max((value for _, value in credits if value > 0.0), default=0.0)
                negative_scale = max((-value for _, value in credits if value < 0.0), default=0.0)
                credits = [
                    (
                        turn,
                        value / positive_scale
                        if value > 0.0 and positive_scale > 0.0
                        else value / negative_scale
                        if value < 0.0 and negative_scale > 0.0
                        else 0.0,
                    )
                    for turn, value in credits
                ]

            for turn, credit in credits:
                start, end = int(turn["start"]), int(turn["end"])
                if not (0 <= start < end <= width):
                    raise ValueError(f"Invalid RECAP response span [{start}, {end}) for width {width}.")
                local[row, start:end] = credit
    advantages = (macro + local_weight * local) * response_mask
    return advantages, advantages
