import numpy as np
import torch

from verl.experimental.pte_grpo.core import (
    compute_pte_grpo_advantage,
    compute_recap_grpo_advantage,
    proportional_replay_budget,
    select_replay_candidates,
)


def test_recap_proportional_budget_is_capped_and_rounded_up():
    assert proportional_replay_budget(4, 0.4, 6) == 2
    assert proportional_replay_budget(8, 0.4, 6) == 4
    assert proportional_replay_budget(20, 0.4, 6) == 6
    assert proportional_replay_budget(0, 0.4, 6) == 0


def test_recap_proportional_budget_supports_nearest_integer_rounding():
    assert proportional_replay_budget(1, 0.2, 4, "nearest") == 1
    assert proportional_replay_budget(4, 0.2, 4, "nearest") == 1
    assert proportional_replay_budget(7, 0.2, 4, "nearest") == 1
    assert proportional_replay_budget(8, 0.2, 4, "nearest") == 2
    assert proportional_replay_budget(20, 0.2, 4, "nearest") == 4


def test_recap_candidate_selection_uses_suffix_quota_then_unique_local_turns():
    local = [1.0, 9.0, 8.0, 2.0, 7.0]
    suffix = [10.0, 9.0, 1.0, 8.0, 2.0]
    # Three suffix slots select 0, 1, 3. Local's top turn 1 is already
    # selected, so turn 2 is used to fill the remaining local slot.
    assert select_replay_candidates(local, suffix, replay_budget=4, suffix_fraction=0.75) == [0, 1, 3, 2]


def test_recap_two_turn_budget_is_suffix_only_at_three_quarter_fraction():
    local = [1.0, 10.0, 9.0]
    suffix = [10.0, 1.0, 9.0]
    assert select_replay_candidates(local, suffix, replay_budget=2, suffix_fraction=0.75) == [0, 2]


def test_recap_candidate_selection_preserves_legacy_mixed_order_without_fraction():
    local = [1.0, 10.0, 9.0, 8.0]
    suffix = [10.0, 1.0, 9.0, 8.0]
    assert select_replay_candidates(local, suffix, replay_budget=4, suffix_fraction=None) == [0, 1, 2, 3]


def test_pte_grpo_prefers_cheaper_success_within_group():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    mask = torch.ones_like(rewards)
    turns = np.array(
        [
            [{"prefill_tokens": 100, "decode_tokens": 10}],
            [{"prefill_tokens": 10, "decode_tokens": 1}],
            [{"prefill_tokens": 1, "decode_tokens": 1}],
        ],
        dtype=object,
    )
    advantages, _ = compute_pte_grpo_advantage(
        rewards, mask, np.array(["task"] * 3, dtype=object), turns,
        config={"pte_grpo": {"cost_coef": 0.5, "gamma": 0.01}, "norm_adv_by_std_in_grpo": False},
    )
    assert advantages[1, 0] > advantages[0, 0] > advantages[2, 0]


def test_pte_grpo_zero_cost_weight_matches_grpo_centering():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    mask = torch.ones_like(rewards)
    turns = np.array([[], []], dtype=object)
    advantages, _ = compute_pte_grpo_advantage(
        rewards, mask, np.array([1, 1]), turns,
        config={"pte_grpo": {"cost_coef": 0.0}, "norm_adv_by_std_in_grpo": False},
    )
    torch.testing.assert_close(advantages[:, 0], torch.tensor([0.5, -0.5]))


def test_recap_penalty_is_scattered_only_to_removable_turn():
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    mask = torch.ones_like(rewards)
    turns = np.array(
        [
            [
                {"prefill_tokens": 10, "decode_tokens": 2, "start": 0, "end": 2, "removable": True},
                {"prefill_tokens": 20, "decode_tokens": 2, "start": 2, "end": 4, "removable": False},
            ],
            [
                {"prefill_tokens": 10, "decode_tokens": 2, "start": 0, "end": 2, "removable": False},
                {"prefill_tokens": 20, "decode_tokens": 2, "start": 2, "end": 4, "removable": False},
            ],
        ],
        dtype=object,
    )
    advantages, _ = compute_recap_grpo_advantage(
        rewards,
        mask,
        np.array(["task", "task"], dtype=object),
        turns,
        config={
            "recap_grpo": {"cost_coef": 0.0, "local_weight": 0.2},
            "norm_adv_by_std_in_grpo": False,
        },
    )
    torch.testing.assert_close(advantages[0], torch.tensor([-0.2, -0.2, 0.0, 0.0]))
    torch.testing.assert_close(advantages[1], torch.zeros(4))


def test_recap_counterfactual_utility_uses_pte_magnitude_and_necessary_credit():
    rewards = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    mask = torch.ones_like(rewards)
    turns = np.array(
        [
            [
                {
                    "prefill_tokens": 10,
                    "decode_tokens": 2,
                    "start": 0,
                    "end": 2,
                    "tested": True,
                    "original_reward": 1.0,
                    "counterfactual_reward": 1.0,
                    "original_pte": 100.0,
                    "pte_saving": 10.0,
                },
                {
                    "prefill_tokens": 20,
                    "decode_tokens": 2,
                    "start": 2,
                    "end": 4,
                    "tested": True,
                    "original_reward": 1.0,
                    "counterfactual_reward": 1.0,
                    "original_pte": 100.0,
                    "pte_saving": 20.0,
                },
                {
                    "prefill_tokens": 30,
                    "decode_tokens": 2,
                    "start": 4,
                    "end": 6,
                    "tested": True,
                    "original_reward": 1.0,
                    "counterfactual_reward": 0.0,
                    "original_pte": 100.0,
                    "pte_saving": 40.0,
                },
            ]
        ],
        dtype=object,
    )
    advantages, _ = compute_recap_grpo_advantage(
        rewards,
        mask,
        np.array(["task"], dtype=object),
        turns,
        config={
            "recap_grpo": {
                "cost_coef": 0.0,
                "local_weight": 0.2,
                "local_credit_mode": "counterfactual_utility",
                "local_normalization": "signed_absmax",
            },
            "norm_adv_by_std_in_grpo": False,
        },
    )
    torch.testing.assert_close(advantages[0], torch.tensor([-0.1, -0.1, -0.2, -0.2, 0.2, 0.2]))
