import numpy as np
import torch

from verl.experimental.gigpo.core import compute_gigpo_advantage
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_gigpo_zero_weight_matches_standard_grpo():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    index = np.array(["task", "task"], dtype=object)
    turns = np.array(
        [
            [{"anchor": "a", "reward": 0.0, "start": 0, "end": 2}],
            [{"anchor": "a", "reward": 1.0, "start": 0, "end": 2}],
        ],
        dtype=object,
    )

    expected, _ = compute_grpo_outcome_advantage(rewards, mask, index)
    actual, _ = compute_gigpo_advantage(
        rewards,
        mask,
        index,
        turns,
        config={"gigpo": {"step_advantage_w": 0.0}},
    )
    torch.testing.assert_close(actual, expected)


def test_gigpo_only_updates_generated_action_span():
    rewards = torch.zeros((2, 5))
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 1, 0, 1, 0]])
    turns = np.array(
        [
            [{"anchor": "same", "reward": 0.0, "start": 0, "end": 2}],
            [{"anchor": "same", "reward": 1.0, "start": 0, "end": 2}],
        ],
        dtype=object,
    )
    advantages, _ = compute_gigpo_advantage(
        rewards,
        mask,
        np.array([1, 1]),
        turns,
        config={"gigpo": {"step_advantage_w": 1.0}},
    )
    assert torch.all(advantages[:, 2:] == 0)
    assert torch.all(advantages[:, :2].abs().sum(dim=1) > 0)


def test_gigpo_similarity_anchor_grouping_matches_exact_anchor_grouping():
    rewards = torch.zeros((2, 2))
    mask = torch.ones((2, 2), dtype=torch.long)
    turns = np.array(
        [
            [{"anchor": "cabinet 1 is open", "reward": 0.0, "start": 0, "end": 2}],
            [{"anchor": "cabinet 1 is opened", "reward": 1.0, "start": 0, "end": 2}],
        ],
        dtype=object,
    )
    no_similarity, _ = compute_gigpo_advantage(
        rewards, mask, np.array([1, 1]), turns, config={"gigpo": {"mode": "mean_norm"}}
    )
    similarity, _ = compute_gigpo_advantage(
        rewards,
        mask,
        np.array([1, 1]),
        turns,
        config={"gigpo": {"mode": "mean_norm", "enable_similarity": True, "similarity_thresh": 0.9}},
    )
    assert torch.all(no_similarity == 0)
    assert torch.all(similarity.abs().sum(dim=1) > 0)
