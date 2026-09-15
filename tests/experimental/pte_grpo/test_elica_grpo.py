import torch

from verl.experimental.pte_grpo.elica import map_turn_credit_to_tokens, pool_turn_hidden


def test_turnless_rollout_maps_to_zero_token_credit():
    local_credit = torch.empty((1, 0), dtype=torch.float32)
    token_turn_index = torch.full((1, 8), -1, dtype=torch.long)

    token_credit = map_turn_credit_to_tokens(local_credit, token_turn_index)

    assert token_credit.shape == (1, 8)
    assert torch.count_nonzero(token_credit) == 0


def test_mixed_rollouts_keep_turnless_tokens_at_zero():
    response_hidden = torch.randn(2, 6, 4)
    turn_hidden, turn_mask, token_turn_index = pool_turn_hidden(
        response_hidden,
        [
            [{"response_start": 1, "response_end": 3}],
            [],
        ],
    )
    local_credit = torch.tensor([[2.0], [0.0]], dtype=torch.float32)

    token_credit = map_turn_credit_to_tokens(local_credit, token_turn_index)

    assert turn_mask.tolist() == [[True], [False]]
    assert token_credit.tolist() == [[0.0, 2.0, 2.0, 0.0, 0.0, 0.0], [0.0] * 6]
