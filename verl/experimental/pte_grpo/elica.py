"""Training-only ELICA prediction head and turn-credit helpers."""

from __future__ import annotations

import torch
from torch import nn


class ELICATurnUtilityHead(nn.Module):
    """Predict trajectory utility from pooled turn states."""

    def __init__(self, hidden_size: int, width: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.gate = nn.Linear(width, 1)
        self.value = nn.Linear(width, 1)

    def forward(self, hidden: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        # Actor hidden states can be bf16 even when the small head is kept in
        # fp32 for stable regression.  Keep the auxiliary computation in one
        # dtype instead of relying on the actor's autocast context.
        encoded = self.encoder(hidden.float())
        logits = self.gate(encoded).squeeze(-1).masked_fill(~turn_mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        values = self.value(encoded).squeeze(-1)
        return (weights * values).sum(dim=-1)


def pool_turn_hidden(
    response_hidden: torch.Tensor,
    pte_turns: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool response hidden states according to rollout turn spans."""
    batch, response_length, hidden_size = response_hidden.shape
    max_turns = max((len(turns or []) for turns in pte_turns), default=0)
    turn_hidden = response_hidden.new_zeros((batch, max_turns, hidden_size))
    turn_mask = torch.zeros((batch, max_turns), dtype=torch.bool, device=response_hidden.device)
    token_turn_index = torch.full((batch, response_length), -1, dtype=torch.long, device=response_hidden.device)
    for row, turns in enumerate(pte_turns):
        cursor = 0
        for turn_index, turn in enumerate(turns or []):
            start = int(turn.get("response_start", cursor))
            end = int(turn.get("response_end", start + int(turn.get("decode_tokens", 0))))
            cursor = end
            start = max(0, min(start, response_length))
            end = max(start, min(end, response_length))
            if end <= start or turn_index >= max_turns:
                continue
            turn_hidden[row, turn_index] = response_hidden[row, start:end].mean(dim=0)
            turn_mask[row, turn_index] = True
            token_turn_index[row, start:end] = turn_index
    return turn_hidden, turn_mask, token_turn_index


@torch.no_grad()
def latent_intervention_credit(
    head: ELICATurnUtilityHead, turn_hidden: torch.Tensor, turn_mask: torch.Tensor
) -> torch.Tensor:
    """Return centered leave-one-turn-out utility changes."""
    if turn_hidden.shape[1] == 0:
        return turn_hidden.new_zeros(turn_mask.shape, dtype=turn_hidden.dtype)
    full = head(turn_hidden, turn_mask)
    marginal = []
    for turn in range(turn_hidden.shape[1]):
        variant = turn_hidden.clone()
        variant[:, turn] = 0.0
        mask = turn_mask.clone()
        present = mask[:, turn]
        mask[:, turn] = False
        masked = head(variant, mask)
        marginal.append(torch.where(present, full - masked, torch.zeros_like(full)))
    values = torch.stack(marginal, dim=1).masked_fill(~turn_mask, 0.0)
    centered = values - values.sum(dim=1, keepdim=True) / turn_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return centered.masked_fill(~turn_mask, 0.0)


def map_turn_credit_to_tokens(local_credit: torch.Tensor, token_turn_index: torch.Tensor) -> torch.Tensor:
    """Map turn credit back to response tokens, tolerating turn-less rollouts.

    A rollout without a parsed valid action has no local signal.  It still
    receives its trajectory-level PTE advantage, so its token-level local
    credit must be exactly zero rather than indexing a nonexistent turn.
    """
    if local_credit.shape[1] == 0:
        return local_credit.new_zeros(token_turn_index.shape)
    safe_turn_index = token_turn_index.clamp_min(0)
    token_credit = torch.gather(local_credit, 1, safe_turn_index)
    return token_credit.masked_fill(token_turn_index < 0, 0.0)
