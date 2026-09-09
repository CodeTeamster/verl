#!/usr/bin/env python3
"""Compare ALFWorld multi-turn trajectories with Prefill Token Equivalents.

PTE follows Su et al. (ACL 2026):
    sum_i(prefill_i + gamma * context_i * decode_i)

ALFWorld inserts an environment observation between every model request.  We
therefore conservatively treat every turn as a cache miss and count its entire
context as prefill.  The script reconstructs the rollout token stream using
the same chat template as the agent loop.  Its ``decode`` count excludes a
possible terminal special token, so PTE is a slightly conservative proxy, but
both compared logs use the identical approximation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from transformers import AutoConfig, AutoTokenizer


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
FAILURE_MARKERS = (
    "nothing happens",
    "you can't",
    "you cannot",
    "don't see",
    "do not see",
    "not holding",
    "is closed",
    "no known action",
)


@dataclass
class TrajectoryCost:
    score: float
    pte: float
    prefills: list[int]
    decodes: list[int]
    per_turn_pte: list[float]
    turns: int
    task: str


def parse_transcript(row: dict) -> tuple[str, list[tuple[str, str]]]:
    """Return initial observation and (assistant, next-observation) turns."""
    initial = row["input"].split("\nuser\n", 1)[1].rsplit("\nassistant\n", 1)[0]
    chunks = row["output"].split("user\n")
    turns: list[tuple[str, str]] = []
    for index, chunk in enumerate(chunks):
        assistant = chunk if index == 0 else chunk.split("\nassistant\n", 1)[1]
        if index + 1 < len(chunks):
            observation = chunks[index + 1].split("\nassistant\n", 1)[0]
        else:
            observation = ""
        if assistant:
            turns.append((assistant, observation))
    return initial, turns


def normalize_token_ids(token_ids: list[int] | list[list[int]]) -> list[int]:
    return token_ids[0] if token_ids and isinstance(token_ids[0], list) else token_ids


def system_prompt_ids(tokenizer) -> list[int]:
    one = normalize_token_ids(tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False))
    two = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False)
    )
    return one[: -(len(two) - len(one))]


def compute_gamma(config, active_params: float, hoi: float) -> float:
    gamma = 2 * config.num_hidden_layers * config.hidden_size * hoi / active_params
    gamma *= config.num_key_value_heads / config.num_attention_heads
    return gamma


def cost_row(row: dict, tokenizer, system_prompt_ids: list[int], gamma: float, max_prompt: int) -> TrajectoryCost:
    initial, turns = parse_transcript(row)
    system = row["input"].split("system\n", 1)[1].split("\n\nuser\n", 1)[0]
    context_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": initial}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )[:max_prompt]
    prefills, decodes = [], []
    for assistant, observation in turns:
        prefills.append(len(context_ids))
        generated = tokenizer.encode(assistant, add_special_tokens=False)
        decodes.append(len(generated))
        context_ids += generated
        if observation:
            observation_ids = normalize_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": observation}], add_generation_prompt=True, tokenize=True
                )
            )[len(system_prompt_ids) :]
            context_ids += observation_ids
    per_turn_pte = [prefill + gamma * prefill * decode for prefill, decode in zip(prefills, decodes, strict=True)]
    pte = sum(per_turn_pte)
    task = row["input"].split("Your task is to: ", 1)[1].split("\n", 1)[0]
    return TrajectoryCost(float(row["score"]), pte, prefills, decodes, per_turn_pte, len(turns), task)


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def actions_and_observations(row: dict) -> tuple[list[str], list[str]]:
    _, turns = parse_transcript(row)
    actions, observations = [], []
    for assistant, observation in turns:
        match = ACTION_RE.search(assistant)
        actions.append(match.group(1).strip().lower() if match else "<invalid-format>")
        observations.append(observation.strip().lower())
    return actions, observations


def redundancy(actions: list[str], observations: list[str]) -> dict[str, int]:
    return {
        "immediate_repeats": sum(left == right for left, right in zip(actions, actions[1:])),
        "repeated_actions": len(actions) - len(set(actions)),
        "failure_observations": sum(
            any(marker in observation for marker in FAILURE_MARKERS) for observation in observations
        ),
    }


def summarize(costs: list[TrajectoryCost]) -> dict:
    successful = [item for item in costs if item.score == 1]
    return {
        "n": len(costs),
        "success": len(successful),
        "success_rate": len(successful) / len(costs),
        "pte_mean": mean(item.pte for item in costs),
        "pte_median": median(item.pte for item in costs),
        "pte_success_mean": mean(item.pte for item in successful),
        "turns_mean": mean(item.turns for item in costs),
        "decoded_tokens_mean": mean(sum(item.decodes) for item in costs),
        "successful_last_quarter_pte_share": mean(
            sum(item.per_turn_pte[-math.ceil(item.turns / 4) :]) / item.pte
            for item in successful
            if item.turns >= 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs=2, type=Path, help="Aligned baseline and candidate validation JSONL files")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--labels", nargs=2, default=["baseline", "candidate"])
    parser.add_argument("--hoi", type=float, default=756.5, help="H100 FP16/BF16 hardware operational intensity")
    parser.add_argument("--active-params", type=float, default=1.54e9)
    parser.add_argument("--max-prompt", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = AutoConfig.from_pretrained(args.tokenizer)
    gamma = compute_gamma(config, args.active_params, args.hoi)
    template_system_ids = system_prompt_ids(tokenizer)
    row_groups = [read_rows(path) for path in args.logs]
    groups = [
        [cost_row(row, tokenizer, template_system_ids, gamma, args.max_prompt) for row in rows]
        for rows in row_groups
    ]
    if len(groups[0]) != len(groups[1]):
        raise ValueError("The two logs must contain the same number of aligned tasks.")
    if [item.task for item in groups[0]] != [item.task for item in groups[1]]:
        raise ValueError("Task order differs; pair the logs before analysis.")

    paired = list(zip(row_groups[0], row_groups[1], groups[0], groups[1], strict=True))
    both_success = [pair for pair in paired if pair[2].score == pair[3].score == 1]
    cost_pairs = [(pair[2], pair[3]) for pair in paired]
    both_success_costs = [(pair[2], pair[3]) for pair in both_success]
    best_success_pte = [min(left.pte, right.pte) for left, right in cost_pairs if left.score or right.score]
    selected_pte = [right.pte if right.score else left.pte for left, right in cost_pairs if left.score or right.score]

    divergences = []
    redundant = {label: Counter() for label in args.labels}
    action_counts = {label: Counter() for label in args.labels}
    for baseline_row, candidate_row, baseline_cost, candidate_cost in both_success:
        baseline_actions, baseline_obs = actions_and_observations(baseline_row)
        candidate_actions, candidate_obs = actions_and_observations(candidate_row)
        first = next(
            (i for i, values in enumerate(zip(baseline_actions, candidate_actions)) if values[0] != values[1]),
            min(len(baseline_actions), len(candidate_actions)),
        )
        divergences.append(
            {
                "common_prefix_turns": first,
                "baseline_tail_pte": sum(baseline_cost.per_turn_pte[first:]),
                "candidate_tail_pte": sum(candidate_cost.per_turn_pte[first:]),
                "baseline_tail_turns": len(baseline_actions) - first,
                "candidate_tail_turns": len(candidate_actions) - first,
            }
        )
        for label, actions, observations in (
            (args.labels[0], baseline_actions, baseline_obs),
            (args.labels[1], candidate_actions, candidate_obs),
        ):
            redundant[label].update(redundancy(actions, observations))
            action_counts[label].update(action.split(" ", 1)[0] for action in actions)
    report = {
        "metric": "PTE = sum(prefill_i + gamma * context_i * decode_i); cache miss after every environment step",
        "gamma": gamma,
        "groups": {label: summarize(group) for label, group in zip(args.labels, groups, strict=True)},
        "paired": {
            "n": len(paired),
            "candidate_lower_pte": sum(right.pte < left.pte for left, right in cost_pairs),
            "equal_pte": sum(math.isclose(right.pte, left.pte) for left, right in cost_pairs),
            "mean_pte_delta_baseline_minus_candidate": mean(left.pte - right.pte for left, right in cost_pairs),
            "both_success_n": len(both_success),
            "both_success_mean_pte_delta_baseline_minus_candidate": mean(
                left.pte - right.pte for left, right in both_success_costs
            ),
            "both_success_pte_reduction_fraction": 1
            - mean(right.pte for _, right in both_success_costs)
            / mean(left.pte for left, _ in both_success_costs),
            "oracle_best_success_pte_mean": mean(best_success_pte),
            "candidate_selected_success_pte_mean": mean(selected_pte),
            "candidate_headroom_to_per_task_best_success": mean(selected_pte) - mean(best_success_pte),
        },
        "divergence": {
            "different_action_sequence": sum(
                actions_and_observations(pair[0])[0] != actions_and_observations(pair[1])[0]
                for pair in both_success
            ),
            "common_prefix_turns_mean": mean(item["common_prefix_turns"] for item in divergences),
            "baseline_tail_turns_mean": mean(item["baseline_tail_turns"] for item in divergences),
            "candidate_tail_turns_mean": mean(item["candidate_tail_turns"] for item in divergences),
            "baseline_tail_pte_mean": mean(item["baseline_tail_pte"] for item in divergences),
            "candidate_tail_pte_mean": mean(item["candidate_tail_pte"] for item in divergences),
        },
        "redundancy_per_success": {
            label: {key: value / len(both_success) for key, value in counts.items()}
            for label, counts in redundant.items()
        },
        "action_type_per_success": {
            label: {key: value / len(both_success) for key, value in counts.items()}
            for label, counts in action_counts.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
