#!/usr/bin/env python3
"""Validate success-preserving turn deletion on logged ALFWorld trajectories.

No model inference is used.  The script resets the deterministic environment,
replays logged actions, and intervenes by deleting selected actions.  It first
requires the unmodified action list to reproduce the logged success.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_alfworld_pte import (  # noqa: E402
    compute_gamma,
    cost_row,
    normalize_token_ids,
    parse_transcript,
    system_prompt_ids,
)
from verl.experimental.environments.alfworld import _ALFWorldEnvironmentSlot  # noqa: E402


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_observation(observation: str) -> str:
    if "Nothing happens" in observation:
        observation += " Please check if the action you take is valid or you have carefully followed the action format."
    return "Observation: " + observation


def get_turns(row: dict) -> list[tuple[str, str]]:
    _, parsed = parse_transcript(row)
    turns = []
    for assistant, _ in parsed:
        match = ACTION_RE.search(assistant)
        if match is None:
            return []
        turns.append((assistant, match.group(1).strip()))
    return turns


def replay(
    slot: _ALFWorldEnvironmentSlot,
    game_file: Path,
    row: dict,
    turns: list[tuple[str, str]],
    deleted: set[int],
    tokenizer,
    template_system_ids: list[int],
    gamma: float,
    max_prompt: int,
) -> tuple[float, float]:
    reset = slot.reset(game_file)
    system = row["input"].split("system\n", 1)[1].split("\n\nuser\n", 1)[0]
    context_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": format_observation(reset.observation)},
            ],
            add_generation_prompt=True,
            tokenize=True,
        )
    )[:max_prompt]
    total_pte = 0.0
    for index, (assistant, action) in enumerate(turns):
        if index in deleted:
            continue
        generated = tokenizer.encode(assistant, add_special_tokens=False)
        prefill = len(context_ids)
        total_pte += prefill + gamma * prefill * len(generated)
        context_ids += generated
        result = slot.step(action)
        if result.done:
            return result.reward, total_pte
        observation_ids = normalize_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": format_observation(result.observation)}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )[len(template_system_ids) :]
        context_ids += observation_ids
    return 0.0, total_pte


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-successes", type=int, default=32)
    parser.add_argument("--max-deletions-per-task", type=int, default=8)
    parser.add_argument("--hoi", type=float, default=756.5)
    parser.add_argument("--active-params", type=float, default=1.54e9)
    parser.add_argument("--max-prompt", type=int, default=1024)
    args = parser.parse_args()

    os.environ["ALFWORLD_DATA"] = str(args.data_root.resolve())
    rows = read_jsonl(args.log)
    tasks = [value.as_py() for value in pq.read_table(args.dataset, columns=["extra_info"])["extra_info"]]
    if len(rows) != len(tasks):
        raise ValueError(f"Log/dataset size mismatch: {len(rows)} != {len(tasks)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = AutoConfig.from_pretrained(args.tokenizer)
    gamma = compute_gamma(config, args.active_params, args.hoi)
    template_ids = system_prompt_ids(tokenizer)
    slot = _ALFWorldEnvironmentSlot(max_episode_steps=50, domain_randomization=False)
    replayable = []
    mismatches = []
    try:
        for row_index, (row, task) in enumerate(zip(rows, tasks, strict=True)):
            if float(row["score"]) <= 0 or len(replayable) >= args.max_successes:
                continue
            turns = get_turns(row)
            if not turns:
                mismatches.append({"row": row_index, "reason": "invalid_action_format"})
                continue
            game_file = args.data_root / task["game_file"]
            reward, replay_pte = replay(
                slot, game_file, row, turns, set(), tokenizer, template_ids, gamma, args.max_prompt
            )
            logged_pte = cost_row(row, tokenizer, template_ids, gamma, args.max_prompt).pte
            if reward <= 0:
                mismatches.append({"row": row_index, "reason": "logged_success_did_not_replay"})
                continue
            replayable.append((row_index, row, game_file, turns, replay_pte, logged_pte))

        task_reports = []
        for row_index, row, game_file, turns, original_pte, logged_pte in replayable:
            original_cost = cost_row(row, tokenizer, template_ids, gamma, args.max_prompt)
            candidates = sorted(range(len(turns)), key=lambda index: original_cost.per_turn_pte[index], reverse=True)
            candidates = candidates[: args.max_deletions_per_task]
            removable = []
            single_pte_savings = []
            for turn_index in candidates:
                reward, counterfactual_pte = replay(
                    slot,
                    game_file,
                    row,
                    turns,
                    {turn_index},
                    tokenizer,
                    template_ids,
                    gamma,
                    args.max_prompt,
                )
                if reward > 0:
                    removable.append(turn_index)
                    single_pte_savings.append(original_pte - counterfactual_pte)

            deleted: set[int] = set()
            greedy_pte = original_pte
            for turn_index in candidates:
                reward, counterfactual_pte = replay(
                    slot,
                    game_file,
                    row,
                    turns,
                    deleted | {turn_index},
                    tokenizer,
                    template_ids,
                    gamma,
                    args.max_prompt,
                )
                if reward > 0:
                    deleted.add(turn_index)
                    greedy_pte = counterfactual_pte
            task_reports.append(
                {
                    "row": row_index,
                    "turns": len(turns),
                    "tested_turns": len(candidates),
                    "single_removable_turns": len(removable),
                    "single_removable_indices": removable,
                    "single_pte_savings": single_pte_savings,
                    "greedy_removed_turns": len(deleted),
                    "greedy_removed_indices": sorted(deleted),
                    "original_replay_pte": original_pte,
                    "logged_pte": logged_pte,
                    "greedy_pte": greedy_pte,
                }
            )
    finally:
        slot.close()

    tested = sum(item["tested_turns"] for item in task_reports)
    removable = sum(item["single_removable_turns"] for item in task_reports)
    total_turns = sum(item["turns"] for item in task_reports)
    greedy_removed = sum(item["greedy_removed_turns"] for item in task_reports)
    report = {
        "source": str(args.log),
        "logged_successes": sum(float(row["score"]) > 0 for row in rows),
        "requested_successes": args.max_successes,
        "replayable_successes": len(replayable),
        "replay_mismatches": mismatches,
        "single_deletion": {
            "tested_turns": tested,
            "success_preserving_deletions": removable,
            "success_preserving_fraction": removable / tested if tested else 0.0,
            "tasks_with_removable_turn": sum(item["single_removable_turns"] > 0 for item in task_reports),
            "pte_savings_mean": mean(
                saving for item in task_reports for saving in item["single_pte_savings"]
            )
            if removable
            else 0.0,
        },
        "greedy_deletion": {
            "total_turns": total_turns,
            "removed_turns": greedy_removed,
            "removed_fraction": greedy_removed / total_turns if total_turns else 0.0,
            "tasks_with_removed_turn": sum(item["greedy_removed_turns"] > 0 for item in task_reports),
            "mean_pte_reduction_fraction": mean(
                1.0 - item["greedy_pte"] / item["original_replay_pte"] for item in task_reports
            )
            if task_reports
            else 0.0,
        },
        "tasks": task_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
