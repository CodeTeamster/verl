#!/usr/bin/env python3
"""Convert the raw ALFWorld files into verl-compatible Parquet files.

The generated rows contain the task prompt and paths to the raw environment
files.  The walkthrough is deliberately not included in the prompt because it
is an expert solution and would leak the answer during rollout.

Example:

    python lab/alfworld_preprocess.py \
        --data_root /nfs-medical1-NB/yrc/datasets/alfworld \
        --output_dir assets/datasets/alfworld
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets/datasets/alfworld"
DEFAULT_SPLITS = ("train", "valid_train", "valid_seen", "valid_unseen")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    return value


def _make_row(split: str, data_root: Path, game_file: Path, index: int) -> dict[str, Any]:
    task_root = game_file.parent
    game = _read_json(game_file)
    traj_file = task_root / "traj_data.json"
    initial_state_file = task_root / "initial_state.pddl"
    traj = _read_json(traj_file) if traj_file.exists() else {}

    task_id = traj.get("task_id") or task_root.parent.name
    task_type = traj.get("task_type") or task_root.parent.name.split("-", maxsplit=1)[0]
    game_relpath = game_file.relative_to(data_root)
    initial_state_relpath = initial_state_file.relative_to(data_root)
    traj_relpath = traj_file.relative_to(data_root)

    return {
        "target": "",
        "agent_name": "alfworld_agent",
        "prompt": [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
        ],
        "data_source": "alfworld",
        "ability": "alfworld",
        # The actual success reward must be computed by the ALFWorld
        # environment/AgentLoop, not read as a supervised answer here.
        "reward_model": {"style": "rule", "ground_truth": 1},
        "extra_info": {
            "index": index,
            "task_id": str(task_id),
            "task_type": str(task_type),
            "split": split,
            "solvable": bool(game.get("solvable", False)),
            "data_root_env": "ALFWORLD_DATA",
            "game_file": str(game_relpath),
            "initial_state_file": str(initial_state_relpath),
            "traj_data_file": str(traj_relpath),
        },
    }


def _build_split(data_root: Path, split: str, max_tasks: int | None = None) -> list[dict[str, Any]]:
    split_root = data_root / "json_2.1.1" / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"ALFWorld split directory does not exist: {split_root}")

    game_files = sorted(split_root.glob("**/game.tw-pddl"))
    if max_tasks is not None:
        game_files = game_files[:max_tasks]

    rows = []
    for index, game_file in enumerate(game_files):
        rows.append(_make_row(split, data_root, game_file.resolve(), index))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default=None,
        help="Raw ALFWorld directory containing json_2.1.1/. Defaults to ALFWORLD_DATA.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory in which split Parquet files are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=DEFAULT_SPLITS,
        help="Splits to process (default: all available splits).",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Only process the first N tasks per split; useful for a smoke test.",
    )
    args = parser.parse_args()

    if args.max_tasks is not None and args.max_tasks <= 0:
        parser.error("--max_tasks must be positive")

    data_root_arg = args.data_root or os.environ.get("ALFWORLD_DATA")
    if not data_root_arg:
        parser.error("Specify --data_root or set the ALFWORLD_DATA environment variable")

    data_root = Path(data_root_arg).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in args.splits:
        rows = _build_split(data_root, split, args.max_tasks)
        output_file = output_dir / f"{split}.parquet"
        Dataset.from_list(rows).to_parquet(str(output_file))
        total += len(rows)
        print(f"{split}: {len(rows)} tasks -> {output_file}")

    print(f"Generated {total} ALFWorld tasks in {output_dir}")


if __name__ == "__main__":
    main()
