#!/usr/bin/env python3
"""Export complete action-deletion trajectories with per-turn annotations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_alfworld_pte import cost_row, parse_transcript, compute_gamma, system_prompt_ids  # noqa: E402


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-prompt", type=int, default=1024)
    parser.add_argument("--hoi", type=float, default=756.5)
    parser.add_argument("--active-params", type=float, default=1.54e9)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.log.open() if line.strip()]
    tasks = [v.as_py() for v in pq.read_table(args.dataset, columns=["extra_info"])["extra_info"]]
    report = json.loads(args.result.read_text())
    by_row = {item["row"]: item for item in report["tasks"]}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = AutoConfig.from_pretrained(args.tokenizer)
    gamma = compute_gamma(config, args.active_params, args.hoi)
    template_ids = system_prompt_ids(tokenizer)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    markdown_blocks = []
    with args.output_jsonl.open("w") as out:
        for row_index, item in by_row.items():
            row = rows[row_index]
            _, parsed = parse_transcript(row)
            assistants = [assistant for assistant, _ in parsed]
            actions = []
            for assistant in assistants:
                match = ACTION_RE.search(assistant)
                actions.append(match.group(1).strip() if match else "<invalid action>")
            costs = cost_row(row, tokenizer, template_ids, gamma, args.max_prompt).per_turn_pte
            # The top-8 experiment tested the highest local-PTE turns; the all-turns
            # experiment tested every turn.  The result JSON stores the successful
            # indices, while this exporter reconstructs the candidate set.
            if item["tested_turns"] < item["turns"]:
                tested = set(sorted(range(len(costs)), key=lambda i: costs[i], reverse=True)[: item["tested_turns"]])
            else:
                tested = set(range(len(costs)))
            removable = set(item["single_removable_indices"])
            greedy = set(item["greedy_removed_indices"])
            annotations = []
            for index, (assistant, action) in enumerate(zip(assistants, actions, strict=True)):
                annotations.append(
                    {
                        "turn": index,
                        "action": action,
                        "local_pte": costs[index] if index < len(costs) else None,
                        "tested": index in tested,
                        "single_deletion_success_preserving": index in removable,
                        "greedy_deleted": index in greedy,
                        "assistant_text": assistant,
                    }
                )
            exported = {
                "row": row_index,
                "task": tasks[row_index],
                "score": row.get("score"),
                "initial_game_context": row.get("input", "").rsplit("\nuser\n", 1)[-1].rsplit("\nassistant\n", 1)[0],
                "turns": annotations,
                "complete_original_trajectory": row.get("output", ""),
            }
            out.write(json.dumps(exported, ensure_ascii=False) + "\n")
            if removable and not markdown_blocks:
                lines = [
                    f"### row {row_index}（{len(annotations)} turns）",
                    "",
                    "**初始进入游戏的 observation（来自 input）**",
                    "```text",
                    exported["initial_game_context"],
                    "```",
                    "",
                    "标记：`[可删除]` = 单 action 删除后仍成功；`[贪心删除]` = 贪心过程最终删除；"
                    "`[未测试]` = top-8 预算没有测试；其余为测试但删除会破坏成功。",
                    "",
                ]
                for ann in annotations:
                    labels = []
                    if ann["single_deletion_success_preserving"]:
                        labels.append("可删除")
                    if ann["greedy_deleted"]:
                        labels.append("贪心删除")
                    if not ann["tested"]:
                        labels.append("未测试")
                    label = ", ".join(labels) if labels else "保留"
                    lines.append(f"**turn {ann['turn']} [{label}] local-PTE={ann['local_pte']:.1f}**")
                    lines.append("```text")
                    lines.append(ann["assistant_text"])
                    lines.append("```")
                    lines.append("")
                markdown_blocks.append("\n".join(lines))

    args.output_md.write_text(
        "# Action-deletion annotated trajectories\n\n"
        f"Source result: `{args.result}`\n\n"
        "完整逐轮数据见同目录 JSONL；下面展示第一条含可删除 action 的完整 trajectory。\n\n"
        + (markdown_blocks[0] if markdown_blocks else "没有找到可删除 action。\n"),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
