#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Batch JSONL for Alibaba Model Studio (OpenAI-compatible Batch)

Reads a questions CSV and emits one or more JSONL files ready for upload to
the batch inference console/API. Each line is a POST /v1/chat/completions
request targeting a single question.

Defaults target dataset:
  finished/verified_dataset_overlap_with_hard_subset.csv

By default this is 0-shot CoT (non-reasoning). You can optionally add few-shot
exemplars using --few-shot-k and --few-shot-csv.

Examples (PowerShell):
  # 0-shot, non-reasoning, qwen-plus
  python build_batch_jsonl.py \
    --dataset finished/verified_dataset_overlap_with_hard_subset.csv \
    --out-dir batches --model qwen-plus

  # 5-shot, random exemplars
  python build_batch_jsonl.py \
    --few-shot-k 5 --few-shot-random \
    --few-shot-csv finished/special_small_subset_for_cot_final.csv

Files are sharded as batch_qwen-plus__fs{K}_partNNN.jsonl
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv


def parse_options(options_str) -> List[str]:
    if options_str is None or (isinstance(options_str, float) and pd.isna(options_str)):
        return []
    s = str(options_str).strip()
    # Try JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj]
    except Exception:
        pass
    # Try Python literal
    try:
        import ast

        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj]
    except Exception:
        pass
    # Fallback: quoted captures
    try:
        import re as _re

        parts = []
        for m in _re.finditer(r"'([^']*)" + r"'|\"([^\"]*)\"", s):
            parts.append((m.group(1) if m.group(1) is not None else m.group(2)).strip())
        if parts:
            return parts
    except Exception:
        pass
    # Last resort: comma split
    try:
        return [p.strip().strip("'\"") for p in s.strip("[]").split(",") if p.strip()]
    except Exception:
        return []


def load_few_shot_text(k: int, random_pick: bool, csv_path: str) -> str:
    try:
        k = max(0, int(k))
        if k == 0 or not csv_path or not os.path.exists(csv_path):
            return ""
        df = pd.read_csv(csv_path, encoding="utf-8")
        if df.empty:
            return ""
        df_s = df.sample(min(k, len(df)), random_state=42) if random_pick else df.head(k)
        exemplars = []
        for _, row in df_s.iterrows():
            q = str(row.get("question", ""))
            options = row.get("options")
            try:
                opts = json.loads(str(options)) if options is not None else []
            except Exception:
                opts = []
            if len(opts) < 4:
                opts += [""] * (4 - len(opts))
            ans = str(row.get("correct_answer", "")).strip().upper()[:1]
            if ans not in ("A", "B", "C", "D"):
                continue
            exemplars.append(
                "Example:\n"
                f"Question: {q}\n"
                f"Options:\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n"
                "Reasoning: Briefly analyze and choose the best option.\n"
                f"Final: {{\"answer\":\"{ans}\"}}\n"
            )
        if exemplars:
            return "Few-shot exemplars (follow the format; last line is ONLY the JSON):\n\n" + "\n".join(
                exemplars
            )
    except Exception:
        pass
    return ""


def format_question(question: str, options: List[str], few_shot_text: str) -> str:
    if len(options) < 4:
        options = options + [""] * (4 - len(options))
    instruction = (
        "Think step by step briefly, then on the final line output ONLY one JSON object {\"answer\":\"A\"}."
    )
    fewshot_sep = "\n" if few_shot_text else ""
    return (
        f"{instruction}\n\n"
        f"{few_shot_text}{fewshot_sep}"
        f"Question: {question}\n\n"
        f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}"
    )


def sanitize_id(val) -> str:
    s = str(val)
    # Keep it simple and JSONL-safe
    return s.replace("\n", " ").replace("\r", " ").strip()


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Create JSONL batch files from a questions CSV"
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="finished/verified_dataset_overlap_with_hard_subset.csv",
        help="Input CSV with columns: question, options, correct_answer, subdomain_name, etc.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="batches",
        help="Directory to write JSONL shards",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="qwen-plus",
        help="Model id to place in each request body",
    )
    ap.add_argument(
        "--system",
        type=str,
        default="Think briefly. Then output exactly one JSON object {\"answer\":\"A\"} on the final line only.",
        help="System prompt",
    )
    ap.add_argument("--start", type=int, default=0, help="Start offset in dataset")
    ap.add_argument("--limit", type=int, default=-1, help="Max rows to export (-1 = all)")
    ap.add_argument("--shard-size", type=int, default=20000, help="Max lines per JSONL file (<= 50000)")
    ap.add_argument("--few-shot-k", type=int, default=0, help="Number of few-shot exemplars to prepend")
    ap.add_argument("--few-shot-random", action="store_true", help="Sample exemplars randomly")
    ap.add_argument(
        "--few-shot-csv",
        type=str,
        default="finished/special_small_subset_for_cot_final.csv",
        help="CSV file containing exemplars (question/options/correct_answer)",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.dataset, encoding="utf-8")
    if args.start > 0:
        df = df.iloc[args.start :]
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit]

    few_shot_text = load_few_shot_text(args.few_shot_k, args.few_shot_random, args.few_shot_csv)

    # Determine how many shards
    total = len(df)
    shard = max(1, min(50000, int(args.shard_size)))
    n_files = int(math.ceil(total / shard))

    base_tag = f"batch_{args.model.replace('/', '-')}__fs{int(args.few_shot_k)}"

    written = []
    for fi in range(n_files):
        start = fi * shard
        end = min(total, (fi + 1) * shard)
        part_df = df.iloc[start:end]
        out_path = os.path.join(args.out_dir, f"{base_tag}_part{fi+1:03d}.jsonl")
        with io.open(out_path, "w", encoding="utf-8") as f:
            for idx, row in part_df.iterrows():
                # Prefer question_id / Unique_Serial else use row index
                qid = None
                try:
                    if "question_id" in row:
                        qid = row["question_id"]
                    elif "Unique_Serial" in row:
                        qid = row["Unique_Serial"]
                except Exception:
                    pass
                if qid is None:
                    qid = idx

                question = str(row.get("question", ""))
                options = parse_options(row.get("options"))
                content = format_question(question, options, few_shot_text)

                req = {
                    "custom_id": sanitize_id(qid),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": args.system},
                            {"role": "user", "content": content},
                        ],
                    },
                }
                f.write(json.dumps(req, ensure_ascii=False))
                f.write("\n")
        written.append(out_path)

    print(f"Created {len(written)} JSONL file(s):")
    for p in written:
        print(f" - {p}")
    print(f"Total requests: {total}")


if __name__ == "__main__":
    main()
