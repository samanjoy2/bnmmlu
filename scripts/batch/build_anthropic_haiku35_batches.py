#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Anthropic Batch JSONL files for Claude 3.5 Haiku
======================================================

Creates JSONL batches (3,000 requests per file by default) for Anthropic's
Messages Batch API using model "claude-3-5-haiku-latest" with 5-shot exemplars
that include reasoning text from a CSV column (model_reasoning_en). The model is
instructed to output only a final JSON answer on the last line.

This script only prepares batch files; it does not submit them. You can submit
these JSONL files using Anthropic's Batch API.

Env/CLI highlights:
- ANTHROPIC_MODEL (default: claude-3-5-haiku-latest)
- FEW_SHOT_CSV (default: finished/special_small_subset_for_cot_final.csv)
- FEW_SHOT_K (default: 5)
- FEW_SHOT_INCLUDE_REASONING (default: 1)
- INPUT_CSV (default: finished/verified_dataset_overlap_with_hard_subset.csv)
- BATCH_SIZE (default: 3000)
- OUTPUT_DIR (default: batches/anthropic_haiku35)
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd


def _env_get(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        val = os.getenv(name)
        if val is None or val == "":
            return default
        return val
    except Exception:
        return default


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
    # Extract quoted parts
    try:
        import re as _re
        parts = []
        for m in _re.finditer(r"'([^']*)"+"'|\"([^\"]*)\"", s):
            parts.append((m.group(1) if m.group(1) is not None else m.group(2)).strip())
        if parts:
            return parts
    except Exception:
        pass
    # Fallback comma split
    try:
        return [p.strip().strip("'\"") for p in s.strip('[]').split(',') if p.strip()]
    except Exception:
        return []


def build_few_shot_text(few_shot_csv: str, k: int, include_reasoning: bool, random_sample: bool = False) -> str:
    try:
        if k <= 0 or not few_shot_csv or not os.path.exists(few_shot_csv):
            return ""
        df = pd.read_csv(few_shot_csv, encoding='utf-8')
        if df.empty:
            return ""
        df_s = df.sample(min(k, len(df)), random_state=42) if random_sample else df.head(k)
        exemplars: List[str] = []
        for _, row in df_s.iterrows():
            q = str(row.get('question', ''))
            opts = parse_options(row.get('options'))
            if len(opts) < 4:
                opts = list(opts) + [''] * (4 - len(opts))
            ans = str(row.get('correct_answer', '')).strip().upper()[:1]
            if ans not in ('A','B','C','D'):
                continue
            exemplar_reasoning = ''
            try:
                exemplar_reasoning = str(row.get('model_reasoning_en', '') or '').strip()
            except Exception:
                exemplar_reasoning = ''
            if include_reasoning and exemplar_reasoning:
                reasoning_line = f"Reasoning: {exemplar_reasoning}\n"
            else:
                reasoning_line = "Reasoning: Briefly analyze and choose the best option.\n"
            exemplars.append(
                "Example:\n"
                f"Question: {q}\n"
                f"Options:\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n"
                f"{reasoning_line}"
                f"Final: {{\"answer\":\"{ans}\"}}\n"
            )
        if exemplars:
            return "Few-shot exemplars (follow the format; last line is ONLY the JSON):\n\n" + "\n".join(exemplars)
    except Exception:
        pass
    return ""


def format_user_prompt(question: str, options: List[str], few_shot_text: str) -> str:
    if len(options) < 4:
        options = list(options) + [''] * (4 - len(options))
    instruction = (
        "Think step by step briefly, but do not reveal your reasoning. "
        "On the final line output ONLY one JSON object {\"answer\":\"A\"}."
    )
    fewshot_sep = "\n" if few_shot_text else ""
    return (
        f"{instruction}\n\n"
        f"{few_shot_text}{fewshot_sep}"
        f"Question: {question}\n\n"
        f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}"
    )


def build_request_obj(custom_id: str, model: str, system_prompt: str, user_text: str, max_tokens: int = 64) -> Dict[str, Any]:
    # Anthropic Messages Batch API expects JSONL lines with fields like:
    # {"custom_id": "...", "params": {"model": "...", "max_tokens": 64, "system": "...", "messages": [{"role":"user","content":"..."}]}}
    return {
        "custom_id": str(custom_id),
        "params": {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_text}
            ],
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Build Anthropic batch JSONL files for Claude 3.5 Haiku")
    parser.add_argument("--input-csv", default=_env_get('INPUT_CSV', 'finished/verified_dataset_overlap_with_hard_subset.csv'))
    parser.add_argument("--few-shot-csv", default=_env_get('FEW_SHOT_CSV', 'finished/special_small_subset_for_cot_final.csv'))
    parser.add_argument("--few-shot-k", type=int, default=int(_env_get('FEW_SHOT_K', '5') or '5'))
    parser.add_argument("--include-reasoning", action="store_true", default=str(_env_get('FEW_SHOT_INCLUDE_REASONING', '1')).lower() in ('1','true','yes','on'))
    parser.add_argument("--batch-size", type=int, default=int(_env_get('BATCH_SIZE', '3000') or '3000'))
    parser.add_argument("--output-dir", default=_env_get('OUTPUT_DIR', 'batches/anthropic_haiku35'))
    parser.add_argument("--model", default=_env_get('ANTHROPIC_MODEL', 'claude-3-5-haiku-latest'))
    parser.add_argument("--max-tokens", type=int, default=int(_env_get('MAX_TOKENS', '64') or '64'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build shared system prompt
    system_prompt = "Output exactly one JSON object {\"answer\":\"A\"} on the final line only."

    # Build few-shot exemplars text
    few_shot_text = build_few_shot_text(
        few_shot_csv=args.few_shot_csv,
        k=max(0, int(args.few_shot_k)),
        include_reasoning=bool(args.include_reasoning),
        random_sample=False,
    )

    # Load dataset
    df = pd.read_csv(args.input_csv, encoding='utf-8')
    if df.empty:
        print("No rows found in input CSV.")
        return

    requests: List[Dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        # Determine a stable id
        custom_id = None
        for key in ("question_id", "Unique_Serial"):
            if key in row and pd.notna(row[key]):
                custom_id = str(row[key])
                break
        if custom_id is None:
            custom_id = str(idx)

        q = str(row.get('question', ''))
        options = parse_options(row.get('options'))
        if len(options) < 4:
            # Skip malformed entries
            continue

        user_text = format_user_prompt(q, options, few_shot_text)
        req = build_request_obj(
            custom_id=custom_id,
            model=args.model,
            system_prompt=system_prompt,
            user_text=user_text,
            max_tokens=args.max_tokens,
        )
        requests.append(req)

    if not requests:
        print("No valid requests were generated.")
        return

    # Write in chunks of batch-size lines per JSONL file
    batch_size = max(1, int(args.batch_size))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    total = len(requests)
    written = 0
    part = 1
    while written < total:
        chunk = requests[written:written + batch_size]
        out_path = os.path.join(
            args.output_dir,
            f"anthropic_haiku35_batch_{timestamp}_part{part:03d}.jsonl",
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            for req in chunk:
                f.write(json.dumps(req, ensure_ascii=False))
                f.write("\n")
        print(f"Wrote {len(chunk)} requests -> {out_path}")
        written += len(chunk)
        part += 1

    print(f"Done. Total requests written: {total}")


if __name__ == "__main__":
    main()
