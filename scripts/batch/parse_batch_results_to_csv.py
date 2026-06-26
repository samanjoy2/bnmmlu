#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse Alibaba Model Studio Batch JSONL results into the same CSV layout
as the interactive runner (question_answerer scripts).

Input(s): one or more success.jsonl files produced by Batch Inference and the
source dataset CSV to obtain the correct answers and question IDs.

Output CSV columns:
  question_id, correct_answer, llm_answer, model_name, timestamp,
  is_correct, raw_output

Usage examples (PowerShell):
  python parse_batch_results_to_csv.py \
    --dataset "finished/verified_dataset_overlap_with_hard_subset.csv" \
    --results batches/2820a10c-*_success.jsonl batches/e8fb6ffe-*_success.jsonl \
    --model "qwen-qwen3-plus" --few-shot-k 5 --few-shot-rand --think 0

  # Or with explicit output filename
  python parse_batch_results_to_csv.py --dataset finished/verified_dataset_overlap_with_hard_subset.csv \
    --results batches/*_success.jsonl --model qwen-plus --few-shot-k 0 --out llm_test_results_qwen_plus__cot_json__thinkOff__fs0.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _extract_answer_from_text(text: Optional[str]) -> Optional[str]:
    """Extract A/B/C/D from a free-form model output.
    Tries strict JSON, embedded JSON, and textual patterns.
    """
    if not text:
        return None
    s = str(text).strip()
    # Try strict JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and 'answer' in obj:
            val = str(obj['answer']).strip().upper()
            if val in ['A', 'B', 'C', 'D']:
                return val
    except Exception:
        pass
    # Embedded JSON
    try:
        import re as _re
        m = _re.search(r"\{[^{}]*\"answer\"\s*:\s*\"([ABCD])\"[^{}]*\}", s, flags=_re.IGNORECASE)
        if m:
            return m.group(1).upper()
    except Exception:
        pass
    # Textual patterns
    try:
        import re as _re
        m = _re.search(r"\banswer\s*[:=\-]?\s*([ABCD])\b", s, flags=_re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m2 = _re.search(r"\b([ABCD])\b", s)
        if m2:
            return m2.group(1).upper()
    except Exception:
        pass
    return None


def _model_tag(model: str) -> str:
    return model.replace('.', '_').replace('-', '_').replace(':', '_').replace('/', '_')


def build_output_path(model: str, think: bool, few_shot_k: int, few_shot_rand: bool, out: Optional[str]) -> str:
    if out:
        return out
    parts = [_model_tag(model), 'cot_json', 'thinkOn' if think else 'thinkOff', f'fs{int(few_shot_k)}']
    if few_shot_k > 0 and few_shot_rand:
        parts.append('rand')
    run_tag = "__".join(parts)
    return f"llm_test_results_{run_tag}.csv"


def load_answer_key(dataset_csv: str) -> Dict[str, str]:
    """Load dataset and return a map from custom_id -> correct_answer.
    custom_id is derived like in build_batch_jsonl.py: prefer question_id, then
    Unique_Serial, else row index.
    """
    df = pd.read_csv(dataset_csv, encoding='utf-8')
    key: Dict[str, str] = {}
    for i, row in df.iterrows():
        qid = None
        try:
            if 'question_id' in row and pd.notna(row['question_id']):
                qid = row['question_id']
            elif 'Unique_Serial' in row and pd.notna(row['Unique_Serial']):
                qid = row['Unique_Serial']
        except Exception:
            pass
        if qid is None:
            qid = i
        qid_str = str(qid)
        ans = str(row.get('correct_answer', '')).strip().upper()[:1]
        key[qid_str] = ans
    return key


def iter_results_lines(paths: List[str]) -> Iterable[dict]:
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    yield obj
                except Exception:
                    continue


def main():
    ap = argparse.ArgumentParser(description='Convert Batch JSONL results to results CSV')
    ap.add_argument('--dataset', type=str, required=True, help='Source dataset CSV used to build batch inputs')
    ap.add_argument('--results', type=str, nargs='+', required=True, help='One or more success.jsonl files (supports globs)')
    ap.add_argument('--model', type=str, default='qwen-plus', help='Model name to record in CSV')
    ap.add_argument('--few-shot-k', type=int, default=0)
    ap.add_argument('--few-shot-rand', action='store_true')
    ap.add_argument('--think', type=int, default=0, help='1 for thinkOn, 0 for thinkOff (for filename tag only)')
    ap.add_argument('--out', type=str, default=None, help='Output CSV path (optional)')
    args = ap.parse_args()

    # Expand globs
    paths: List[str] = []
    for p in args.results:
        expanded = glob.glob(p)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(p)
    if not paths:
        raise FileNotFoundError('No result JSONL files found')

    out_csv = build_output_path(args.model, bool(args.think), args.few_shot_k, args.few_shot_rand, args.out)
    answer_key = load_answer_key(args.dataset)

    # Prepare output CSV
    columns = ['question_id', 'correct_answer', 'llm_answer', 'model_name', 'timestamp', 'is_correct', 'raw_output']
    df_out = pd.DataFrame(columns=columns)
    df_out.to_csv(out_csv, index=False, encoding='utf-8')

    # Iterate lines and extract
    rows: List[dict] = []
    for obj in iter_results_lines(paths):
        custom_id = str(obj.get('custom_id', ''))
        error = obj.get('error')
        body = None
        content = ''
        if not error:
            try:
                resp = obj.get('response') or {}
                body = resp.get('body') or {}
                choices = body.get('choices') or []
                if choices:
                    msg = choices[0].get('message') or {}
                    content = str(msg.get('content', '')).strip()
            except Exception:
                pass
        else:
            content = f"ERROR: {error}"

        llm_answer = _extract_answer_from_text(content)
        correct_answer = (answer_key.get(custom_id) or '').upper()[:1]
        is_correct = (llm_answer == correct_answer) if llm_answer and correct_answer else False

        row = {
            'question_id': custom_id,
            'correct_answer': correct_answer,
            'llm_answer': llm_answer,
            'model_name': args.model,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_correct': is_correct,
            'raw_output': content,
        }
        rows.append(row)

    # Append to CSV
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, mode='a', header=False, index=False, encoding='utf-8')

    print(f"Saved results CSV: {out_csv}")
    print(f"Lines processed: {len(rows)} from {len(paths)} file(s)")


if __name__ == '__main__':
    main()

