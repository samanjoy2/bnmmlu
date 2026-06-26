#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect Anthropic batch outputs into CSVs
========================================

Reads finished Anthropic Messages Batch result files (JSONL) from a folder and
writes per-file CSVs with the schema:

question_id,correct_answer,llm_answer,model_name,timestamp,is_correct,raw_output

- Attempts to extract `llm_answer` (A/B/C/D) from the model's text.
- If a dataset CSV is provided, fills `correct_answer` by joining on `question_id`
  (or `Unique_Serial`) that was used as `custom_id` in batch requests.

Usage
- python collect_anthropic_batches_to_csv.py --batches-dir batches/finised_batches \
    --dataset finished/verified_dataset_overlap_with_hard_subset.csv \
    --output-dir .

Notes
- Expects each JSONL line to include `custom_id` and the model response payload
  (fields like `model` and a text `content`). The exact shape varies by API; this
  collector uses best-effort fields and robust text extraction.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _safe_read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def _extract_text_from_result(obj: Dict[str, Any]) -> str:
    """Return the full human-visible text from a result.
    Aggregates all text segments in content lists so the final JSON line is included.
    """
    # 1) messages-style: {'content': [{'type':'text','text':'...'}, ...]}
    try:
        content = obj.get('content')
        if isinstance(content, list) and content:
            parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    parts.append(str(item.get('text') or ''))
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "\n".join([p for p in parts if p])
    except Exception:
        pass
    # 2) direct text field
    try:
        txt = obj.get('text')
        if isinstance(txt, str):
            return txt
    except Exception:
        pass
    # 3) nested under 'response' / 'result' / 'message'
    for k in ('response', 'result', 'message'):
        try:
            inner = obj.get(k)
            if isinstance(inner, dict):
                t = _extract_text_from_result(inner)
                if t:
                    return t
        except Exception:
            pass
    return ""


def _extract_answer_from_text(text: str) -> Optional[str]:
    """Extract A/B/C/D robustly, avoiding stray letters in prose.
    Priority:
      1) Last JSON object with key 'answer'
      2) Phrases like 'final answer is: A', 'answer: B', 'I choose C', 'option D'
      3) Line that is exactly A/B/C/D or JSON fenced in code blocks
    Never use a generic single-letter fallback.
    """
    if not text:
        return None
    s = text.strip()
    import re as _re

    # 1) Any JSON objects with 'answer' (take last occurrence)
    try:
        json_objs = list(_re.finditer(r"\{[^{}]*\"answer\"\s*:\s*\"?([ABCD1-4])\"?[^{}]*\}", s, flags=_re.IGNORECASE))
        if json_objs:
            val = json_objs[-1].group(1).upper()
            if val in ('1','2','3','4'):
                return {'1':'A','2':'B','3':'C','4':'D'}[val]
            return val
    except Exception:
        pass

    # 2) Scan lines from bottom to top for explicit phrases
    try:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        for ln in reversed(lines):
            # Normalize
            low = ln.lower()
            # common phrases
            m = _re.search(r"(final\s+answer|answer|correct\s+answer|i\s+choose|my\s+choice|option)\s*[:=\-]?\s*([abcd1-4])\b", low)
            if m:
                val = m.group(2).upper()
                if val in ('1','2','3','4'):
                    return {'1':'A','2':'B','3':'C','4':'D'}[val]
                if val in ('A','B','C','D'):
                    return val
            # Standalone JSON code fence line
            m2 = _re.search(r"\{\s*\"answer\"\s*:\s*\"?([ABCD1-4])\"?\s*\}\s*$", ln, flags=_re.IGNORECASE)
            if m2:
                val = m2.group(1).upper()
                if val in ('1','2','3','4'):
                    return {'1':'A','2':'B','3':'C','4':'D'}[val]
                return val
            # A single-letter line like 'A' or 'B'
            if ln in ('A','B','C','D'):
                return ln
    except Exception:
        pass

    # 3) Option word pattern anywhere
    try:
        m = _re.search(r"option\s*([abcd])\b", s, flags=_re.IGNORECASE)
        if m:
            return m.group(1).upper()
    except Exception:
        pass

    return None


def _load_answer_map(dataset_path: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not dataset_path or not os.path.exists(dataset_path):
        return mapping
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
        for _, row in df.iterrows():
            ca = str(row.get('correct_answer', '')).strip().upper()[:1]
            if ca not in ('A','B','C','D'):
                continue
            # prefer question_id then Unique_Serial
            qid = None
            if 'question_id' in row and pd.notna(row['question_id']):
                qid = str(row['question_id'])
            elif 'Unique_Serial' in row and pd.notna(row['Unique_Serial']):
                qid = str(row['Unique_Serial'])
            if qid is not None:
                mapping[qid] = ca
    except Exception:
        return {}
    return mapping


def model_safe_name(name: Optional[str]) -> str:
    if not name:
        return 'unknown_model'
    return str(name).replace(' ', '_').replace('/', '_').replace(':', '_').replace('-', '_').replace('.', '_')


def main():
    parser = argparse.ArgumentParser(description='Collect Anthropic batch JSONL outputs into CSV files')
    parser.add_argument('--batches-dir', required=True, help='Directory with finished JSONL result files')
    parser.add_argument('--dataset', default=_env('DATASET_CSV', 'finished/verified_dataset_overlap_with_hard_subset.csv'), help='Dataset CSV to look up correct_answer (by question_id or Unique_Serial)')
    parser.add_argument('--output-dir', default='.', help='Where to write CSVs')
    parser.add_argument('--single-csv', default=None, help='If set, write a single combined CSV to this path (or filename within output-dir if relative). When used, per-file CSVs are not written.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    answer_map = _load_answer_map(args.dataset)

    # gather jsonl files
    files = [os.path.join(args.batches_dir, f) for f in os.listdir(args.batches_dir) if f.lower().endswith('.jsonl')]
    if not files:
        print(f'No .jsonl files found in {args.batches_dir}')
        return

    combined_rows = []
    for path in sorted(files):
        rows = []
        inferred_model = None
        for rec in _safe_read_jsonl(path):
            custom_id = str(rec.get('custom_id') or rec.get('id') or '')
            # payload may be under 'result', 'response', or top-level fields
            payload = rec.get('result') or rec.get('response') or rec
            model_name = payload.get('model') if isinstance(payload, dict) else None
            if model_name and not inferred_model:
                inferred_model = model_name
            text = _extract_text_from_result(payload)
            llm_answer = _extract_answer_from_text(text)
            correct_answer = answer_map.get(custom_id, '')
            is_correct = bool(llm_answer and correct_answer and (llm_answer == correct_answer))
            row_obj = {
                'question_id': custom_id,
                'correct_answer': correct_answer,
                'llm_answer': llm_answer or '',
                'model_name': model_name or '',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'is_correct': is_correct,
                'raw_output': text,
            }
            rows.append(row_obj)
        if not rows:
            print(f'Skipping empty file: {path}')
            continue

        if args.single_csv:
            combined_rows.extend(rows)
        else:
            df = pd.DataFrame(rows)
            # name like: llm_test_results_<model>__batch__<stem>.csv
            stem = os.path.splitext(os.path.basename(path))[0]
            model_key = model_safe_name(inferred_model or df['model_name'].dropna().astype(str).unique()[0] if any(df['model_name']) else 'anthropic')
            out_name = f"llm_test_results_{model_key}__batch__{stem}.csv"
            out_path = os.path.join(args.output_dir, out_name)
            df.to_csv(out_path, index=False, encoding='utf-8')
            print(f'Wrote CSV: {out_path} ({len(df)} rows)')

    if args.single_csv:
        if not combined_rows:
            print('No rows collected to combine.')
            return
        df_all = pd.DataFrame(combined_rows)
        out_path = args.single_csv
        if not os.path.isabs(out_path):
            out_path = os.path.join(args.output_dir, out_path)
        # Default name if only a filename hint was not provided
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path, 'llm_test_results_anthropic_batches_combined.csv')
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        df_all.to_csv(out_path, index=False, encoding='utf-8')
        print(f'Wrote combined CSV: {out_path} ({len(df_all)} rows)')


if __name__ == '__main__':
    main()
