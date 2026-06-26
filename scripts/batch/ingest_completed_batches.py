#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest locally saved completed OpenAI batch outputs and build/update the results dataset.

- Scans `finished/completed_batches/*.jsonl`
- Parses each line, extracts `qid:{question_id}` and the JSON content from the model
- Joins with `finished/hard_subset_by_llm_failures.csv` to include dataset fields
- Appends to `finished/verification_results_gpt_5_mini.csv` with resume-safety

Usage:
  python ingest_completed_batches.py
  python ingest_completed_batches.py --only-accepted  # also writes filtered CSV
  python ingest_completed_batches.py --limit 5000      # for testing
"""

import os
import io
import sys
import json
import glob
from datetime import datetime
from typing import Dict, Any, Iterable, Optional, List, Set

import pandas as pd
from tqdm import tqdm


# Ensure UTF-8 stdout for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


COMPLETED_DIR = os.path.join("finished", "completed_batches")
INPUT_CSV = os.path.join("finished", "hard_subset_by_llm_failures.csv")
RESULTS_CSV = os.path.join("finished", "verification_results_gpt_5_mini.csv")
FILTERED_ACCEPTED_CSV = os.path.join("finished", "verified_dataset_gpt_5_mini_ACCEPTED.csv")


RESULT_COLUMNS: List[str] = [
    "subdomain_name",
    "question_id",
    "failure_ratio",
    "n_models",
    "subject",
    "question",
    "correct_answer",
    "options",
    "accepted",
    "reason",
    "question_valid",
    "given_answer_correct",
    "impossible_to_answer",
    "llm_selected_answer",
    "flags",
    "model_name",
    "timestamp",
]


def ensure_results_header() -> None:
    if os.path.exists(RESULTS_CSV):
        return
    pd.DataFrame(columns=RESULT_COLUMNS).to_csv(RESULTS_CSV, index=False, encoding="utf-8")


def get_processed_ids() -> Set[str]:
    if not os.path.exists(RESULTS_CSV):
        return set()
    try:
        df = pd.read_csv(RESULTS_CSV, dtype={"question_id": str}, low_memory=False, encoding="utf-8")
        if df.empty or "question_id" not in df.columns:
            return set()
        return set(df["question_id"].astype(str))
    except Exception:
        return set()


def load_dataset_map() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, dtype={"question_id": str}, low_memory=False, encoding="utf-8")
    # Set index for fast lookup
    df = df.set_index("question_id", drop=False)
    return df


def epoch_to_iso(ts: Any) -> str:
    try:
        # Some outputs might be seconds; others already string
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        s = str(ts)
        if s.isdigit():
            return datetime.utcfromtimestamp(int(s)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_output_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one JSONL line from a batch output file into a result dict.

    Expected structure (per OpenAI batches):
      {
        "custom_id": "qid:<question_id>",
        "response": { "body": { "choices": [ { "message": { "content": "{...json...}" } } ], "model": "...", "created": 1711471533 } }
      }
    """
    try:
        obj = json.loads(line)
    except Exception:
        return None

    cid = obj.get("custom_id", "")
    if not isinstance(cid, str) or not cid.startswith("qid:"):
        return None
    qid = cid.split(":", 1)[1]

    response = obj.get("response") or {}
    body = response.get("body") or {}
    try:
        content = body["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
    except Exception:
        return None

    result: Dict[str, Any] = {
        "question_id": qid,
        "accepted": bool(data.get("accepted")),
        "reason": str(data.get("reason", "")),
        "question_valid": bool(data.get("question_valid")),
        "given_answer_correct": bool(data.get("given_answer_correct")),
        "impossible_to_answer": bool(data.get("impossible_to_answer")),
        "llm_selected_answer": data.get("llm_selected_answer"),
        "flags": json.dumps(data.get("flags", []), ensure_ascii=False),
        "model_name": body.get("model", "gpt-5-mini"),
        "timestamp": epoch_to_iso(body.get("created")),
    }
    return result


def merge_with_dataset(dataset_map: pd.DataFrame, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        qid = str(r["question_id"])
        ds = dataset_map.loc[qid] if qid in dataset_map.index else None
        out_rows.append({
            "subdomain_name": "" if ds is None else ds.get("subdomain_name", ""),
            "question_id": qid,
            "failure_ratio": "" if ds is None else ds.get("failure_ratio", ""),
            "n_models": "" if ds is None else ds.get("n_models", ""),
            "subject": "" if ds is None else ds.get("subject", ""),
            "question": "" if ds is None else ds.get("question", ""),
            "correct_answer": "" if ds is None else str(ds.get("correct_answer", "")),
            "options": "" if ds is None else str(ds.get("options", "")),
            "accepted": r.get("accepted"),
            "reason": r.get("reason"),
            "question_valid": r.get("question_valid"),
            "given_answer_correct": r.get("given_answer_correct"),
            "impossible_to_answer": r.get("impossible_to_answer"),
            "llm_selected_answer": r.get("llm_selected_answer"),
            "flags": r.get("flags"),
            "model_name": r.get("model_name"),
            "timestamp": r.get("timestamp"),
        })
    return pd.DataFrame(out_rows, columns=RESULT_COLUMNS)


def append_results(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df.to_csv(RESULTS_CSV, mode="a", header=False, index=False, encoding="utf-8")


def write_filtered_accepted() -> None:
    try:
        df = pd.read_csv(RESULTS_CSV, low_memory=False, encoding="utf-8")
        if "accepted" in df.columns:
            acc = df[df["accepted"] == True]
            acc.to_csv(FILTERED_ACCEPTED_CSV, index=False, encoding="utf-8")
            print(f"✅ Wrote accepted-only dataset -> {FILTERED_ACCEPTED_CSV} ({len(acc):,} rows)")
    except Exception as e:
        print(f"⚠️ Could not write accepted-only file: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest completed batch output JSONLs into verification results")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N lines across files (testing)")
    parser.add_argument("--only-accepted", action="store_true", help="After append, also export accepted-only dataset")
    args = parser.parse_args()

    ensure_results_header()
    processed_ids = get_processed_ids()
    dataset_map = load_dataset_map()

    files = sorted(glob.glob(os.path.join(COMPLETED_DIR, "*.jsonl")))
    if not files:
        print(f"❌ No JSONL files found under {COMPLETED_DIR}")
        return

    total_appended = 0
    remaining = args.limit
    for path in files:
        print(f"📥 Ingesting: {path}")
        buf: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=os.path.basename(path)):
                row = parse_output_line(line)
                if not row:
                    continue
                qid = str(row["question_id"])
                if qid in processed_ids:
                    continue
                processed_ids.add(qid)
                buf.append(row)
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        pass
                # Flush periodically to reduce memory
                if len(buf) >= 2000:
                    df_append = merge_with_dataset(dataset_map, buf)
                    append_results(df_append)
                    total_appended += len(df_append)
                    buf = []
                if remaining is not None and remaining <= 0:
                    break
        if buf:
            df_append = merge_with_dataset(dataset_map, buf)
            append_results(df_append)
            total_appended += len(df_append)
        if remaining is not None and remaining <= 0:
            break

    print(f"✅ Appended {total_appended:,} new rows to {RESULTS_CSV}")
    if args.only_accepted:
        write_filtered_accepted()


if __name__ == "__main__":
    main()





