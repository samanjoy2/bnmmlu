#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect completed OpenAI batch outputs and merge into results CSV.

- Reads index from `finished/batches/batches_created.csv`
- For each batch id, retrieves status; if completed, downloads output file
- Parses batch output JSONL and appends rows to `finished/verification_results_gpt_5_mini.csv`
- Resume-safe: skips `question_id` already present in results CSV
"""

import os
import io
import sys
import json
from typing import Dict, Any, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv

try:
    import openai
except Exception:
    openai = None


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
load_dotenv()

BATCH_DIR = os.path.join("finished", "batches")
INDEX_CSV = os.path.join(BATCH_DIR, "batches_created.csv")
RESULTS_CSV = os.path.join("finished", "verification_results_gpt_5_mini.csv")


def create_client():
    if openai is None:
        raise RuntimeError("openai package is not installed. Please `pip install openai`.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
    return openai.OpenAI(api_key=api_key)


def load_index() -> pd.DataFrame:
    if not os.path.exists(INDEX_CSV):
        raise FileNotFoundError("No batches_created.csv index found; run build_openai_batches.py first")
    return pd.read_csv(INDEX_CSV, low_memory=False)


def get_processed_ids() -> set:
    if not os.path.exists(RESULTS_CSV):
        return set()
    try:
        return set(pd.read_csv(RESULTS_CSV, dtype={"question_id": str})["question_id"].astype(str))
    except Exception:
        return set()


def ensure_results_header():
    if os.path.exists(RESULTS_CSV):
        return
    cols = [
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
    pd.DataFrame(columns=cols).to_csv(RESULTS_CSV, index=False, encoding="utf-8")


def parse_batch_line(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Expected per-line structure for output files
    # {"custom_id": "qid:...", "response": { ... chat.completions response ... }}
    try:
        cid = obj.get("custom_id", "")
        if not cid.startswith("qid:"):
            return None
        qid = cid.split(":", 1)[1]
        resp = obj.get("response", {})
        content = resp["body"]["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        # populate minimal fields; dataset fields are not in batch outputs, leave blank
        return {
            "subdomain_name": "",
            "question_id": qid,
            "failure_ratio": "",
            "n_models": "",
            "subject": "",
            "question": "",
            "correct_answer": "",
            "options": "",
            "accepted": bool(data.get("accepted")),
            "reason": str(data.get("reason", "")),
            "question_valid": bool(data.get("question_valid")),
            "given_answer_correct": bool(data.get("given_answer_correct")),
            "impossible_to_answer": bool(data.get("impossible_to_answer")),
            "llm_selected_answer": data.get("llm_selected_answer"),
            "flags": json.dumps(data.get("flags", []), ensure_ascii=False),
            "model_name": resp.get("body", {}).get("model", "gpt-5-mini"),
            "timestamp": resp.get("body", {}).get("created", None),
        }
    except Exception:
        return None


def append_rows(rows: Iterable[Dict[str, Any]]):
    df = pd.DataFrame(list(rows))
    if df.empty:
        return
    df.to_csv(RESULTS_CSV, mode="a", header=False, index=False, encoding="utf-8")


def download_file(client, file_id: str) -> str:
    # Save to batches directory for auditing
    content = client.files.content(file_id)
    save_path = os.path.join(BATCH_DIR, f"out_{file_id}.jsonl")
    with open(save_path, "wb") as f:
        f.write(content.read())
    return save_path


def collect():
    ensure_results_header()
    processed_ids = get_processed_ids()
    client = create_client()
    idx = load_index()

    for _, row in idx.iterrows():
        batch_id = row["batch_id"]
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"Batch {batch_id}: {status}")
        if status != "completed":
            continue
        out_id = batch.output_file_id
        if not out_id:
            continue
        path = download_file(client, out_id)
        print(f"📥 Downloaded output -> {path}")
        rows_to_append = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                row_out = parse_batch_line(obj)
                if not row_out:
                    continue
                qid = row_out["question_id"]
                if qid in processed_ids:
                    continue
                processed_ids.add(qid)
                rows_to_append.append(row_out)
        append_rows(rows_to_append)
        print(f"✅ Appended {len(rows_to_append)} rows to results")


def main():
    collect()


if __name__ == "__main__":
    main()





