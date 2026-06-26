#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build and submit OpenAI batch jobs for verifying questions (chunks of N=1000).

- Reads `finished/hard_subset_by_llm_failures.csv`
- Skips question_ids already present in `finished/verification_results_gpt_5_mini.csv` (resume)
- Creates JSONL request files (one per batch chunk)
- Uploads each JSONL with purpose="batch" and submits a batch to /v1/chat/completions
"""

import os
import io
import sys
import json
import math
from datetime import datetime
from typing import List, Dict, Any, Iterable, Set

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import openai
except Exception:
    openai = None


# Ensure UTF-8 stdout for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

load_dotenv()


INPUT_CSV = os.path.join("finished", "hard_subset_by_llm_failures.csv")
RESULTS_CSV = os.path.join("finished", "verification_results_gpt_5_mini.csv")
MODEL_NAME = "gpt-5-mini"
BATCH_DIR = os.path.join("finished", "batches")
INDEX_CSV = os.path.join(BATCH_DIR, "batches_created.csv")


RESPONSE_SYSTEM_INSTRUCTIONS = (
    "You are a strict dataset verifier. Always reply as a compact JSON object. "
    "If needed, consult reliable open internet sources to verify facts, but still return only JSON."
)

RESPONSE_USER_SCHEMA = (
    "You will be given a multiple-choice question with four options and an\n"
    "existing dataset-provided answer (A/B/C/D). You must decide: \n"
    "- accepted: whether this question should be accepted into the final dataset (true/false)\n"
    "- reason: one-sentence justification\n"
    "- question_valid: whether the question is well-formed (true/false)\n"
    "- given_answer_correct: whether the provided correct answer is actually correct (true/false)\n"
    "- impossible_to_answer: true if the question cannot be answered from the given options\n"
    "- llm_selected_answer: your selected answer among A/B/C/D if answerable, else null\n"
    "- flags: array of strings like ['ambiguous', 'misleading', 'bad_translation'] if any\n"
    "You may search the internet if needed to verify facts. Return ONLY a JSON object with exactly these keys and booleans/strings as appropriate."
)


def parse_options(options_str: str) -> List[str]:
    if pd.isna(options_str):
        return ["", "", "", ""]
    txt = str(options_str).strip()
    try:
        loaded = json.loads(txt.replace("'", '"'))
        if isinstance(loaded, list):
            opts = [str(x) for x in loaded]
            while len(opts) < 4:
                opts.append("")
            return opts[:4]
    except Exception:
        pass
    if txt.startswith("[") and txt.endswith("]"):
        inner = txt[1:-1]
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
    else:
        parts = [p.strip() for p in txt.split(",")]
    while len(parts) < 4:
        parts.append("")
    return parts[:4]


def build_prompt(question: str, options: List[str], correct_answer: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"A) {options[0]}\n"
        f"B) {options[1]}\n"
        f"C) {options[2]}\n"
        f"D) {options[3]}\n\n"
        f"Dataset-provided correct answer: {str(correct_answer).strip().upper()}\n\n"
        f"{RESPONSE_USER_SCHEMA}"
    )


def get_processed_ids(results_csv: str) -> Set[str]:
    if not os.path.exists(results_csv):
        return set()
    try:
        df = pd.read_csv(results_csv, dtype={"question_id": str}, low_memory=False, encoding="utf-8")
        if df.empty or "question_id" not in df.columns:
            return set()
        return set(df["question_id"].astype(str))
    except Exception:
        return set()


def make_requests(df: pd.DataFrame, start_idx: int, end_idx: int) -> Iterable[Dict[str, Any]]:
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        question_id = str(row.get("question_id", ""))
        question = str(row.get("question", ""))
        options = parse_options(row.get("options", ""))
        correct_answer = str(row.get("correct_answer", "")).strip().upper()
        prompt = build_prompt(question, options, correct_answer)

        yield {
            "custom_id": f"qid:{question_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": RESPONSE_SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "reasoning_effort": "low",
            },
        }


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


def create_client():
    if openai is None:
        raise RuntimeError("openai package is not installed. Please `pip install openai`.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
    return openai.OpenAI(api_key=api_key)


def ensure_dirs() -> None:
    os.makedirs(BATCH_DIR, exist_ok=True)


def append_index(batch_id: str, input_file_id: str, chunk_idx: int, count: int, meta: Dict[str, Any]) -> None:
    row = {
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "chunk_index": chunk_idx,
        "requests": count,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **meta,
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(INDEX_CSV)
    df.to_csv(INDEX_CSV, mode="a", header=header, index=False, encoding="utf-8")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build and submit OpenAI batch jobs (verify questions)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Requests per batch file")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to include (after resume filter)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing results and include all rows")
    parser.add_argument("--dry-run", action="store_true", help="Only write JSONL files, do not upload/submit")
    parser.add_argument("--tag", type=str, default="job=hard_subset_verify", help="Metadata tag to attach to batch")
    args = parser.parse_args()

    ensure_dirs()
    df = pd.read_csv(INPUT_CSV, dtype={"question_id": str}, low_memory=False, encoding="utf-8")
    processed = set() if args.no_resume else get_processed_ids(RESULTS_CSV)

    # choose unprocessed indices
    indices = [i for i in range(len(df)) if str(df.iloc[i]["question_id"]) not in processed]
    if args.limit is not None:
        indices = indices[: args.limit]

    if not indices:
        print("✅ Nothing to include (all questions processed or limit=0)")
        return

    total = len(indices)
    num_chunks = math.ceil(total / args.batch_size)
    print(f"🚀 Preparing {total} requests across {num_chunks} batch file(s) of size {args.batch_size}")

    client = None if args.dry_run else create_client()

    for chunk_idx in range(num_chunks):
        start = chunk_idx * args.batch_size
        end = min(total, (chunk_idx + 1) * args.batch_size)
        sel = indices[start:end]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = os.path.join(BATCH_DIR, f"requests_{MODEL_NAME}_{ts}_{chunk_idx:04d}.jsonl")
        count = write_jsonl(jsonl_path, make_requests(df, sel[0], sel[-1] + 1))
        print(f"📝 Wrote {count} requests -> {jsonl_path}")

        if args.dry_run:
            continue

        # Upload file
        with open(jsonl_path, "rb") as fh:
            up = client.files.create(file=fh, purpose="batch")
        input_file_id = up.id
        # Submit batch
        meta_kv = {
            "tag": args.tag,
            "model": MODEL_NAME,
            "chunk_index": str(chunk_idx),
        }
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=meta_kv,
        )
        print(f"📤 Submitted batch: {batch.id} (file={input_file_id})")
        append_index(batch.id, input_file_id, chunk_idx, count, meta_kv)


if __name__ == "__main__":
    main()


