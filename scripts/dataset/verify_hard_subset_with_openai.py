#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify hard subset questions with OpenAI (GPT-5-mini)
=====================================================

- Loads `finished/hard_subset_by_llm_failures.csv`
- Creates a live-updating results CSV that includes dataset columns and LLM judgments
- Forces JSON responses from the model and uses low reasoning effort
- Supports resume: skips rows already present in the results CSV

Environment:
- Reads `OPENAI_API_KEY` from .env or environment
"""

import os
import sys
import io
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import openai
except Exception as e:  # pragma: no cover
    openai = None


# Ensure UTF-8 stdout for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Load environment variables
load_dotenv()


INPUT_CSV = os.path.join("finished", "hard_subset_by_llm_failures.csv")
MODEL_NAME = "gpt-5-mini"  # per user request
RESULTS_CSV = os.path.join(
    "finished",
    f"verification_results_{MODEL_NAME.replace('.', '_').replace('-', '_')}.csv",
)


RESPONSE_SYSTEM_INSTRUCTIONS = (
    "You are a strict dataset verifier. Always reply as a compact JSON object. "
    "If needed, consult reliable open internet sources to verify facts, but still return only JSON."
)

# We force a JSON schema in the user message; also pass response_format for safety
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


def initialize_results_csv() -> None:
    """Create the results CSV with headers if not exists."""
    if os.path.exists(RESULTS_CSV):
        print(f"✅ Using existing results CSV: {RESULTS_CSV}")
        return

    columns = [
        # dataset columns (from input CSV)
        "subdomain_name",
        "question_id",
        "failure_ratio",
        "n_models",
        "subject",
        "question",
        "correct_answer",
        "options",
        # LLM verification outputs
        "accepted",
        "reason",
        "question_valid",
        "given_answer_correct",
        "impossible_to_answer",
        "llm_selected_answer",
        "flags",
        # metadata
        "model_name",
        "timestamp",
    ]
    pd.DataFrame(columns=columns).to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"✅ Created results CSV: {RESULTS_CSV}")


def load_input_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"question_id": str}, low_memory=False, encoding="utf-8")
    print(f"✅ Loaded input dataset: {len(df):,} rows from {path}")
    return df


def get_processed_question_ids() -> set:
    if not os.path.exists(RESULTS_CSV):
        return set()
    try:
        df = pd.read_csv(RESULTS_CSV, dtype={"question_id": str}, low_memory=False, encoding="utf-8")
        if df.empty or "question_id" not in df.columns:
            return set()
        return set(df["question_id"].astype(str))
    except Exception:
        return set()


def parse_options(options_str: str) -> List[str]:
    """Parse the options field from the input CSV into a list of four strings."""
    if pd.isna(options_str):
        return ["", "", "", ""]
    txt = str(options_str).strip()
    # Try to load as Python list literal first
    try:
        loaded = json.loads(txt.replace("'", '"'))
        if isinstance(loaded, list):
            opts = [str(x) for x in loaded]
            while len(opts) < 4:
                opts.append("")
            return opts[:4]
    except Exception:
        pass

    # Fallback: naive split
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


def create_openai_client():
    if openai is None:
        raise RuntimeError("openai package is not installed. Please `pip install openai`.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment or .env")
    return openai.OpenAI(api_key=api_key)


def call_model(client, prompt: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": RESPONSE_SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},  # force JSON
                reasoning_effort="low",  # low reasoning effort
                # temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            # Basic validation
            expected_keys = {
                "accepted",
                "reason",
                "question_valid",
                "given_answer_correct",
                "impossible_to_answer",
                "llm_selected_answer",
                "flags",
            }
            if not isinstance(data, dict) or not expected_keys.issubset(set(data.keys())):
                raise ValueError("Invalid JSON keys in model response")
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"⚠️ Final failure parsing/calling model: {e}")
    return None


def append_result_row(row_out: Dict[str, Any]) -> None:
    df = pd.DataFrame([row_out])
    df.to_csv(RESULTS_CSV, mode="a", header=False, index=False, encoding="utf-8")


def process_all(resume: bool = True, limit: Optional[int] = None) -> None:
    initialize_results_csv()
    df = load_input_dataset(INPUT_CSV)
    processed_ids = get_processed_question_ids() if resume else set()

    client = create_openai_client()

    indices = []
    for i in range(len(df)):
        qid = str(df.iloc[i]["question_id"]) if "question_id" in df.columns else str(df.iloc[i].get("Unique_Serial", ""))
        # this file has column `question_id` already
        if resume and qid in processed_ids:
            continue
        indices.append(i)
        if limit is not None and len(indices) >= limit:
            break

    if not indices:
        print("✅ Nothing to process (all rows present in results CSV).")
        return

    print(f"🚀 Processing {len(indices)} questions (resume={resume})")
    for i in tqdm(indices, desc="Verifying questions"):
        src = df.iloc[i]
        subdomain = src.get("subdomain_name", "")
        question_id = str(src.get("question_id", ""))
        failure_ratio = src.get("failure_ratio", "")
        n_models = src.get("n_models", "")
        subject = src.get("subject", "")
        question = src.get("question", "")
        correct_answer = str(src.get("correct_answer", "")).strip().upper()
        options_raw = src.get("options", "")
        options = parse_options(options_raw)

        prompt = build_prompt(question, options, correct_answer)
        result = call_model(client, prompt)

        row_out: Dict[str, Any] = {
            "subdomain_name": subdomain,
            "question_id": question_id,
            "failure_ratio": failure_ratio,
            "n_models": n_models,
            "subject": subject,
            "question": question,
            "correct_answer": correct_answer,
            "options": json.dumps(options, ensure_ascii=False),
            "accepted": None,
            "reason": "",
            "question_valid": None,
            "given_answer_correct": None,
            "impossible_to_answer": None,
            "llm_selected_answer": None,
            "flags": json.dumps([], ensure_ascii=False),
            "model_name": MODEL_NAME,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if result is not None:
            # Coerce types
            row_out.update({
                "accepted": bool(result.get("accepted")),
                "reason": str(result.get("reason", "")),
                "question_valid": bool(result.get("question_valid")),
                "given_answer_correct": bool(result.get("given_answer_correct")),
                "impossible_to_answer": bool(result.get("impossible_to_answer")),
                "llm_selected_answer": (str(result.get("llm_selected_answer")) if result.get("llm_selected_answer") is not None else None),
                "flags": json.dumps(result.get("flags", []), ensure_ascii=False),
            })

        append_result_row(row_out)
        # throttle to be safe
        time.sleep(0.15)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify hard subset with OpenAI GPT-5-mini")
    parser.add_argument("--no-resume", action="store_true", help="Process all rows from scratch (ignore existing results)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows (for testing)")
    args = parser.parse_args()

    process_all(resume=not args.no_resume, limit=args.limit)


if __name__ == "__main__":
    main()


