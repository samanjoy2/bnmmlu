#!/usr/bin/env python3
"""
Verify answerability and correctness of hard questions using an OpenAI model with web verification instructions.

Input:  finished/hard_subset_by_llm_failures.csv (expects columns: question_id, question, options, correct_answer, subdomain_name, subject, ...)
Output: finished/hard_subset_web_verified.csv (original columns + web_is_answerable, web_is_correct)

Notes:
- The model name defaults to 'gpt-4.1-mini'.
- We request strict JSON from the model and persist progress incrementally.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


DEFAULT_INPUT = Path("finished/hard_subset_by_llm_failures.csv")
DEFAULT_OUTPUT = Path("finished/hard_subset_web_verified.csv")
DEFAULT_MODEL = "gpt-4.1-mini"


def parse_options(options_str: str) -> list:
    """Parse options string like "['A', 'B', 'C', 'D']" to a list of 4 strings.
    Returns empty list on error.
    """
    try:
        s = (options_str or "").strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            # naive split that works for simple 4-choice lists in this dataset
            parts = s[1:-1].split(",")
            parts = [p.strip().strip("'\"") for p in parts]
            return parts
        return []
    except Exception:
        return []


def build_prompt(question: str, options: list, correct_answer: str) -> str:
    options_block = ""
    if options:
        labels = ["A", "B", "C", "D"]
        lines = []
        for i, opt in enumerate(options[:4]):
            lines.append(f"{labels[i]}) {opt}")
        options_block = "\n" + "\n".join(lines)

    return (
        "You are a fact-checking assistant with reliable web access. "
        "Search the public internet and verify whether the following multiple-choice question is (1) answerable and (2) whether the provided correct answer matches reality. "
        "If web access is unavailable, use your best knowledge and be conservative. Respond ONLY in strict JSON with keys: "
        "web_is_answerable (boolean), web_is_correct (boolean). No extra keys.\n\n"
        f"Question:\n{question}\n"
        f"Options:{options_block}\n\n"
        f"Claimed correct answer (letter): {str(correct_answer).strip().upper()}\n\n"
        "Return JSON only, e.g.: {\"web_is_answerable\": true, \"web_is_correct\": false}"
    )


def call_model(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> Tuple[Optional[bool], Optional[bool], str, Optional[float]]:
    """Call OpenAI chat.completions with a JSON response requirement.
    Returns: (web_is_answerable, web_is_correct, raw_json_str, response_time_sec)
    """
    last_raw = ""
    duration: Optional[float] = None

    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Respond only with a single JSON object."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            duration = time.time() - start
            try:
                last_raw = resp.model_dump_json()
            except Exception:
                try:
                    last_raw = json.dumps(resp, default=str)
                except Exception:
                    last_raw = str(resp)

            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            web_is_answerable = bool(data.get("web_is_answerable"))
            web_is_correct = bool(data.get("web_is_correct"))
            return web_is_answerable, web_is_correct, last_raw, duration
        except Exception as e:
            # capture error text into last_raw for debugging visibility
            try:
                last_raw = json.dumps({"error": str(e)})
            except Exception:
                last_raw = str(e)
            # backoff
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                # final failure
                return None, None, last_raw, duration


def process_rows(df: pd.DataFrame, output_path: Path, model: str, resume: bool = True) -> None:
    # Prepare client
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("❌ OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... or export it in your environment.")
        sys.exit(1)
    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Determine resume set
    processed_ids = set()
    if resume and output_path.exists():
        try:
            prev = pd.read_csv(output_path, encoding="utf-8")
            if "question_id" in prev.columns:
                processed_ids = set(prev["question_id"].astype(str))
        except Exception:
            processed_ids = set()

    # Open output: write header if not existing
    write_header = not output_path.exists()
    # Determine consistent column order
    base_cols = list(df.columns)
    col_order = base_cols + [
        "web_is_answerable",
        "web_is_correct",
        "response_raw",
        "response_time_sec",
    ]
    out_f = open(output_path, "a", encoding="utf-8", newline="")
    try:
        # We'll write row-by-row using pandas to_csv with header control
        for _, row in df.iterrows():
            qid = str(row.get("question_id", ""))
            if not qid:
                continue
            if qid in processed_ids:
                continue

            question = str(row.get("question", "")).strip()
            options = parse_options(row.get("options", ""))
            correct_answer = str(row.get("correct_answer", "")).strip().upper()

            prompt = build_prompt(question, options, correct_answer)
            web_is_answerable, web_is_correct, raw_json, rtt = call_model(client, model, prompt)

            out_row = row.to_dict()
            out_row["web_is_answerable"] = web_is_answerable
            out_row["web_is_correct"] = web_is_correct
            out_row["response_raw"] = raw_json
            out_row["response_time_sec"] = rtt if rtt is not None else ""

            pd.DataFrame([out_row], columns=col_order).to_csv(out_f, index=False, header=write_header)
            if write_header:
                write_header = False
    finally:
        out_f.close()


def main() -> None:
    # Load environment variables from .env
    load_dotenv()
    parser = argparse.ArgumentParser(description="Verify answerability and correctness via web-backed LLM")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input CSV path")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume; overwrite output")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of rows to process")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    if args.no_resume and out.exists():
        out.unlink()

    if not inp.exists():
        print(f"❌ Input not found: {inp}")
        sys.exit(1)

    df = pd.read_csv(inp, encoding="utf-8")
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    # Ensure required columns exist
    for col in ["question_id", "question", "correct_answer", "options"]:
        if col not in df.columns:
            print(f"❌ Missing required column: {col}")
            sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    process_rows(df, out, args.model, resume=(not args.no_resume))
    print(f"✅ Done. Wrote: {out}")


if __name__ == "__main__":
    main()


