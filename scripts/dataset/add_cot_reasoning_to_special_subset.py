import json
import os
import time
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# OpenAI SDK (>=1.0.0)
try:
    from openai import OpenAI  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "OpenAI SDK not installed. Please run: pip install -r requirements.txt"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "finished" / "special_small_subset_for_cot.csv"
OUTPUT_CSV = BASE_DIR / "finished" / "special_small_subset_for_cot_with_reasoning.csv"

# Default model; override via env MODEL if desired
DEFAULT_MODEL = os.environ.get("MODEL", "gpt-5")


def parse_options(raw: str) -> List[str]:
    """Parse the options column which is stored as a Python-style list string.

    Uses literal_eval safely; if it fails, attempts a simple fallback.
    """
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, list):
            return [str(x) for x in value]
    except Exception:
        pass
    # Fallback: strip brackets and split conservatively on ','
    cleaned = raw.strip()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    parts = [p.strip().strip("'\"") for p in cleaned.split(",")]
    return [p for p in parts if p]


def build_prompt(question: str, options: List[str]) -> str:
    letters = [chr(ord('a') + i) for i in range(len(options))]
    option_lines = [f"{letter}) {text}" for letter, text in zip(letters, options)]
    options_block = "\n".join(option_lines)
    instruction = (
        "Think step by step and explain why the correct answer is correct and the others are wrong. "
        "Provide two versions of the reasoning: one in English and one in Bengali. "
        "Each reasoning should end with the line: Final Answer: (X) where X is the option letter."
    )
    return (
        f"Question:\n{question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Instruction:\n{instruction}\n"
    )


def get_response_json(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 800,
    retry: int = 3,
    sleep_seconds: float = 2.0,
) -> Dict:
    """Call Chat Completions with JSON output and return parsed JSON dict."""
    system_msg = (
        "You are an expert test-solver. Output ONLY valid JSON with keys: "
        "reasoning_en (string), reasoning_bn (string), final_answer (single lowercase letter), "
        "explanations (object keyed by option letter). No extra text."
    )

    last_err: Optional[Exception] = None
    for _ in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                # temperature=0.2,
                # max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(sleep_seconds)
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unknown error calling OpenAI Chat Completions API")


def label_for_index(idx: int) -> str:
    return chr(ord('a') + idx)


def process_and_stream_to_csv(df: pd.DataFrame, client: OpenAI, model: str, output_path: Path) -> None:
    original_columns: List[str] = list(df.columns)
    extra_columns: List[str] = [
        "model_reasoning_json",
        "model_final_answer",
        "model_reasoning_en",
        "model_reasoning_bn",
        "model_is_correct",
        "answers_match",
    ]
    all_columns: List[str] = original_columns + extra_columns

    # Create empty CSV with headers first for real-time visibility
    pd.DataFrame(columns=all_columns).to_csv(output_path, index=False, encoding="utf-8")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CoT reasoning"):
        question_text = str(row.get("question", "")).strip()
        raw_options = str(row.get("options", "")).strip()
        correct_answer = str(row.get("correct_answer", "")).strip().lower()

        options = parse_options(raw_options)

        # Default outputs
        response_obj: Dict = {}
        final_answer: Optional[str] = None
        reasoning_en: Optional[str] = None
        reasoning_bn: Optional[str] = None
        is_correct: Optional[bool] = None

        if options:
            try:
                prompt = build_prompt(question_text, options)
                response_obj = get_response_json(client, model, prompt)
                final_answer = str(response_obj.get("final_answer", "")).strip().lower() or None
                reasoning_en = str(response_obj.get("reasoning_en", "")).strip() or None
                reasoning_bn = str(response_obj.get("reasoning_bn", "")).strip() or None
                if final_answer and correct_answer:
                    is_correct = (final_answer == correct_answer)
            except Exception:
                # Leave defaults as None if call fails; still write the row
                pass

        answers_match = is_correct if is_correct is not None else None

        row_out: Dict[str, Optional[str]] = {col: row.get(col, None) for col in original_columns}
        row_out.update({
            "model_reasoning_json": json.dumps(response_obj, ensure_ascii=False) if response_obj else None,
            "model_final_answer": final_answer,
            "model_reasoning_en": reasoning_en,
            "model_reasoning_bn": reasoning_bn,
            "model_is_correct": is_correct,
            "answers_match": answers_match,
        })

        # Append this single row
        pd.DataFrame([row_out], columns=all_columns).to_csv(
            output_path, mode="a", header=False, index=False, encoding="utf-8"
        )


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY not set. Please export it or add to .env before running this script."
        )
        return

    if not INPUT_CSV.exists():
        print(f"Input CSV not found: {INPUT_CSV}")
        return

    model = DEFAULT_MODEL
    print(f"Using model: {model}")
    client = OpenAI()

    df = pd.read_csv(INPUT_CSV)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    process_and_stream_to_csv(df, client, model, OUTPUT_CSV)
    print(f"Updated (streamed): {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


