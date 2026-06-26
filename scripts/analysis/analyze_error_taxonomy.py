#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FINISHED_DIR = BASE_DIR / "finished"
OUTPUT_DIR = FINISHED_DIR / "error_taxonomy"


@dataclass(frozen=True)
class PromptMeta:
    prompt_type: Optional[str]  # "cot" | "direct" | None
    shots: Optional[int]        # 0 | 5 | None
    reasoning: Optional[str]    # "thinkOn" | "thinkOff" | None
    source_file: str


HTML_PATTERN = re.compile(r"<!doctype|<html", re.I)
RATE_LIMIT_PATTERN = re.compile(r"(rate limit|429|too many requests)", re.I)
TIMEOUT_PATTERN = re.compile(r"(timeout|timed out|connection.*(reset|closed|refused))", re.I)

# Bengali digits ১,২,৩,৪ and Arabic numerals 1,2,3,4
NUM_TO_OPTION: Dict[str, str] = {
    "1": "A", "2": "B", "3": "C", "4": "D",
    "১": "A", "২": "B", "৩": "C", "৪": "D",
}

OPTION_PATTERN = re.compile(r"\b([ABCD])\b")
NUM_PATTERN = re.compile(r"\b([1234১২৩৪])\b")


def list_result_files() -> List[Path]:
    files: List[Path] = []
    files += sorted(BASE_DIR.glob("llm_test_results_*.csv"))
    if FINISHED_DIR.exists():
        files += sorted(FINISHED_DIR.glob("llm_test_results_*.csv"))
        news_dir = FINISHED_DIR / "news"
        if news_dir.exists():
            files += sorted(news_dir.glob("llm_test_results_*.csv"))
    # De-duplicate by path string
    dedup: Dict[str, Path] = {str(p): p for p in files}
    return list(dedup.values())


def parse_prompt_meta(path: Path) -> PromptMeta:
    stem = path.stem  # e.g., llm_test_results_gpt_5__thinkOff__fs5__cot_json
    parts = stem.split("__")
    prompt_type: Optional[str] = None
    shots: Optional[int] = None
    reasoning: Optional[str] = None
    for part in parts:
        low = part.lower()
        if low.startswith("fs") and low[2:].isdigit():
            shots = int(low[2:])
        if "thinkon" in low:
            reasoning = "thinkOn"
        if "thinkoff" in low:
            reasoning = "thinkOff"
        if "cot" in low:
            prompt_type = "cot"
        if "direct" in low:
            prompt_type = "direct"
    return PromptMeta(prompt_type=prompt_type, shots=shots, reasoning=reasoning, source_file=path.name)


def coalesce_raw_text(row: pd.Series) -> str:
    for col in ("raw_output", "original_llm_response", "raw_text"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return str(row[col])
    return ""


def maybe_letter_from_text(text: str) -> Optional[str]:
    # Look for a single clear option letter; fall back to a single number mapping
    letters = set(m.group(1) for m in OPTION_PATTERN.finditer(text))
    if len(letters) == 1:
        letter = next(iter(letters))
        if letter in {"A", "B", "C", "D"}:
            return letter
    nums = set(m.group(1) for m in NUM_PATTERN.finditer(text))
    mapped = set(NUM_TO_OPTION.get(n) for n in nums if NUM_TO_OPTION.get(n))
    if len(mapped) == 1:
        return next(iter(mapped))
    return None


def classify_error(row: pd.Series) -> str:
    is_correct = str(row.get("is_correct", "")).strip().lower()
    is_correct_truthy = is_correct in {"true", "1", "yes", "y", "t"}
    llm_answer = str(row.get("llm_answer", "")).strip().upper()
    correct_answer = str(row.get("correct_answer", "")).strip().upper()
    raw_text = coalesce_raw_text(row)

    if is_correct_truthy:
        return "Correct"

    # From here, the example is not correct; identify failure category
    if llm_answer in {"A", "B", "C", "D"} and correct_answer in {"A", "B", "C", "D"}:
        if llm_answer != correct_answer:
            return "Content: wrong option"

    if llm_answer == "" or llm_answer not in {"A", "B", "C", "D"}:
        # Infrastructure / formatting errors
        if raw_text:
            if HTML_PATTERN.search(raw_text or ""):
                return "Infrastructure: HTML/router response"
            if RATE_LIMIT_PATTERN.search(raw_text or ""):
                return "Infrastructure: API rate limit"
            if TIMEOUT_PATTERN.search(raw_text or ""):
                return "Infrastructure: timeout/connection"
            inferred = maybe_letter_from_text(raw_text)
            if inferred is not None and llm_answer == "":
                return "Extraction: parser failed to extract option"
            # Non-option free text present
            return "Formatting: non-option output"
        else:
            return "No response/empty output"

    # Fallback
    return "Uncategorized"


def model_family_from_name(model_name: str, fallback_from_filename: str) -> str:
    name = (model_name or fallback_from_filename).lower()
    for key in ("gpt", "qwen", "llama", "gemini", "grok", "deepseek", "unsloth", "titulm", "claude", "haiku"):
        if key in name:
            return key
    return "other"


def load_all_results(files: Iterable[Path]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_csv(p, dtype={"question_id": str}, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        # Ensure essential columns exist
        for col in ("question_id", "correct_answer", "llm_answer", "model_name", "timestamp", "is_correct"):
            if col not in df.columns:
                # Create empty columns if missing
                df[col] = None
        meta = parse_prompt_meta(p)
        df["prompt_type"] = meta.prompt_type
        df["shots"] = meta.shots
        df["reasoning"] = meta.reasoning
        df["source_file"] = meta.source_file
        # Normalize model name if absent or mixed
        if ("model_name" not in df.columns) or (df["model_name"].astype(str).nunique() != 1):
            stem = p.stem
            prefix = "llm_test_results_"
            inferred = stem[len(prefix):] if stem.startswith(prefix) else stem
            df["model_name"] = inferred
        df["model_family"] = df["model_name"].apply(lambda m: model_family_from_name(str(m), p.stem))
        # Classify error type
        df["error_type"] = df.apply(classify_error, axis=1)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def join_question_metadata(df: pd.DataFrame) -> pd.DataFrame:
    qcsv = BASE_DIR / "merged_all_questions_with_subdomains_renamed.csv"
    if not qcsv.exists():
        return df
    qdf = pd.read_csv(qcsv, dtype={"Unique_Serial": str, "question_id": str}, low_memory=False, encoding="utf-8")
    # Attempt to locate the id column
    id_col = None
    for c in ("question_id", "Unique_Serial", "unique_serial", "id"):
        if c in qdf.columns:
            id_col = c
            break
    if id_col is None:
        return df
    qdf = qdf.rename(columns={id_col: "question_id"})
    keep_cols = [c for c in qdf.columns if c in {"question_id", "question", "options", "correct_answer", "domain", "subdomain"}]
    qdf = qdf[keep_cols].copy()
    return df.merge(qdf, on="question_id", how="left", suffixes=("", "_q"))


def write_breakdowns(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Overall taxonomy distribution and accuracy
    overall = (
        df.assign(is_correct_num=df["is_correct"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"}).astype(int))
        .groupby("error_type", as_index=False)
        .agg(count=("question_id", "size"), accuracy=("is_correct_num", "mean"))
        .sort_values(["count"], ascending=False)
    )
    overall.to_csv(OUTPUT_DIR / "taxonomy_overall.csv", index=False, encoding="utf-8")

    by_model = (
        df.groupby(["model_name", "error_type"], as_index=False)
        .agg(count=("question_id", "size"))
        .sort_values(["model_name", "count"], ascending=[True, False])
    )
    by_model.to_csv(OUTPUT_DIR / "taxonomy_by_model.csv", index=False, encoding="utf-8")

    by_prompt = (
        df.groupby(["prompt_type", "shots", "reasoning", "error_type"], as_index=False)
        .agg(count=("question_id", "size"))
        .sort_values(["count"], ascending=False)
    )
    by_prompt.to_csv(OUTPUT_DIR / "taxonomy_by_prompt.csv", index=False, encoding="utf-8")

    by_model_prompt = (
        df.groupby(["model_name", "prompt_type", "shots", "reasoning", "error_type"], as_index=False)
        .agg(count=("question_id", "size"))
        .sort_values(["model_name", "count"], ascending=[True, False])
    )
    by_model_prompt.to_csv(OUTPUT_DIR / "taxonomy_by_model_prompt.csv", index=False, encoding="utf-8")

    # Per subdomain if available
    if "subdomain" in df.columns:
        by_subdomain = (
            df.groupby(["subdomain", "error_type"], as_index=False)
            .agg(count=("question_id", "size"))
            .sort_values(["subdomain", "count"], ascending=[True, False])
        )
        by_subdomain.to_csv(OUTPUT_DIR / "taxonomy_by_subdomain.csv", index=False, encoding="utf-8")

    # Confusion matrix among wrong answers
    wrong = df[(df["error_type"] == "Content: wrong option") & df["llm_answer"].isin(list("ABCD")) & df["correct_answer"].isin(list("ABCD"))]
    if not wrong.empty:
        cm = wrong.pivot_table(index="correct_answer", columns="llm_answer", values="question_id", aggfunc="size", fill_value=0)
        cm.to_csv(OUTPUT_DIR / "confusion_matrix.csv", encoding="utf-8")


def write_case_studies(df: pd.DataFrame, per_type: int = 25) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    needed_cols = [
        "question_id", "question", "options", "correct_answer", "llm_answer",
        "model_name", "model_family", "prompt_type", "shots", "reasoning",
        "error_type", "source_file",
    ]
    # Collect a concise raw text snippet
    df = df.copy()
    df["raw_text"] = df.apply(coalesce_raw_text, axis=1)
    df["raw_snippet"] = df["raw_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 500)
    cols = needed_cols + ["raw_snippet"]
    cols = [c for c in cols if c in df.columns]

    for etype, grp in df.groupby("error_type"):
        sample = grp.head(per_type)[cols]
        safe_name = re.sub(r"[^a-z0-9]+", "_", etype.lower()).strip("_")
        (OUTPUT_DIR / f"case_studies_{safe_name}.csv").write_text(sample.to_csv(index=False, encoding="utf-8"), encoding="utf-8")


def summarize_to_text(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    n = len(df)
    acc = df["is_correct"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"}).mean()
    lines.append(f"Total rows: {n:,}")
    lines.append(f"Overall accuracy: {acc*100:.2f}%")
    lines.append("")
    lines.append("Top error types:")
    err_counts = df["error_type"].value_counts().head(10)
    for etype, cnt in err_counts.items():
        pct = cnt / max(n, 1)
        lines.append(f"- {etype}: {cnt:,} ({pct*100:.2f}%)")
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze error taxonomy across LLM results and extract case studies")
    parser.add_argument("--per_type", type=int, default=25, help="Number of case studies per error type")
    args = parser.parse_args()

    files = list_result_files()
    if not files:
        raise SystemExit("No llm_test_results_*.csv files found in root or finished/")

    df = load_all_results(files)
    if df.empty:
        raise SystemExit("Loaded zero rows from result files")

    df = join_question_metadata(df)

    write_breakdowns(df)
    write_case_studies(df, per_type=args.per_type)
    summarize_to_text(df)
    print(f"Wrote taxonomy outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



