#!/usr/bin/env python3
"""
Aggregate per-model averages from usage CSVs.

Reads one or more usage CSV files (typically under the `usage/` folder)
and produces a single CSV with one row per model, containing the average
values for the token-related columns.

Input schema expected per row (extra columns ignored):
  question_id, model_name, timestamp, prompt_tokens, completion_tokens,
  total_tokens, reasoning_tokens, cached_prompt_tokens

Output schema:
  model_name, rows, avg_prompt_tokens, avg_completion_tokens, avg_total_tokens,
  avg_reasoning_tokens, avg_completion_plus_reasoning_tokens, avg_cached_prompt_tokens

Usage examples (run from the project root or the folder containing `usage/`):

  # Default: scans usage/llm_usage_*.csv and writes usage/llm_usage_averages_by_model.csv
  python aggregate_usage_averages.py

  # Custom input directory and output file
  python aggregate_usage_averages.py --input-dir usage --glob "llm_usage_*.csv" \
      --output usage\\llm_usage_averages_by_model.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


NUMERIC_COLUMNS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_prompt_tokens",
    # Optional per-row accuracy column from usage files (0/1)
    "accyrqcy",
]


def _iter_usage_files(input_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(input_dir.glob(pattern))


def _parse_int(value: str) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))  # tolerate accidental float strings
    except Exception:
        return None


def _normalize_model_tokencost(name: str) -> Optional[str]:
    """Map known model name variants to tokencost canonical IDs where sensible.

    Returns canonical string if a confident mapping exists; otherwise None.
    """
    s = (name or "").strip()
    low = s.lower()
    if not s:
        return None
    # Skip Claude entirely if desired (keep original grouping by default)
    if "claude" in low:
        return None
    # DeepSeek
    if low.startswith("deepseek-chat"):
        return "deepseek-chat"
    # OpenAI
    if low.startswith("gpt-4.1-nano") or low.startswith("gpt_4_1_nano"):
        return "gpt-4o-mini"
    if low.startswith("gpt-4.1-"):
        return "gpt-4o"
    if low.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"
    if low.startswith("gpt-4o"):
        return "gpt-4o"
    if low.startswith("gpt-4-turbo"):
        return "gpt-4-turbo"
    if low.startswith("gpt-3.5-turbo") or low.startswith("gpt_3_5_turbo"):
        return "gpt-3.5-turbo"
    if low.startswith("gpt-5") or low.startswith("gpt_5"):
        return "gpt-4o"
    # Gemini
    if "gemini_2_5_flash_lite" in low:
        return "gemini/gemini-1.5-flash"
    if "gemini 2.5" in low or "gemini_2_5" in low:
        return "gemini/gemini-1.5-pro"
    if low.startswith("gemini/") or low.startswith("google/gemini"):
        return s
    # Qwen Plus – leave as-is (no canonical in tokencost list yet)
    if "qwen-plus" in low or "qwen_plus" in low:
        return None
    # Everything else – keep original
    return None


def aggregate_file(
    path: Path,
    accum: Dict[str, Dict[str, float]],
    counts: Dict[str, Dict[str, int]],
    row_counts: Dict[str, int],
    *,
    normalize: Optional[str] = None,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        for row in reader:
            model = (row.get("model_name") or "").strip()
            if not model:
                continue
            key = model
            if normalize == "tokencost":
                mapped = _normalize_model_tokencost(model)
                if mapped:
                    key = mapped
            row_counts[key] = row_counts.get(key, 0) + 1
            if key not in accum:
                accum[key] = {k: 0.0 for k in NUMERIC_COLUMNS}
                counts[key] = {k: 0 for k in NUMERIC_COLUMNS}
            for col in NUMERIC_COLUMNS:
                if col not in (reader.fieldnames or []):
                    continue
                val = _parse_int(row.get(col))
                if val is None:
                    continue
                accum[key][col] += float(val)
                counts[key][col] += 1


def write_averages(
    output_path: Path,
    accum: Dict[str, Dict[str, float]],
    counts: Dict[str, Dict[str, int]],
    row_counts: Dict[str, int],
    *,
    include_rows_by_original: bool = True,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model_name",
            "rows",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "avg_total_tokens",
            "avg_reasoning_tokens",
            "avg_completion_plus_reasoning_tokens",
            "avg_cached_prompt_tokens",
            "accruaC Y AVERAGE",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # We aggregate by the keys in accum (which may be normalized)
        for model in sorted(accum.keys()):
            sums = accum.get(model, {})
            cnts = counts.get(model, {})
            out = {
                "model_name": model,
                # Note: rows reflect aggregated counts by normalized key if available
                "rows": row_counts.get(model, 0),
            }
            for col, out_col in [
                ("prompt_tokens", "avg_prompt_tokens"),
                ("completion_tokens", "avg_completion_tokens"),
                ("total_tokens", "avg_total_tokens"),
                ("reasoning_tokens", "avg_reasoning_tokens"),
                ("cached_prompt_tokens", "avg_cached_prompt_tokens"),
            ]:
                c = cnts.get(col, 0)
                if c > 0:
                    out[out_col] = round(sums.get(col, 0.0) / c, 3)
                else:
                    out[out_col] = ""
            # Accuracy average from accyrqcy (0/1)
            acc_c = cnts.get("accyrqcy", 0)
            if acc_c > 0:
                out["accruaC Y AVERAGE"] = round(sums.get("accyrqcy", 0.0) / acc_c, 4)
            else:
                out["accruaC Y AVERAGE"] = ""
            # Derived metric: avg_completion_tokens + avg_reasoning_tokens
            c_cnt = cnts.get("completion_tokens", 0)
            r_cnt = cnts.get("reasoning_tokens", 0)
            c_avg = (sums.get("completion_tokens", 0.0) / c_cnt) if c_cnt > 0 else None
            r_avg = (sums.get("reasoning_tokens", 0.0) / r_cnt) if r_cnt > 0 else None
            if c_avg is None and r_avg is None:
                out["avg_completion_plus_reasoning_tokens"] = ""
            else:
                out["avg_completion_plus_reasoning_tokens"] = round((c_avg or 0.0) + (r_avg or 0.0), 3)
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-model averages from usage CSVs.")
    parser.add_argument("--input-dir", type=Path, default=Path("usage"), help="Directory containing usage CSV files")
    parser.add_argument("--glob", type=str, default="llm_usage_*.csv", help="Glob pattern for usage files inside input-dir")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (default: usage/llm_usage_averages_by_model.csv)")
    parser.add_argument("--normalize", choices=["none", "tokencost"], default="none", help="Normalize model names before aggregating")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if args.output is None:
        output_path = (input_dir / "llm_usage_averages_by_model.csv").resolve()
    else:
        output_path = args.output.resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = list(_iter_usage_files(input_dir, args.glob))
    if not files:
        raise SystemExit(f"No usage files matched: {input_dir} / {args.glob}")

    accum: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    row_counts: Dict[str, int] = {}

    for path in files:
        aggregate_file(path, accum, counts, row_counts, normalize=(args.normalize if args.normalize != "none" else None))

    write_averages(output_path, accum, counts, row_counts)
    print(f"[done] Wrote per-model averages to: {output_path}")


if __name__ == "__main__":
    main()
