#!/usr/bin/env python3
"""
Create a metadata CSV skeleton for compute info from a usage averages CSV.

Reads the usage averages file (must contain a 'model_name' column) and writes
Usage Proper Documentation/model_compute_metadata.csv with columns:

  model_name,family,params_b,train_tokens_b,notes

The script derives a 'family' label using deterministic rules from the model
name and leaves params_b and train_tokens_b blank for you to fill with factual
values. This avoids making assumptions or fetching external data.

Example:
  python make_metadata_from_usage.py \
    --usage-csv "Usage Proper Documentation/llm_usage_averages_by_model_0-shot Direct (Non-Reasoning).csv" \
    --out "Usage Proper Documentation/model_compute_metadata.csv"
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Set


def _read_model_names(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "model_name" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'model_name' column.")
        names: Set[str] = set()
        for row in reader:
            name = (row.get("model_name") or "").strip()
            if name:
                names.add(name)
        return names


def _derive_family(model_name: str) -> str:
    n = model_name.strip()
    low = n.lower()

    # Common families
    if low.startswith("qwen/") or low.startswith("qwen3") or "qwen3" in low:
        return "Qwen3"
    if low.startswith("meta-llama/") or "llama-3" in low:
        return "Meta-Llama-3"
    if low.startswith("google/gemma") or "gemma-3" in low:
        return "Gemma-3"
    if low.startswith("unsloth/"):
        # Try to infer from the remainder
        rest = low.split("unsloth/", 1)[1]
        if rest.startswith("gemma-3"):
            return "Gemma-3 (Unsloth)"
        if rest.startswith("meta-llama-3") or rest.startswith("llama-3"):
            return "Meta-Llama-3 (Unsloth)"
        if rest.startswith("qwen3") or "qwen" in rest:
            return "Qwen3 (Unsloth)"
        return "Unsloth"
    if low.startswith("banglallm/") or "banglallama" in low:
        return "BanglaLLama"
    if low.startswith("md-nishat-008/") or "tigerllm" in low:
        return "TigerLLM"
    if low.startswith("hishab/") or "titulm" in low:
        return "Titulm-Llama"
    if low.startswith("deepseek-chat") or "deepseek" in low:
        return "DeepSeek"
    if low.startswith("gpt-") or low.startswith("gpt_"):
        return "OpenAI GPT"
    if low.startswith("gemini") or low.startswith("google/gemini"):
        return "Gemini"
    if low.startswith("grok"):
        return "Grok"
    # Default: use the namespace prefix if present
    if "/" in n:
        return n.split("/", 1)[0]
    return n


def _write_metadata(names: Iterable[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "family", "params_b", "train_tokens_b", "notes"],
        )
        writer.writeheader()
        for name in sorted(set(names)):
            writer.writerow(
                {
                    "model_name": name,
                    "family": _derive_family(name),
                    "params_b": "",
                    "train_tokens_b": "",
                    "notes": "",
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Create model compute metadata skeleton from usage averages CSV.")
    ap.add_argument("--usage-csv", type=Path, required=True, help="Path to usage averages CSV (with model_name column)")
    ap.add_argument("--out", type=Path, required=True, help="Output metadata CSV path")
    args = ap.parse_args()

    names = _read_model_names(args.usage_csv)
    if not names:
        raise SystemExit("No model_name rows found in usage CSV.")
    _write_metadata(names, args.out)
    print(f"[done] Wrote metadata skeleton: {args.out}")


if __name__ == "__main__":
    main()

