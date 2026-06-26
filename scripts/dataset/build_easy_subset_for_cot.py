import math
from pathlib import Path
from typing import List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FINISHED_DIR = BASE_DIR / "finished"
QUESTIONS_CSV = BASE_DIR / "merged_all_questions_with_subdomains_renamed.csv"
OUTPUT_EASY_CSV = FINISHED_DIR / "special_small_subset_for_cot.csv"


def normalize_is_correct(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    return as_str.isin(truthy).astype(int)


def discover_main_model_files(csv_paths: List[Path]) -> List[Path]:
    """Heuristically keep larger/main models from finished/ results.

    Preference:
      - include filenames indicating 32b, 12b, 9b, 8b
      - include well-known providers: deepseek, grok, gemini_2_5_flash_lite, claude_3_haiku
      - exclude obviously small: 1b, 3b, nano
    Fallback to keeping all if heuristic yields none.
    """
    keep: List[Path] = []
    for path in csv_paths:
        name = path.name.lower()
        include = (
            ("32b" in name)
            or ("12b" in name)
            or ("-9b" in name or "9b" in name)
            or ("8b" in name)
            or ("deepseek" in name)
            or ("grok" in name)
            or ("gemini_2_5_flash_lite" in name)
            or ("claude_3_haiku" in name)
        )
        exclude = ("1b" in name) or ("3b" in name) or ("nano" in name)
        if include and not exclude:
            keep.append(path)
    if not keep:
        # If heuristic returns nothing, use all available as a safe fallback
        keep = list(csv_paths)
    return keep


def load_selected_results() -> pd.DataFrame:
    """Load selected finished llm_test_results_*.csv and return concatenated DataFrame."""
    csv_paths = sorted(FINISHED_DIR.glob("llm_test_results_*.csv"))
    selected_paths = discover_main_model_files(csv_paths)

    frames = []
    for path in selected_paths:
        try:
            df = pd.read_csv(path, dtype={"question_id": str}, low_memory=False)
        except Exception:
            continue
        expected_cols = {"question_id", "is_correct"}
        if not expected_cols.issubset(set(df.columns)):
            continue
        df = df[["question_id", "is_correct"]].copy()
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["question_id", "is_correct", "source_file"])  # empty
    return pd.concat(frames, ignore_index=True)


def compute_question_accuracy(results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-question accuracy across selected models."""
    results = results.copy()
    results["is_correct_num"] = normalize_is_correct(results["is_correct"])  # 0/1
    stats = (
        results.groupby("question_id", as_index=False)
        .agg(n_models=("is_correct_num", "size"), correct_sum=("is_correct_num", "sum"))
    )
    stats["accuracy_ratio"] = stats["correct_sum"] / stats["n_models"].clip(lower=1)
    return stats


def pick_top5_per_subdomain(question_stats: pd.DataFrame, questions: pd.DataFrame, min_accuracy: float = 0.8) -> pd.DataFrame:
    """Join stats with questions and pick top 5 per subdomain by accuracy and coverage.

    Strategy:
      1) Prefer items with accuracy >= min_accuracy
      2) Sort by accuracy desc, then n_models desc
      3) If fewer than 5 meet the threshold for a subdomain, fill from the remainder
    """
    joined = question_stats.merge(
        questions[[
            "Unique_Serial",
            "subject",
            "subdomain_name",
            "question",
            "correct_answer",
            "options",
        ]],
        left_on="question_id",
        right_on="Unique_Serial",
        how="inner",
    )

    picked_frames: List[pd.DataFrame] = []
    for subdomain, group in joined.groupby("subdomain_name", as_index=False):
        group_sorted = group.sort_values(["accuracy_ratio", "n_models"], ascending=[False, False])
        primary = group_sorted[group_sorted["accuracy_ratio"] >= min_accuracy].head(5)
        if len(primary) < 5:
            needed = 5 - len(primary)
            remainder = group_sorted[~group_sorted.index.isin(primary.index)].head(needed)
            primary = pd.concat([primary, remainder], ignore_index=True)
        picked_frames.append(primary)

    if not picked_frames:
        return pd.DataFrame(columns=[
            "subdomain_name",
            "question_id",
            "accuracy_ratio",
            "n_models",
            "question",
            "correct_answer",
            "options",
        ])

    picked = pd.concat(picked_frames, ignore_index=True)
    picked = picked[[
        "subdomain_name",
        "question_id",
        "accuracy_ratio",
        "n_models",
        "question",
        "correct_answer",
        "options",
    ]].copy()
    return picked


def main() -> None:
    results = load_selected_results()
    if results.empty:
        print("No suitable finished results found in 'finished/'.")
        return

    questions = pd.read_csv(
        QUESTIONS_CSV,
        dtype={"Unique_Serial": str},
        low_memory=False,
    )

    stats = compute_question_accuracy(results)
    easy_subset = pick_top5_per_subdomain(stats, questions, min_accuracy=0.8)

    OUTPUT_EASY_CSV.parent.mkdir(parents=True, exist_ok=True)
    easy_subset.sort_values(["subdomain_name", "accuracy_ratio", "n_models"], ascending=[True, False, False]).to_csv(
        OUTPUT_EASY_CSV, index=False, encoding="utf-8"
    )
    print(f"Wrote easy CoT subset to: {OUTPUT_EASY_CSV}")


if __name__ == "__main__":
    main()


