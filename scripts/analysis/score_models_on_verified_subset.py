import argparse
from pathlib import Path
from typing import List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FINISHED_DIR = BASE_DIR / "finished"
# VERIFIED_CSV = FINISHED_DIR / "verified_dataset_gpt_5_mini_ACCEPTED.csv"
VERIFIED_CSV = FINISHED_DIR / "verified_dataset_overlap_with_hard_subset.csv"
OUTPUT_CSV = FINISHED_DIR / "verified_subset_model_scores.csv"
OUTPUT_TXT = FINISHED_DIR / "verified_subset_model_scores.txt"


def normalize_is_correct(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    return as_str.isin(truthy).astype(int)


def list_result_files(include_news: bool = True) -> List[Path]:
    files: List[Path] = []
    files += sorted(FINISHED_DIR.glob("llm_test_results_*.csv"))
    if include_news:
        news_dir = FINISHED_DIR / "news"
        if news_dir.exists():
            files += sorted(news_dir.glob("llm_test_results_*.csv"))
    # Remove the verified_dataset files if any match the prefix
    files = [p for p in files if not p.name.startswith("verified_dataset_")]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-model scores on the verified subset")
    parser.add_argument("--no_news", action="store_true", help="Do not include files under finished/news/")
    parser.add_argument("--use_filename_for_model", action="store_true", help="Derive model_name from filename instead of CSV column")
    args = parser.parse_args()

    if not VERIFIED_CSV.exists():
        raise SystemExit(f"Verified dataset not found: {VERIFIED_CSV}")

    df_verified = pd.read_csv(VERIFIED_CSV, dtype={"question_id": str}, low_memory=False)
    verified_ids = set(df_verified["question_id"].astype(str))
    n_target = len(verified_ids)

    result_files = list_result_files(include_news=not args.no_news)
    if not result_files:
        raise SystemExit("No llm_test_results_*.csv files found under finished/ (and news/)")

    per_file_rows: List[pd.DataFrame] = []
    all_details: List[pd.DataFrame] = []
    for p in result_files:
        try:
            df = pd.read_csv(p, dtype={"question_id": str}, low_memory=False)
        except Exception:
            continue
        needed = {"question_id", "is_correct"}
        if not needed.issubset(df.columns):
            continue
        # Filter to verified subset
        df = df[df["question_id"].astype(str).isin(verified_ids)].copy()
        if df.empty:
            continue
        # Decide model name
        if args.use_filename_for_model or ("model_name" not in df.columns) or (df["model_name"].astype(str).nunique() != 1):
            # Derive from filename (strip prefix)
            stem = p.stem
            prefix = "llm_test_results_"
            model_name = stem[len(prefix):] if stem.startswith(prefix) else stem
            df["model_name"] = model_name
        else:
            df["model_name"] = df["model_name"].astype(str)

        df["is_correct_num"] = normalize_is_correct(df["is_correct"])  # 0/1
        # Keep detailed rows for dedup across files later
        all_details.append(df[["question_id", "model_name", "is_correct_num"]].assign(source_file=p.name))

        grp = (
            df.groupby("model_name", as_index=False)
            .agg(n=("is_correct_num", "size"), accuracy=("is_correct_num", "mean"))
        )
        grp["source_file"] = p.name
        per_file_rows.append(grp)

    if not per_file_rows:
        raise SystemExit("No overlapping results with the verified subset were found.")

    per_file = pd.concat(per_file_rows, ignore_index=True)

    # Deduplicate across files per (model_name, question_id) so each question counts once per model
    details = pd.concat(all_details, ignore_index=True)
    details_dedup = (
        details.sort_values(["model_name", "question_id", "source_file"])  # deterministic keep
        .drop_duplicates(["model_name", "question_id"], keep="first")
    )
    overall = (
        details_dedup.groupby("model_name", as_index=False)
        .agg(n=("question_id", "nunique"), accuracy=("is_correct_num", "mean"))
        .sort_values(["accuracy", "n"], ascending=[False, False])
    )

    # Save outputs
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    per_file.to_csv(OUTPUT_CSV.with_name("verified_subset_model_scores_by_file.csv"), index=False, encoding="utf-8")
    overall.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # Text summary
    lines: List[str] = []
    lines.append("Model scores on verified subset")
    lines.append("")
    lines.append(f"Verified dataset: {VERIFIED_CSV.name} (unique questions: {n_target:,})")
    lines.append(f"Result files considered: {len(result_files)}")
    lines.append("")
    lines.append("Top models (accuracy, n):")
    for _, r in overall.head(25).iterrows():
        lines.append(f"- {r['model_name']}: {r['accuracy']*100:.2f}% (n={int(r['n'])})")
    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote per-file scores: {OUTPUT_CSV.with_name('verified_subset_model_scores_by_file.csv')}")
    print(f"Wrote overall scores: {OUTPUT_CSV}")
    print(f"Wrote text summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()


