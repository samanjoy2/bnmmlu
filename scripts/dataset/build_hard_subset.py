import pandas as pd
from pathlib import Path
import math


FINISHED_DIR = Path("finished")
QUESTIONS_CSV = Path("finished/merged_all_questions_dedup.csv")
OUTPUT_HARD_CSV = FINISHED_DIR / "hard_subset_by_llm_failures.csv"
OUTPUT_SUMMARY_TXT = FINISHED_DIR / "hard_subset_summary.txt"


def _derive_model_name_from_filename(p: Path) -> str:
    stem = p.stem  # e.g., llm_test_results_modelname_here
    prefix = "llm_test_results_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def load_finished_results() -> pd.DataFrame:
    """Load all finished llm_test_results_*.csv and return concatenated DataFrame with model_name."""
    csv_paths = sorted(FINISHED_DIR.glob("llm_test_results_*.csv"))
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, dtype={"question_id": str}, low_memory=False)
            if "question_id" in df.columns and "is_correct" in df.columns:
                model_series = (
                    df["model_name"] if "model_name" in df.columns else pd.Series([_derive_model_name_from_filename(path)] * len(df))
                )
                frames.append(
                    pd.DataFrame({
                        "question_id": df["question_id"].astype(str),
                        "is_correct": df["is_correct"],
                        "model_name": model_series.astype(str),
                        "source_file": path.name,
                    })
                )
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["question_id", "is_correct", "model_name", "source_file"]) 
    return pd.concat(frames, ignore_index=True)


def normalize_is_correct(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    return as_str.isin(truthy).astype(int)


def select_hardest_by_subdomain(df_joined: pd.DataFrame, percent: float = 0.2) -> pd.DataFrame:
    """For each subdomain, select top questions with highest failure ratio.
    At least ceil(percent * num_questions) per subdomain, minimum 1.
    """
    # failure_ratio = 1 - mean(is_correct) per question across LLMs
    question_stats = (
        df_joined.groupby(["subdomain_name", "question_id"], as_index=False)
        .agg(n_models=("is_correct_num", "size"), correct_sum=("is_correct_num", "sum"))
    )
    question_stats["failure_ratio"] = 1.0 - (question_stats["correct_sum"] / question_stats["n_models"])

    # pick hardest per subdomain
    hardest_rows = []
    for subdomain, group in question_stats.groupby("subdomain_name"):
        group_sorted = group.sort_values(["failure_ratio", "n_models"], ascending=[False, False])
        n_total = len(group_sorted)
        n_pick = max(1, math.ceil(percent * n_total))
        hardest_rows.append(group_sorted.head(n_pick))
    hardest = pd.concat(hardest_rows, ignore_index=True)
    return hardest


def main() -> None:
    # Load finished results and normalize correctness
    results = load_finished_results()
    if results.empty:
        print("No finished results found in 'finished/'.")
        return
    results["is_correct_num"] = normalize_is_correct(results["is_correct"])  # 0/1

    # Select top-10 models by overall accuracy (weighted by all rows in finished/)
    model_stats = (
        results.groupby("model_name", as_index=False)
        .agg(n_rows=("is_correct_num", "size"), accuracy=("is_correct_num", "mean"))
        .sort_values(["accuracy", "n_rows"], ascending=[False, False])
    )
    top_models = model_stats.head(10)["model_name"].tolist()
    results = results[results["model_name"].isin(top_models)].copy()

    # Load questions to get subdomain_name and content
    questions = pd.read_csv(
        QUESTIONS_CSV,
        dtype={"Unique_Serial": str},
        low_memory=False,
    )

    # Join on question id
    joined = results.merge(
        questions[["Unique_Serial", "subdomain_name", "subject", "question", "correct_answer", "options"]],
        left_on="question_id",
        right_on="Unique_Serial",
        how="inner",
    )

    # Select hardest per subdomain
    hardest = select_hardest_by_subdomain(joined, percent=0.3)

    # Attach question text and subject for output
    hardest = hardest.merge(
        questions[["Unique_Serial", "subdomain_name", "subject", "question", "correct_answer", "options"]],
        left_on="question_id",
        right_on="Unique_Serial",
        how="left",
        suffixes=("", "_q"),
    )

    # Write outputs
    OUTPUT_HARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    hardest_columns = [
        "subdomain_name",
        "question_id",
        "failure_ratio",
        "n_models",
        "subject",
        "question",
        "correct_answer",
        "options",
    ]
    (hardest[hardest_columns]
     .sort_values(["subdomain_name", "failure_ratio"], ascending=[True, False])
     .to_csv(OUTPUT_HARD_CSV, index=False, encoding="utf-8"))

    # Summary
    lines = []
    lines.append("Hard subset generated by LLM failure aggregation")
    lines.append(f"Finished dir: {FINISHED_DIR}")
    lines.append(f"Questions file: {QUESTIONS_CSV.name}")
    lines.append("Top models used (by accuracy):")
    for _, r in model_stats.head(10).iterrows():
        lines.append(f"- {r['model_name']}: acc={r['accuracy']*100:.2f}% (n={int(r['n_rows'])})")
    lines.append(f"Total result rows loaded: {len(results):,}")
    lines.append(f"Unique questions in results: {results['question_id'].nunique():,}")
    lines.append(f"Total hardest picked: {len(hardest):,}")
    lines.append("")
    # Per subdomain counts
    counts = hardest.groupby("subdomain_name", as_index=False).size().rename(columns={"size": "count"})
    for _, row in counts.sort_values("subdomain_name").iterrows():
        lines.append(f"- {row['subdomain_name']}: {int(row['count'])}")
    OUTPUT_SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote hard subset to: {OUTPUT_HARD_CSV}")
    print(f"Wrote summary to: {OUTPUT_SUMMARY_TXT}")


if __name__ == "__main__":
    main()


