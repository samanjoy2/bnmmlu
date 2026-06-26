import argparse
import pandas as pd
from pathlib import Path


DEFAULT_RESULTS = Path("llm_test_results_gpt_4_1_nano_2025_04_14.csv")
QUESTIONS_CSV = Path("merged_all_questions_with_subdomains_renamed.csv")


def normalize_is_correct(series: pd.Series) -> pd.Series:
    """Convert the is_correct column to numeric 0/1 robustly."""
    as_str = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    return as_str.isin(truthy).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-subdomain and supercategory accuracy")
    parser.add_argument(
        "results_csv",
        nargs="?",
        default=str(DEFAULT_RESULTS),
        help="Path to results CSV (default: llm_test_results_gpt_4_1_nano_2025_04_14.csv)",
    )
    parser.add_argument(
        "--out",
        dest="out_txt",
        default=None,
        help="Optional output .txt path. If omitted, inferred from results filename.",
    )
    args = parser.parse_args()

    results_path = Path(args.results_csv)
    if args.out_txt:
        output_txt = Path(args.out_txt)
    else:
        stem = results_path.stem
        output_txt = Path(f"subdomain_accuracy_{stem}.txt")
    # Read CSVs
    df_results = pd.read_csv(
        results_path,
        dtype={"question_id": str},
        low_memory=False,
    )
    df_questions = pd.read_csv(
        QUESTIONS_CSV,
        dtype={"Unique_Serial": str},
        low_memory=False,
    )

    # Normalize correctness and compute overall accuracy across ALL results rows
    df_results["is_correct_num"] = normalize_is_correct(df_results["is_correct"])  # 0/1
    total_results = len(df_results)
    overall_accuracy = df_results["is_correct_num"].mean() if total_results else 0.0

    # Join on shared ID: question_id (results) == Unique_Serial (questions)
    df_joined = (
        df_results[["question_id", "is_correct_num"]]
        .merge(
            df_questions[["Unique_Serial", "subdomain_name"]],
            left_on="question_id",
            right_on="Unique_Serial",
            how="inner",
        )
    )

    # Compute per-subdomain accuracy
    grouped = df_joined.groupby("subdomain_name", dropna=False, as_index=False).agg(
        total=("is_correct_num", "size"),
        correct=("is_correct_num", "sum"),
    )
    grouped["accuracy"] = grouped["correct"] / grouped["total"]

    # Sort for readability
    grouped = grouped.sort_values(["subdomain_name"]).reset_index(drop=True)

    # Map each subdomain to a supercategory
    subdomain_to_super = {
        # Humanities
        "Bengali Language & Syntax": "Humanities",
        "Bengali Literature": "Humanities",
        "Bengali Poetry": "Humanities",
        "Comparative Religion": "Humanities",
        "Moral & Ethical Studies": "Humanities",
        "Critical Thinking": "Humanities",
        "Formal Logic": "Humanities",

        # Social Sciences
        "Entrepreneurship": "Social Sciences",
        "History & Culture (Bangladesh/World Basics)": "Social Sciences",
        "History & Culture (Bangladesh/World basics)": "Social Sciences",  # dataset variant
        "Banking & Investment": "Social Sciences",
        "Production & Operations": "Social Sciences",
        "Business Strategy & Management": "Social Sciences",
        "Economics": "Social Sciences",
        "Financial Accounting": "Social Sciences",
        "Corporate Finance": "Social Sciences",
        "Civics & Governance": "Social Sciences",
        "Geography": "Social Sciences",
        "Cognitive Psychology": "Social Sciences",
        "Behavioral Psychology": "Social Sciences",
        "Social Work & Welfare": "Social Sciences",

        # STEM
        "Agricultural Sciences": "STEM",
        "Cell Biology & Genetics": "STEM",
        "Algebra & Number Theory": "STEM",
        "Physical & Analytical Chemistry": "STEM",
        "Human Biology & Anatomy": "STEM",
        "Inorganic Chemistry": "STEM",
        "Thermodynamics & Electromagnetism": "STEM",
        "Conceptual Physics (basic laws)": "STEM",
        "Organic Chemistry": "STEM",
        "General Science": "STEM",
        "Mechanics": "STEM",
        "Statistics: Probability & Inference": "STEM",
        "Relativity & Modern Physics": "STEM",
        "Programming & Algorithms": "STEM",
        "Calculus & Analysis": "STEM",
        "Networking & Security": "STEM",
        "Elementary Mathematics": "STEM",
        "AI & Data Science Basics": "STEM",
        
        # Others
        "Miscellaneous GK (sports, arts, pop culture)": "Others",
        "Global Facts & Current Affairs": "Others",
        "(NA)": "Others",
    }

    df_joined["supercategory"] = (
        df_joined["subdomain_name"].map(subdomain_to_super).fillna("Others")
    )

    grouped_super = df_joined.groupby("supercategory", dropna=False, as_index=False).agg(
        total=("is_correct_num", "size"),
        correct=("is_correct_num", "sum"),
    )
    grouped_super["accuracy"] = grouped_super["correct"] / grouped_super["total"]
    grouped_super = grouped_super.sort_values(["supercategory"]).reset_index(drop=True)

    # Prepare report text
    lines = []
    lines.append("Subdomain accuracy report")
    lines.append("")
    lines.append(f"Results file: {results_path.name}")
    lines.append(f"Questions file: {QUESTIONS_CSV.name}")
    lines.append(f"Matched rows (for subdomain calc): {len(df_joined):,}")
    lines.append(f"Total result rows: {total_results:,}")
    lines.append("")
    lines.append(
        f"Overall accuracy (all {total_results:,}): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)"
    )
    lines.append("")
    lines.append("Per-supercategory accuracy:")
    for _, row in grouped_super.iterrows():
        cat = row["supercategory"] if pd.notna(row["supercategory"]) else "(NA)"
        total = int(row["total"])  # type: ignore[call-arg]
        correct = int(row["correct"])  # type: ignore[call-arg]
        acc = float(row["accuracy"])  # type: ignore[call-arg]
        lines.append(
            f"- {cat}: {acc:.4f} ({acc*100:.2f}%)  [{correct:,}/{total:,}]"
        )
    lines.append("")
    lines.append("Per-subdomain accuracy:")
    for _, row in grouped.iterrows():
        sub = row["subdomain_name"] if pd.notna(row["subdomain_name"]) else "(NA)"
        total = int(row["total"])  # type: ignore[call-arg]
        correct = int(row["correct"])  # type: ignore[call-arg]
        acc = float(row["accuracy"])  # type: ignore[call-arg]
        lines.append(
            f"- {sub}: {acc:.4f} ({acc*100:.2f}%)  [{correct:,}/{total:,}]"
        )

    output_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {output_txt.resolve()}")


if __name__ == "__main__":
    main()


