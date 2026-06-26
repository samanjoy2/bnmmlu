import argparse
from pathlib import Path
from typing import List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FINISHED_DIR = BASE_DIR / "finished"
VERIFIED_CSV = FINISHED_DIR / "verified_dataset_overlap_with_hard_subset.csv"
QUESTIONS_CSV = BASE_DIR / "merged_all_questions_with_subdomains_renamed.csv"
OUTPUT_CSV = FINISHED_DIR / "verified_subset_model_scores_by_domain.csv"


def normalize_is_correct(series: pd.Series) -> pd.Series:
    as_str = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    return as_str.isin(truthy).astype(int)


def subdomain_to_super_map() -> dict:
    # Mirrors mapping used in run_subdomain_accuracy_for_news.py
    return {
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
        "History & Culture (Bangladesh/World basics)": "Social Sciences",
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


def list_result_files(include_news: bool = True, only_news: bool = False) -> List[Path]:
    files: List[Path] = []
    news_dir = FINISHED_DIR / "news"
    if only_news:
        if news_dir.exists():
            files += sorted(news_dir.glob("llm_test_results_*.csv"))
    else:
        files += sorted(FINISHED_DIR.glob("llm_test_results_*.csv"))
        if include_news and news_dir.exists():
            files += sorted(news_dir.glob("llm_test_results_*.csv"))
    files = [p for p in files if not p.name.startswith("verified_dataset_")]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-domain (STEM/SocSci/Humanities/Others) accuracies on verified-overlap set")
    parser.add_argument("--no_news", action="store_true", help="Do not include files under finished/news/")
    parser.add_argument("--only_news", action="store_true", help="Only include files under finished/news/ (ignore finished root)")
    parser.add_argument("--use_filename_for_model", action="store_true", help="Derive model_name from filename instead of CSV column")
    args = parser.parse_args()

    if not VERIFIED_CSV.exists():
        raise SystemExit(f"Verified dataset not found: {VERIFIED_CSV}")
    if not QUESTIONS_CSV.exists():
        raise SystemExit(f"Questions CSV (with subdomains) not found: {QUESTIONS_CSV}")

    df_verified = pd.read_csv(VERIFIED_CSV, dtype={"question_id": str}, low_memory=False)
    verified_ids = set(df_verified["question_id"].astype(str))

    # Load subdomain mapping for the full question set
    df_questions = pd.read_csv(QUESTIONS_CSV, dtype={"Unique_Serial": str}, low_memory=False)
    df_questions = df_questions[["Unique_Serial", "subdomain_name"]].rename(columns={"Unique_Serial": "question_id"})

    sub_to_super = subdomain_to_super_map()

    result_files = list_result_files(include_news=not args.no_news, only_news=args.only_news)
    if not result_files:
        raise SystemExit("No llm_test_results_*.csv files found under finished/ (and news/)")

    per_model_domain_rows: List[pd.DataFrame] = []

    for p in result_files:
        try:
            df = pd.read_csv(p, dtype={"question_id": str}, low_memory=False)
        except Exception:
            continue
        needed = {"question_id", "is_correct"}
        if not needed.issubset(df.columns):
            continue

        # Filter to verified overlap set
        df = df[df["question_id"].astype(str).isin(verified_ids)].copy()
        if df.empty:
            continue

        # Determine model name
        if args.use_filename_for_model or ("model_name" not in df.columns) or (df["model_name"].astype(str).nunique() != 1):
            stem = p.stem
            prefix = "llm_test_results_"
            model_name = stem[len(prefix):] if stem.startswith(prefix) else stem
            df["model_name"] = model_name
        else:
            df["model_name"] = df["model_name"].astype(str)

        df["is_correct_num"] = normalize_is_correct(df["is_correct"])  # 0/1

        # Join with subdomain and map to supercategory
        df_joined = (
            df[["question_id", "model_name", "is_correct_num"]]
            .merge(df_questions, on="question_id", how="left")
        )
        df_joined["supercategory"] = df_joined["subdomain_name"].map(sub_to_super).fillna("Others")

        # Group by model and domain
        grouped = (
            df_joined.groupby(["model_name", "supercategory"], as_index=False)
            .agg(n=("is_correct_num", "size"), accuracy=("is_correct_num", "mean"))
        )
        grouped["source_file"] = p.name
        per_model_domain_rows.append(grouped)

    if not per_model_domain_rows:
        raise SystemExit("No overlapping results with the verified subset were found.")

    per_model_domain = pd.concat(per_model_domain_rows, ignore_index=True)

    # Aggregate across multiple files per (model_name, supercategory) by re-computing accuracy from deduped details
    # To get exact dedup across files, rebuild detail rows first
    detail_rows: List[pd.DataFrame] = []
    for p in result_files:
        try:
            df = pd.read_csv(p, dtype={"question_id": str}, low_memory=False)
        except Exception:
            continue
        needed = {"question_id", "is_correct"}
        if not needed.issubset(df.columns):
            continue
        df = df[df["question_id"].astype(str).isin(verified_ids)].copy()
        if df.empty:
            continue
        if args.use_filename_for_model or ("model_name" not in df.columns) or (df["model_name"].astype(str).nunique() != 1):
            stem = p.stem
            prefix = "llm_test_results_"
            model_name = stem[len(prefix):] if stem.startswith(prefix) else stem
            df["model_name"] = model_name
        else:
            df["model_name"] = df["model_name"].astype(str)
        df["is_correct_num"] = normalize_is_correct(df["is_correct"])  # 0/1
        df = df[["question_id", "model_name", "is_correct_num"]]
        df = df.merge(df_questions, on="question_id", how="left")
        df["supercategory"] = df["subdomain_name"].map(sub_to_super).fillna("Others")
        df["source_file"] = p.name
        detail_rows.append(df)

    details = pd.concat(detail_rows, ignore_index=True)
    details = (
        details.sort_values(["model_name", "question_id", "source_file"])  # deterministic keep
        .drop_duplicates(["model_name", "question_id"], keep="first")
    )
    overall_by_domain = (
        details.groupby(["model_name", "supercategory"], as_index=False)
        .agg(n=("question_id", "nunique"), accuracy=("is_correct_num", "mean"))
        .sort_values(["model_name", "supercategory"])
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_CSV
    if args.only_news:
        out_path = OUTPUT_CSV.with_name("verified_subset_model_scores_by_domain_news.csv")
    overall_by_domain.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote domain scores: {out_path}")


if __name__ == "__main__":
    main()


