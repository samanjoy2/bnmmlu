import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "finished" / "verified_dataset_gpt_5_mini_ACCEPTED.csv"
BY_SUBDOMAIN_CSV = BASE_DIR / "finished" / "verified_dataset_summary_by_subdomain.csv"
REPORT_TXT = BASE_DIR / "finished" / "verified_dataset_summary_report.txt"


def safe_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"true", "1", "yes", "y", "t"})


def main() -> None:
    df = pd.read_csv(INPUT, dtype={"question_id": str}, low_memory=False)

    # Coerce types
    df["failure_ratio"] = pd.to_numeric(df["failure_ratio"], errors="coerce")
    df["given_answer_correct_bool"] = safe_bool(df.get("given_answer_correct", False))
    df["question_valid_bool"] = safe_bool(df.get("question_valid", True))
    df["impossible_to_answer_bool"] = safe_bool(df.get("impossible_to_answer", False))

    # Overall stats
    total = len(df)
    n_subdomains = df["subdomain_name"].nunique()
    fr = df["failure_ratio"].dropna()
    overall = {
        "total_rows": total,
        "unique_subdomains": int(n_subdomains),
        "failure_ratio_mean": float(fr.mean()) if not fr.empty else np.nan,
        "failure_ratio_median": float(fr.median()) if not fr.empty else np.nan,
        "failure_ratio_std": float(fr.std(ddof=0)) if not fr.empty else np.nan,
        "failure_ratio_min": float(fr.min()) if not fr.empty else np.nan,
        "failure_ratio_max": float(fr.max()) if not fr.empty else np.nan,
        "pct_given_answer_correct": float(df["given_answer_correct_bool"].mean()),
        "pct_question_valid": float(df["question_valid_bool"].mean()),
        "pct_impossible_to_answer": float(df["impossible_to_answer_bool"].mean()),
    }

    # Distribution of correct_answer letters
    letter_counts = (
        df["correct_answer"].astype(str).str.upper().str.strip().value_counts().reindex(list("ABCD"), fill_value=0)
    )

    # Per-subdomain table
    grouped = (
        df.groupby("subdomain_name", dropna=False)
        .agg(
            n=("question_id", "size"),
            failure_ratio_mean=("failure_ratio", "mean"),
            failure_ratio_median=("failure_ratio", "median"),
            failure_ratio_std=("failure_ratio", "std"),
            failure_ratio_min=("failure_ratio", "min"),
            failure_ratio_max=("failure_ratio", "max"),
            pct_given_answer_correct=("given_answer_correct_bool", "mean"),
            pct_question_valid=("question_valid_bool", "mean"),
            pct_impossible_to_answer=("impossible_to_answer_bool", "mean"),
        )
        .reset_index()
        .sort_values(["n"], ascending=False)
    )

    BY_SUBDOMAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(BY_SUBDOMAIN_CSV, index=False, encoding="utf-8")

    # Prepare report
    lines: list[str] = []
    lines.append("Verified dataset summary (gpt_5_mini_ACCEPTED)")
    lines.append("")
    lines.append(f"Input file: {INPUT.name}")
    lines.append(f"Total rows: {overall['total_rows']:,}")
    lines.append(f"Unique subdomains: {overall['unique_subdomains']:,}")
    lines.append("")
    lines.append(
        "Failure ratio: "
        f"mean={overall['failure_ratio_mean']:.4f}, median={overall['failure_ratio_median']:.4f}, "
        f"std={overall['failure_ratio_std']:.4f}, min={overall['failure_ratio_min']:.4f}, max={overall['failure_ratio_max']:.4f}"
    )
    lines.append(
        "Quality flags: "
        f"given_correct={overall['pct_given_answer_correct']*100:.2f}%, "
        f"question_valid={overall['pct_question_valid']*100:.2f}%, "
        f"impossible_to_answer={overall['pct_impossible_to_answer']*100:.2f}%"
    )
    lines.append("")
    lines.append("Correct answer distribution (A/B/C/D):")
    for letter in "ABCD":
        lines.append(f"- {letter}: {int(letter_counts.get(letter, 0)):,}")
    lines.append("")
    lines.append("Top 15 subdomains by count:")
    for _, r in grouped.head(15).iterrows():
        lines.append(
            f"- {r['subdomain_name']}: n={int(r['n'])}, fr_mean={r['failure_ratio_mean']:.3f}, "
            f"given_correct={r['pct_given_answer_correct']*100:.1f}%"
        )
    lines.append("")
    lines.append("Top 15 subdomains by highest mean failure ratio (>= 100 questions):")
    tmp = grouped[grouped["n"] >= 100].sort_values("failure_ratio_mean", ascending=False).head(15)
    for _, r in tmp.iterrows():
        lines.append(
            f"- {r['subdomain_name']}: n={int(r['n'])}, fr_mean={r['failure_ratio_mean']:.3f}, "
            f"question_valid={r['pct_question_valid']*100:.1f}%"
        )

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote per-subdomain CSV: {BY_SUBDOMAIN_CSV}")
    print(f"Wrote report: {REPORT_TXT}")


if __name__ == "__main__":
    main()





