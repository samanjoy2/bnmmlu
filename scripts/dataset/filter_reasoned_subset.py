from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "finished" / "special_small_subset_for_cot_with_reasoning.csv"
OUTPUT_CSV = BASE_DIR / "finished" / "special_small_subset_for_cot_filtered.csv"


def to_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y", "t"])


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"Input not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    if "model_is_correct" not in df.columns or "answers_match" not in df.columns:
        print("Required columns missing: model_is_correct, answers_match")
        return

    keep = to_bool(df["model_is_correct"]) & to_bool(df["answers_match"])
    filtered = df[keep].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Wrote filtered rows: {len(filtered)} -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


