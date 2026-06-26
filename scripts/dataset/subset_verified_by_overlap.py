import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FINISHED = BASE_DIR / "finished"
VERIFIED_CSV = FINISHED / "verified_dataset_gpt_5_mini_ACCEPTED.csv"
OVERLAP_IDS_CSV = FINISHED / "overlap_hard_vs_verified_ids.csv"
OUTPUT_CSV = FINISHED / "verified_dataset_overlap_with_hard_subset.csv"


def main() -> None:
    if not OVERLAP_IDS_CSV.exists():
        raise SystemExit(f"Overlap ids not found: {OVERLAP_IDS_CSV}")
    ids = pd.read_csv(OVERLAP_IDS_CSV, dtype={"question_id": str})["question_id"].astype(str)
    id_set = set(ids)

    df = pd.read_csv(VERIFIED_CSV, dtype={"question_id": str}, low_memory=False)
    sub = df[df["question_id"].astype(str).isin(id_set)].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Kept {len(sub):,} of {len(df):,} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()





