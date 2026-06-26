import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FINISHED_DIR = BASE_DIR / "finished"
INPUT_CSV = BASE_DIR / "merged_all_questions_with_subdomains_renamed.csv"
PAIRS_CSV = FINISHED_DIR / "potential_duplicate_questions.csv"
OUTPUT_CSV = FINISHED_DIR / "merged_all_questions_dedup.csv"
CLUSTERS_CSV = FINISHED_DIR / "duplicate_clusters.csv"


def build_graph(pairs: pd.DataFrame) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for _, r in pairs.iterrows():
        a = str(r["question_id_a"]) if "question_id_a" in r else None
        b = str(r["question_id_b"]) if "question_id_b" in r else None
        if not a or not b:
            continue
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def connected_components(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    for node in graph.keys():
        if node in seen:
            continue
        stack = [node]
        comp: Set[str] = set()
        seen.add(node)
        while stack:
            u = stack.pop()
            comp.add(u)
            for v in graph.get(u, set()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def choose_representative(comp: Set[str], df: pd.DataFrame, rule: str) -> str:
    if rule == "longest_question":
        # Keep the item with the longest question text; tie-break by id
        sub = df[df["Unique_Serial"].astype(str).isin(comp)].copy()
        sub["q_len"] = sub["question"].astype(str).str.len()
        sub = sub.sort_values(["q_len", "Unique_Serial"], ascending=[False, True])
        return str(sub.iloc[0]["Unique_Serial"]) if not sub.empty else sorted(comp)[0]
    # Default: lexicographically smallest id
    return sorted(comp)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove duplicate questions using pairs file")
    parser.add_argument("--rule", choices=["min_id", "longest_question"], default="min_id",
                        help="Representative selection rule per duplicate cluster")
    args = parser.parse_args()

    FINISHED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, dtype={"Unique_Serial": str}, low_memory=False)
    if not PAIRS_CSV.exists():
        print(f"Pairs file not found: {PAIRS_CSV}. Nothing to deduplicate.")
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        return

    pairs = pd.read_csv(PAIRS_CSV, dtype=str, low_memory=False)
    # If empty or missing required columns, just copy input
    if pairs.empty or not {"question_id_a", "question_id_b"}.issubset(pairs.columns):
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print("No valid duplicate pairs. Wrote original as deduped output.")
        return

    graph = build_graph(pairs)
    comps = connected_components(graph)

    # Build mapping representative -> members
    keep_ids: Set[str] = set()
    drop_ids: Set[str] = set()
    cluster_rows: List[Tuple[int, str, str, bool]] = []
    for idx, comp in enumerate(comps, start=1):
        rep = choose_representative(comp, df, args.rule)
        keep_ids.add(rep)
        for m in sorted(comp):
            is_rep = (m == rep)
            if not is_rep:
                drop_ids.add(m)
            cluster_rows.append((idx, rep, m, is_rep))

    # Also include isolated nodes mentioned only once (if any)
    ids_in_pairs = set(pairs["question_id_a"].astype(str)) | set(pairs["question_id_b"].astype(str))
    isolated = ids_in_pairs - set(graph.keys())
    for m in sorted(isolated):
        keep_ids.add(m)
        cluster_rows.append((len(comps) + 1, m, m, True))

    # Write mapping
    clusters_df = pd.DataFrame(cluster_rows, columns=["cluster_id", "representative_id", "member_id", "is_representative"])
    clusters_df.to_csv(CLUSTERS_CSV, index=False, encoding="utf-8")

    # Produce deduped dataset
    before = len(df)
    dedup = df[~df["Unique_Serial"].astype(str).isin(drop_ids)].copy()
    after = len(dedup)
    dedup.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"Clusters: {len(comps)} | Dropped: {before - after} | Kept: {after}")
    print(f"Wrote deduped dataset: {OUTPUT_CSV}")
    print(f"Wrote cluster mapping: {CLUSTERS_CSV}")


if __name__ == "__main__":
    main()



