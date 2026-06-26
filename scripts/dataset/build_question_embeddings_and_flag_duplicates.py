import os
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    from annoy import AnnoyIndex
except Exception:
    AnnoyIndex = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
FINISHED_DIR = BASE_DIR / "finished"
INPUT_CSV = BASE_DIR / "merged_all_questions_with_subdomains_renamed.csv"
TEXT_CSV = FINISHED_DIR / "questions_text_for_embedding.csv"
EMB_NPY = FINISHED_DIR / "questions_text_embeddings.npy"
ANNOY_IDX = FINISHED_DIR / "questions_text_annoy.ann"
DUPES_CSV = FINISHED_DIR / "potential_duplicate_questions.csv"
CHECKSUM_PATH = FINISHED_DIR / "questions_text_checksum.txt"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # text-embedding-3-small dimension


def build_text_column(df: pd.DataFrame) -> pd.DataFrame:
    q = df.copy()
    # Ensure required columns exist
    required = {"Unique_Serial", "question", "options"}
    missing = required - set(q.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # Normalize types
    q["Unique_Serial"] = q["Unique_Serial"].astype(str)
    q["question"] = q["question"].astype(str)
    q["options"] = q["options"].astype(str)

    def make_text(row) -> str:
        # options is a list-like string; keep raw text to avoid parsing pitfalls
        return f"Question: {row['question']}\nOptions: {row['options']}"

    out = q[["Unique_Serial", "question", "options"]].copy()
    out["text_for_embedding"] = out.apply(make_text, axis=1)
    return out


def get_openai_client() -> "OpenAI":  # type: ignore
    load_dotenv()
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")
    return OpenAI(api_key=api_key)


def embed_texts(client: "OpenAI", texts: List[str], batch_size: int = 512, max_retries: int = 5, sleep_base: float = 1.5) -> np.ndarray:
    embeddings: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        # Retry with exponential backoff
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
                for d in resp.data:
                    embeddings.append(d.embedding)
                break
            except Exception:
                attempt += 1
                if attempt >= max_retries:
                    raise
                time.sleep(sleep_base ** attempt)
    arr = np.array(embeddings, dtype=np.float32)
    if arr.shape[1] != EMBED_DIM:
        # Do not hard fail; dimensions could change. Keep as-is.
        pass
    return arr


def build_annoy(emb: np.ndarray, num_trees: int = 50) -> "AnnoyIndex":  # type: ignore
    if AnnoyIndex is None:
        raise RuntimeError("annoy package not available. Install: pip install annoy")
    dim = int(emb.shape[1])
    idx = AnnoyIndex(dim, metric='angular')
    for i in tqdm(range(emb.shape[0]), desc="Building Annoy index", unit="vec"):
        idx.add_item(i, emb[i].tolist())
    idx.build(num_trees)
    return idx


def find_potential_duplicates(ids: List[str], emb: np.ndarray, idx: "AnnoyIndex", top_k: int = 6, sim_threshold: float = 0.92) -> pd.DataFrame:  # type: ignore
    # Angular distance d ~ sqrt(2*(1-cos_sim)) for normalized; use Annoy distances directly
    rows = []
    for i in tqdm(range(len(ids)), desc="Querying neighbors", unit="item"):
        neighbors = idx.get_nns_by_item(i, top_k, include_distances=True)
        n_ids, dists = neighbors
        for j, dist in zip(n_ids, dists):
            if j == i:
                continue
            # Convert angular distance in [0,2] to similarity proxy (cosine approx)
            # For small angles: cos_sim ~ 1 - dist^2/2, but Annoy's angular returns 2*(1-cos)
            # Annoy's angular distance ≈ 2*(1 - cos_sim) => cos_sim ≈ 1 - dist/2
            cos_sim = max(0.0, 1.0 - (dist / 2.0))
            if cos_sim >= sim_threshold:
                a, b = ids[i], ids[j]
                if a < b:  # dedupe pair ordering
                    rows.append({"question_id_a": a, "question_id_b": b, "similarity": cos_sim})
    if not rows:
        return pd.DataFrame(columns=["question_id_a", "question_id_b", "similarity"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["question_id_a", "question_id_b"]).sort_values("similarity", ascending=False)
    return df


def compute_checksum(df: pd.DataFrame) -> str:
    # Stable checksum over ids + text to detect changes
    h = hashlib.sha256()
    for _, r in df[["Unique_Serial", "text_for_embedding"]].sort_values("Unique_Serial").iterrows():
        h.update(str(r["Unique_Serial"]).encode("utf-8", errors="ignore"))
        h.update(b"\n")
        h.update(str(r["text_for_embedding"]).encode("utf-8", errors="ignore"))
        h.update(b"\n\n")
    return h.hexdigest()


def load_cached_embeddings(expected_len: int, expected_checksum: str, force: bool) -> Optional[np.ndarray]:
    if force:
        return None
    try:
        if not EMB_NPY.exists() or not CHECKSUM_PATH.exists():
            return None
        prev_checksum = CHECKSUM_PATH.read_text(encoding="utf-8").strip()
        if prev_checksum != expected_checksum:
            return None
        emb = np.load(EMB_NPY)
        if emb.shape[0] != expected_len:
            return None
        return emb
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build question embeddings and flag potential duplicates (no re-embedding)")
    parser.add_argument("--top_k", type=int, default=10, help="Neighbors to retrieve per item")
    parser.add_argument("--sim_threshold", type=float, default=0.80, help="Cosine-like similarity threshold for duplicates")
    args = parser.parse_args()

    FINISHED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, dtype={"Unique_Serial": str}, low_memory=False)
    text_df = build_text_column(df)
    text_df.to_csv(TEXT_CSV, index=False, encoding="utf-8")

    # Do not compute embeddings again. Always load from existing .npy, or error.
    if not EMB_NPY.exists():
        raise RuntimeError(
            f"Embeddings file not found: {EMB_NPY}. Embedding generation is disabled in this script."
        )
    emb = np.load(EMB_NPY)
    # Optional sanity checks
    if CHECKSUM_PATH.exists():
        try:
            checksum_now = compute_checksum(text_df)
            checksum_prev = CHECKSUM_PATH.read_text(encoding="utf-8").strip()
            if checksum_now != checksum_prev:
                print("Warning: current text checksum differs from cached embeddings; results may be stale.")
        except Exception:
            pass
    if emb.shape[0] != len(text_df):
        print(f"Warning: embedding count ({emb.shape[0]}) != questions ({len(text_df)}). Proceeding with min length.")
        min_len = min(emb.shape[0], len(text_df))
        emb = emb[:min_len]
        text_df = text_df.iloc[:min_len].reset_index(drop=True)

    # Always (re)build a small index; cheap and ensures consistency with current settings
    idx = build_annoy(emb, num_trees=64)
    idx.save(str(ANNOY_IDX))

    ids = text_df["Unique_Serial"].astype(str).tolist()
    dupes = find_potential_duplicates(ids, emb, idx, top_k=args.top_k, sim_threshold=args.sim_threshold)

    # Join actual texts for human review
    id_to_q = text_df.set_index("Unique_Serial")["question"].to_dict()
    id_to_opts = text_df.set_index("Unique_Serial")["options"].to_dict()
    if not dupes.empty:
        dupes = dupes.assign(
            question_a=dupes["question_id_a"].map(id_to_q),
            options_a=dupes["question_id_a"].map(id_to_opts),
            question_b=dupes["question_id_b"].map(id_to_q),
            options_b=dupes["question_id_b"].map(id_to_opts),
        )
    dupes.to_csv(DUPES_CSV, index=False, encoding="utf-8")

    print(f"Saved text CSV: {TEXT_CSV}")
    print(f"Loaded embeddings: {EMB_NPY}")
    print(f"Saved Annoy index: {ANNOY_IDX}")
    print(f"Saved potential duplicates: {DUPES_CSV}")


if __name__ == "__main__":
    main()


