import os
import json
import ast
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

try:
	from openai import OpenAI  # type: ignore
except Exception as exc:
	raise RuntimeError("OpenAI SDK not installed. Please run: pip install -r requirements.txt") from exc


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILTERED = BASE_DIR / "finished" / "special_small_subset_for_cot_filtered.csv"
INPUT_BASE = BASE_DIR / "finished" / "special_small_subset_for_cot.csv"
INPUT_QUESTIONS = BASE_DIR / "merged_all_questions_with_subdomains_renamed.csv"
FINISHED_DIR = BASE_DIR / "finished"
OUTPUT_FINAL = BASE_DIR / "finished" / "special_small_subset_for_cot_final.csv"

DEFAULT_MODEL = os.environ.get("MODEL", "gpt-5")
TARGET_PER_SUBDOMAIN = 5


def parse_options(raw: str) -> List[str]:
	try:
		value = ast.literal_eval(raw)
		if isinstance(value, list):
			return [str(x) for x in value]
	except Exception:
		pass
	cleaned = str(raw).strip()
	if cleaned.startswith("[") and cleaned.endswith("]"):
		cleaned = cleaned[1:-1]
	parts = [p.strip().strip("'\"") for p in cleaned.split(",")]
	return [p for p in parts if p]


def load_finished_results() -> pd.DataFrame:
	csv_paths = sorted(FINISHED_DIR.glob("llm_test_results_*.csv"))
	frames: List[pd.DataFrame] = []
	for path in csv_paths:
		try:
			df = pd.read_csv(path, dtype={"question_id": str}, low_memory=False)
			if {"question_id", "is_correct"}.issubset(df.columns):
				df = df[["question_id", "is_correct"]].copy()
				df["source_file"] = path.name
				frames.append(df)
		except Exception:
			continue
	if not frames:
		return pd.DataFrame(columns=["question_id", "is_correct", "source_file"]) 
	return pd.concat(frames, ignore_index=True)


def normalize_is_correct(series: pd.Series) -> pd.Series:
	as_str = series.astype(str).str.strip().str.lower()
	truthy = {"true", "1", "yes", "y", "t"}
	return as_str.isin(truthy).astype(int)


def compute_accuracy_across_models() -> pd.DataFrame:
	results = load_finished_results()
	if results.empty:
		return pd.DataFrame(columns=["question_id", "n_models", "correct_sum", "accuracy_ratio"]) 
	results["is_correct_num"] = normalize_is_correct(results["is_correct"])  # 0/1
	stats = (
		results.groupby("question_id", as_index=False)
		.agg(n_models=("is_correct_num", "size"), correct_sum=("is_correct_num", "sum"))
	)
	stats["accuracy_ratio"] = stats["correct_sum"] / stats["n_models"].clip(lower=1)
	return stats


def build_prompt(question: str, options: List[str]) -> str:
	letters = [chr(ord('a') + i) for i in range(len(options))]
	option_lines = [f"{letter}) {text}" for letter, text in zip(letters, options)]
	options_block = "\n".join(option_lines)
	instruction = (
		"Think step by step and explain why the correct answer is correct and the others are wrong. "
		"Provide two versions of the reasoning: one in English and one in Bengali. "
		"Each reasoning should end with the line: Final Answer: (X) where X is the option letter."
	)
	return (
		f"Question:\n{question}\n\n"
		f"Options:\n{options_block}\n\n"
		f"Instruction:\n{instruction}\n"
	)


def get_response_json(client: OpenAI, model: str, prompt: str, retry: int = 3, sleep_seconds: float = 2.0) -> Dict:
	system_msg = (
		"You are an expert test-solver. Output ONLY valid JSON with keys: "
		"reasoning_en (string), reasoning_bn (string), final_answer (single lowercase letter), "
		"explanations (object keyed by option letter). No extra text."
	)
	last_err: Optional[Exception] = None
	for _ in range(retry):
		try:
			resp = client.chat.completions.create(
				model=model,
				messages=[
					{"role": "system", "content": system_msg},
					{"role": "user", "content": prompt},
				],
				response_format={"type": "json_object"},
				# temperature=0.2,
				# max_tokens=800,
			)
			content = resp.choices[0].message.content or "{}"
			return json.loads(content)
		except Exception as e:
			last_err = e
			time.sleep(sleep_seconds)
	if last_err is not None:
		raise last_err
	raise RuntimeError("Unknown error calling OpenAI Chat Completions API")


def main() -> None:
	load_dotenv()
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		print("OPENAI_API_KEY not set. Please set it before running.")
		return

	if not INPUT_FILTERED.exists() or not INPUT_BASE.exists() or not INPUT_QUESTIONS.exists():
		print("Input CSVs not found.")
		return

	model = DEFAULT_MODEL
	print(f"Using model: {model}")
	client = OpenAI()

	df_filtered = pd.read_csv(INPUT_FILTERED)
	df_base = pd.read_csv(INPUT_BASE)
	df_questions = pd.read_csv(INPUT_QUESTIONS, dtype={"Unique_Serial": str})

	# Candidate pool: full dataset ranked by cross-model accuracy
	acc_stats = compute_accuracy_across_models()
	df_questions_subset = df_questions[[
		"Unique_Serial", "subdomain_name", "subject", "question", "correct_answer", "options"
	]].rename(columns={"Unique_Serial": "question_id"})
	df_candidates = df_questions_subset.merge(acc_stats, on="question_id", how="left")
	df_candidates["accuracy_ratio"] = df_candidates["accuracy_ratio"].fillna(0.0)
	df_candidates["n_models"] = df_candidates["n_models"].fillna(0)

	# Prepare output: write existing filtered rows ONCE
	columns_out = list(df_filtered.columns)
	df_filtered.drop_duplicates(subset=["question_id"]).to_csv(OUTPUT_FINAL, index=False, encoding="utf-8")

	used_ids = set(df_filtered["question_id"].astype(str).tolist())

	# Determine which subdomains actually need top-up
	base_subdomains = list(pd.unique(df_base["subdomain_name"]))
	counts = df_filtered.groupby("subdomain_name").size()
	subdomains_to_fill: List[str] = []
	for sub in base_subdomains:
		cnt = int(counts.get(sub, 0))
		if cnt < TARGET_PER_SUBDOMAIN:
			subdomains_to_fill.append(sub)

	for subdomain in tqdm(subdomains_to_fill, desc="Top-up per subdomain"):
		current_cnt = int(counts.get(subdomain, 0))
		need = max(0, TARGET_PER_SUBDOMAIN - current_cnt)
		if need <= 0:
			continue

		# Find candidates not yet used
		cands = (
			df_candidates[df_candidates["subdomain_name"] == subdomain]
			.sort_values(["accuracy_ratio", "n_models"], ascending=[False, False])
		)
		cands = cands[~cands["question_id"].astype(str).isin(used_ids)]

		filled_rows: List[pd.DataFrame] = []
		for _, cand in cands.iterrows():
			if need <= 0:
				break
			options = parse_options(str(cand.get("options", "")))
			if not options:
				continue
			prompt = build_prompt(str(cand.get("question", "")), options)
			try:
				response_obj = get_response_json(client, model, prompt)
				final_answer = str(response_obj.get("final_answer", "")).strip().lower()
				correct_answer = str(cand.get("correct_answer", "")).strip().lower()
				is_match = final_answer == correct_answer if final_answer and correct_answer else False
				if is_match:
					row = cand.to_dict()
					row.update({
						"model_reasoning_json": json.dumps(response_obj, ensure_ascii=False),
						"model_final_answer": final_answer,
						"model_reasoning_en": response_obj.get("reasoning_en", ""),
						"model_reasoning_bn": response_obj.get("reasoning_bn", ""),
						"model_is_correct": True,
						"answers_match": True,
					})
					filled_rows.append(pd.DataFrame([row], columns=columns_out))
					used_ids.add(str(cand.get("question_id")))
					need -= 1
			except Exception:
				continue

		if filled_rows:
			pd.concat(filled_rows, ignore_index=True).to_csv(
				OUTPUT_FINAL, mode="a", header=False, index=False, encoding="utf-8"
			)

	print(f"Wrote top-up final CSV: {OUTPUT_FINAL}")


if __name__ == "__main__":
	main()
