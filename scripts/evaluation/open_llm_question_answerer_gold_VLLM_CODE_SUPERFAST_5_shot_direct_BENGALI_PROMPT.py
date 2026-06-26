    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM-Optimized TigerLLM Question Answerer (Linux-ready)
=======================================================

This script uses vLLM's high-throughput engine for TigerLLM-1B-it inference,
with batched-friendly generation and minimal latency.
"""

import pandas as pd
from vllm import LLM, SamplingParams
import json
import os
import time
import random
from tqdm import tqdm
import sys
import io
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

chudling_pong = "TigerLLM-9B-it"
actual_batch_size = 32

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# vLLM runs best on Linux with CUDA. No torch.compile changes necessary here.

class QuestionAnswerer:
    def __init__(self):
        """Initialize the question answerer with vLLM engine"""
        self.model_name = "md-nishat-008/TigerLLM-9B-it"  # TigerLLM model
        self.results_csv = f"llm_test_results_{chudling_pong}.csv"
        self.csv_lock = threading.Lock()  # Thread-safe CSV writing
        self.model_lock = threading.Lock()  # Thread-safe model access
        self.failed_questions_file = f"failed_questions_{chudling_pong}.txt"
        self.random_answers_file = f"random_answers_{chudling_pong}.txt"
        self.random_count = 0
        self.total_processed = 0
        self.total_correct = 0

        print("🔄 Loading TigerLLM with vLLM...")
        self.load_model()
        self.initialize_results_csv()
        # Usage CSV for token/cost tracking
        self.usage_csv = f"llm_usage_{chudling_pong}.csv"
        self.initialize_usage_csv()
        # Load exemplars dataframe once for 5-shot CoT prompting
        self.exemplars_df = self.load_exemplars_df()
    def _get_row_question_id(self, row, fallback=None):
        """Return a stable question id from row supporting different schemas."""
        try:
            if 'question_id' in row:
                return row['question_id']
            if 'Unique_Serial' in row:
                return row['Unique_Serial']
        except Exception:
            pass
        return fallback
    
    def load_model(self):
        """Load TigerLLM with vLLM for fast inference"""
        try:
            print("🚀 Starting vLLM engine...")
            # vLLM LLM engine (uses CUDA if available). Adjust gpu_memory_utilization as needed.
            self.llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=0.92,
                max_model_len=2704,
            )
            print("✅ TigerLLM loaded with vLLM!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure you have vllm installed and NVIDIA drivers set up on Linux:")
            print("   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121")
            raise
    
    def initialize_results_csv(self):
        """Initialize the results CSV file if it doesn't exist"""
        if not os.path.exists(self.results_csv):
            # Create CSV with headers
            columns = [
                'question_id',
                'correct_answer', 
                'llm_answer',
                'original_llm_response',
                'model_name',
                'timestamp',
                'is_correct'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.results_csv, index=False, encoding='utf-8')
            print(f"✅ Created results CSV: {self.results_csv}")
        else:
            print(f"✅ Using existing results CSV: {self.results_csv}")
    
    def initialize_usage_csv(self):
        """Initialize the usage CSV file if it doesn't exist"""
        if not os.path.exists(self.usage_csv):
            columns = [
                'question_id',
                'model_name',
                'timestamp',
                'prompt_tokens',
                'completion_tokens',
                'total_tokens',
                'prompt_cost_usd',
                'completion_cost_usd',
                'estimated_cost_usd'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.usage_csv, index=False, encoding='utf-8')
            print(f"✅ Created usage CSV: {self.usage_csv}")
        else:
            print(f"✅ Using existing usage CSV: {self.usage_csv}")

    def save_usage_to_csv(self, question_id, usage_info):
        """Append token/cost usage for a question to the usage CSV (thread-safe)."""
        if not usage_info:
            return
        row = {
            'question_id': question_id,
            'model_name': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prompt_tokens': usage_info.get('prompt_tokens'),
            'completion_tokens': usage_info.get('completion_tokens'),
            'total_tokens': usage_info.get('total_tokens'),
            'prompt_cost_usd': usage_info.get('prompt_cost_usd'),
            'completion_cost_usd': usage_info.get('completion_cost_usd'),
            'estimated_cost_usd': usage_info.get('estimated_cost_usd'),
        }
        with self.csv_lock:
            df = pd.DataFrame([row])
            df.to_csv(self.usage_csv, mode='a', header=False, index=False, encoding='utf-8')
    
    def get_processed_question_ids(self):
        """Get list of already processed question IDs"""
        if os.path.exists(self.results_csv):
            try:
                df = pd.read_csv(self.results_csv, encoding='utf-8')
                if not df.empty:
                    processed_ids = set(df['question_id'].astype(str))
                    print(f"📋 Found {len(processed_ids)} already processed questions")
                    return processed_ids
            except Exception as e:
                print(f"⚠️ Error reading existing results: {e}")
        return set()
    
    def save_result_to_csv(self, question_id, correct_answer, llm_answer, original_response):
        """Save a single result to CSV immediately (thread-safe)"""
        is_correct = llm_answer == correct_answer.upper() if llm_answer else False
        
        new_row = {
            'question_id': question_id,
            'correct_answer': correct_answer,
            'llm_answer': llm_answer,
            'original_llm_response': original_response,
            'model_name': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_correct': is_correct
        }
        
        # Thread-safe CSV writing
        with self.csv_lock:
            df = pd.DataFrame([new_row])
            df.to_csv(self.results_csv, mode='a', header=False, index=False, encoding='utf-8')

    def save_failure_row(self, question_id, correct_answer, reason_text: str):
        """Record a failure/result placeholder so no dataset rows are missed."""
        # Save a results row with no answer and the reason echoed in original_llm_response
        self.save_result_to_csv(question_id, correct_answer, None, reason_text)
        # Save a zeroed usage row
        self.save_usage_to_csv(question_id, {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'prompt_cost_usd': 0.0,
            'completion_cost_usd': 0.0,
            'estimated_cost_usd': 0.0,
        })
    
    def log_failed_question(self, question_id, question, reason):
        """Log failed questions to a separate text file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] ID: {question_id} - REASON: {reason}\nQUESTION: {question}\n{'-'*80}\n"
        
        with self.csv_lock:  # Reuse the same lock for thread safety
            with open(self.failed_questions_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
    
    def log_random_answer(self, question_id, question, random_answer):
        """Log when we had to use a random answer"""
        self.random_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] ID: {question_id} - RANDOM ANSWER: {random_answer} (Total random: {self.random_count})\nQUESTION: {question}\n{'-'*80}\n"
        
        with self.csv_lock:  # Reuse the same lock for thread safety
            with open(self.random_answers_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
    
    def update_accuracy_stats(self, is_correct):
        """Update real-time accuracy statistics"""
        self.total_processed += 1
        if is_correct:
            self.total_correct += 1
    
    def get_current_accuracy(self):
        """Get current accuracy percentage"""
        if self.total_processed == 0:
            return 0.0
        return (self.total_correct / self.total_processed) * 100
        
    def load_dataset(self, file_path):
        """Load the questions dataset"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"✅ Loaded dataset with {len(df):,} questions")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def load_exemplars_df(self):
        """Load exemplars CSV into a dataframe for per-subdomain 5-shot selection."""
        exemplars_csv = Path("finished/special_small_subset_for_cot_final.csv")
        if not exemplars_csv.exists():
            return None
        try:
            df = pd.read_csv(exemplars_csv, encoding='utf-8', low_memory=False)
            return df
        except Exception as e:
            print(f"⚠️ Could not read exemplars CSV: {e}")
            return None

    def _format_exemplar_block(self, row) -> str:
        """Format one exemplar block: question, options, brief reasoning (if present), and answer."""
        q = str(row.get('question', ''))
        opts_raw = row.get('options')
        try:
            opts = self.parse_options(opts_raw)
        except Exception:
            opts = []
        if len(opts) < 4:
            opts = list(opts) + [''] * (4 - len(opts))
        ans = str(row.get('correct_answer', '')).strip().upper()
        reason_col = None
        for c in ['reason', 'reasoning', 'explanation']:
            if c in row.index:
                reason_col = c
                break
        block = (
            f"উদাহরণ:\n"
            f"প্রশ্ন: {q}\n\n"
            f"বিকল্পসমূহ:\n"
            f"A) {opts[0]}\n"
            f"B) {opts[1]}\n"
            f"C) {opts[2]}\n"
            f"D) {opts[3]}\n\n"
            f"উত্তর: {ans}\n"
        )
        return block

    def _get_exemplar_blocks_for_subdomain(self, subdomain: str | None, n: int = 5) -> list[str]:
        """Return up to n exemplar blocks filtered by subdomain; fallback to global if needed."""
        if self.exemplars_df is None or self.exemplars_df.empty:
            return []
        df = self.exemplars_df
        # Identify subdomain column
        sub_col = None
        for c in ['subdomain_name', 'subdomain', 'Subdomain']:
            if c in df.columns:
                sub_col = c
                break
        blocks: list[str] = []
        if subdomain and sub_col:
            try:
                df_sub = df[df[sub_col].astype(str) == str(subdomain)].head(n)
            except Exception:
                df_sub = df.head(0)
            for _, r in df_sub.iterrows():
                blocks.append(self._format_exemplar_block(r))
        # Top-up from global exemplars if insufficient
        if len(blocks) < n:
            needed = n - len(blocks)
            topup = df.head(n).iloc[:needed]
            for _, r in topup.iterrows():
                blocks.append(self._format_exemplar_block(r))
        return blocks[:n]
    
    def parse_options(self, options_str):
        """Robustly parse the options string into a 4-item list."""
        if options_str is None or (isinstance(options_str, float) and pd.isna(options_str)):
            return []
        s = str(options_str).strip()
        # Try JSON array
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj]
        except Exception:
            pass
        # Try Python literal list
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj]
        except Exception:
            pass
        # Extract quoted segments
        try:
            parts = []
            for m in re.finditer(r"'([^']*)'|\"([^\"]*)\"", s):
                parts.append((m.group(1) if m.group(1) is not None else m.group(2)).strip())
            if parts:
                return parts
        except Exception:
            pass
        # Fallback: comma split
        try:
            return [p.strip().strip("'\"") for p in s.strip('[]').split(',') if p.strip()]
        except Exception:
            return []
    
    def format_question_for_llm(self, question, options):
        """Format question and options for the LLM using English prompt"""
        # Ensure we have exactly 4 options
        if len(options) < 4:
            options.extend([''] * (4 - len(options)))
        
        # Use English prompt for better compatibility
        formatted_question = f"""প্রশ্ন: {question}

বিকল্পসমূহ:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

উত্তর দিন (শুধু A, B, C, বা D)। অন্য কিছু ছাড়া শুধু অক্ষরটি লিখুন: """
        
        return formatted_question
    
    def ask_llm(self, question, options, max_retries=5):
        """Ask the LLM to answer the question using vLLM"""

        # Different prompt variations to try (zero-shot CoT with concise reasoning + JSON output)
        prompt_variations = [
            # Variation 1: CoT with explicit last-line JSON instruction
            f"""প্রশ্ন: {question}

বিকল্পসমূহ:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

সরাসরি সঠিক উত্তর দিন। শুধু A, B, C, বা D লিখুন। অন্য কিছু লিখবেন না:
""",
            
            # Variation 2: Simpler CoT format + JSON
            f"""{question}

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

সরাসরি উত্তর দিন। শুধু A, B, C, বা D লিখুন:
""",
            
            # Variation 3: Direct compact CoT + JSON
            f"""{question}
A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}
সরাসরি সঠিক বিকল্পটি লিখুন। শুধু A, B, C, বা D থাকবে, অন্য কিছু নয়:
""",
            
            # Variation 4: Bengali CoT format + JSON
            f"""প্রশ্ন: {question}
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}
সরাসরি সঠিক উত্তর লিখুন। শুধু A, B, C, বা D হবে; অন্য কিছু নয়:
""",
            
            # Variation 5: Minimal CoT + JSON
            f"""{question}
A){options[0]} B){options[1]} C){options[2]} D){options[3]}
সরাসরি উত্তর দিন। শুধু A, B, C, বা D লিখুন। অন্য কিছু নয়:
"""
        ]

        for attempt in range(max_retries):
            try:
                current_prompt = prompt_variations[attempt]

                # vLLM is thread-safe and batches internally; still guard for safety
                with self.model_lock:
                    sp = SamplingParams(
                        # temperature=0.2,
                        max_tokens=1024,
                        # top_p=0.95,
                    )
                    outputs = self.llm.generate([current_prompt], sp)
                    req_out = outputs[0]
                    generated = req_out.outputs[0]
                    generated_text = (generated.text or '').strip()

                    # Token usage (best-effort via vLLM outputs)
                    prompt_tokens = None
                    completion_tokens = None
                    try:
                        if hasattr(req_out, 'prompt_token_ids') and req_out.prompt_token_ids is not None:
                            prompt_tokens = len(req_out.prompt_token_ids)
                        if hasattr(generated, 'token_ids') and generated.token_ids is not None:
                            completion_tokens = len(generated.token_ids)
                    except Exception:
                        pass
                    # Fallback: try tokenizer if counts missing
                    try:
                        if prompt_tokens is None and hasattr(self.llm, 'tokenizer'):
                            prompt_tokens = len(self.llm.tokenizer(current_prompt)['input_ids'])
                    except Exception:
                        pass
                    if prompt_tokens is None:
                        prompt_tokens = 0
                    if completion_tokens is None:
                        completion_tokens = 0
                    total_tokens = prompt_tokens + completion_tokens

                    # Cost estimation from environment (defaults to 0)
                    try:
                        import os as _os
                        rate_prompt = float(_os.getenv('PROMPT_COST_PER_1K_USD', '0') or '0')
                        rate_completion = float(_os.getenv('COMPLETION_COST_PER_1K_USD', '0') or '0')
                    except Exception:
                        rate_prompt = 0.0
                        rate_completion = 0.0
                    prompt_cost = (prompt_tokens / 1000.0) * rate_prompt
                    completion_cost = (completion_tokens / 1000.0) * rate_completion
                    est_cost = prompt_cost + completion_cost

                    usage_info = {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens,
                        'prompt_cost_usd': prompt_cost,
                        'completion_cost_usd': completion_cost,
                        'estimated_cost_usd': est_cost,
                    }

                    if generated_text:
                        answer = self.extract_answer(generated_text)
                        if answer:
                            return answer, generated_text, usage_info

                # If we get here, try next prompt variation
                time.sleep(0.3)

            except Exception as e:
                time.sleep(0.3)

        # If all attempts failed, return random answer
        import random
        random_answer = random.choice(['A', 'B', 'C', 'D'])
        # No reliable usage in failure; return zeros
        usage_info = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'prompt_cost_usd': 0.0,
            'completion_cost_usd': 0.0,
            'estimated_cost_usd': 0.0,
        }
        return random_answer, f"RANDOM_{random_answer}", usage_info
    
    def extract_answer(self, text):
        """Extract A, B, C, or D from the generated text"""
        # Clean the text
        text = text.strip()
        
        # Bengali to English number mapping
        bengali_to_english = {
            '১': '1', '২': '2', '৩': '3', '৪': '4',
            'এক': '1', 'দুই': '2', 'তিন': '3', 'চার': '4'
        }
        
        # Number to letter mapping
        number_to_letter = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
        
        # Replace Bengali numbers with English
        for bengali, english in bengali_to_english.items():
            text = text.replace(bengali, english)
        
        text_upper = text.upper()
        
        # Try different patterns to extract the answer
        patterns = [
            r'^([ABCD])\b',  # Answer at the beginning
            r'\b([ABCD])\s*$',  # Answer at the end
            r'\b([ABCD])\)',  # Answer with parenthesis
            r'উত্তর:\s*([ABCD])',  # Bengali "Answer:" pattern
            r'ANSWER:\s*([ABCD])',  # English "Answer:" pattern
            r'\b([ABCD])\b',  # Any single letter A, B, C, or D
            r'^([1-4])\b',  # Numbers at the beginning
            r'\b([1-4])\s*$',  # Numbers at the end
            r'\b([1-4])\b',  # Any single number 1, 2, 3, or 4
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                answer = match.group(1)
                if answer in ['A', 'B', 'C', 'D']:
                    return answer
                elif answer in ['1', '2', '3', '4']:
                    return number_to_letter[answer]
        
        # Check for numbers in the original text
        numbers = re.findall(r'\b([1-4])\b', text)
        if len(numbers) == 1:
            return number_to_letter[numbers[0]]
        
        # If no pattern matches, check if the text contains only one of A, B, C, D
        letters = re.findall(r'\b([ABCD])\b', text_upper)
        if len(letters) == 1:
            return letters[0]
        
        return None
    
    
    def clear_gpu_cache(self):
        """No-op with vLLM; engine manages memory and KV cache internally."""
        return
    
    def build_cot_json_prompt(self, question, options, subdomain):
        """Build a 5-shot CoT prompt: include 5 exemplars from the same subdomain, then the target."""
        if len(options) < 4:
            options = list(options) + [''] * (4 - len(options))
        exemplar_blocks = self._get_exemplar_blocks_for_subdomain(subdomain, n=5)
        exemplars_text = "\n\n".join(exemplar_blocks) if exemplar_blocks else ""
        target = (
            f"এখন নিচের প্রশ্নটির উত্তর দিন:\n\n"
            f"প্রশ্ন: {question}\n\n"
            f"বিকল্পসমূহ:\n"
            f"A) {options[0]}\n"
            f"B) {options[1]}\n"
            f"C) {options[2]}\n"
            f"D) {options[3]}\n\n"
            f"সরাসরি সঠিক উত্তর দিন। শুধু A, B, C, বা D লিখুন। অন্য কিছু লিখবেন না:\n"
        )
        if exemplars_text:
            return f"{exemplars_text}\n\n{target}"
        return target

    def process_questions_batch(self, df, batch_indices):
        """Batch-generate for a list of indices using one vLLM generate call."""
        prompts = []
        meta = []  # store (index, question_id, correct_answer, subject, subdomain)
        for idx in batch_indices:
            row = df.iloc[idx]
            qid = self._get_row_question_id(row, fallback=idx)
            question = row.get('question') if 'question' in row.index else row['question']
            correct_answer = row.get('correct_answer') if 'correct_answer' in row.index else row['correct_answer']
            options = self.parse_options(row.get('options') if 'options' in row.index else row['options'])
            if len(options) != 4:
                continue
            subdomain = row.get('subdomain_name') if 'subdomain_name' in row.index else (row.get('subdomain') if 'subdomain' in row.index else None)
            prompts.append(self.build_cot_json_prompt(question, options, subdomain))
            meta.append((idx, qid, correct_answer, row.get('subject'), subdomain, question))

        results = []
        if not prompts:
            return results

        with self.model_lock:
            sp = SamplingParams(
                temperature=0.2,
                max_tokens=2048,
                top_p=0.95,
            )
            try:
                outputs = self.llm.generate(prompts, sp)
            except Exception as e:
                # Batch-level failure: record all rows as failures so nothing is missed
                for _, (idx, qid, correct_answer, subject, subdomain, question) in enumerate(meta):
                    self.save_failure_row(qid, correct_answer, f"batch_error: {e}")
                return []

        for out, (idx, qid, correct_answer, subject, subdomain, question) in zip(outputs, meta):
            try:
                generated = out.outputs[0]
                generated_text = (generated.text or '').strip()
            except Exception as e:
                generated_text = ""
                # Per-item failure: record placeholder rows
                self.save_failure_row(qid, correct_answer, f"item_error: {e}")
                continue

            # Token usage (best-effort)
            prompt_tokens = 0
            completion_tokens = 0
            try:
                if hasattr(out, 'prompt_token_ids') and out.prompt_token_ids is not None:
                    prompt_tokens = len(out.prompt_token_ids)
                if hasattr(generated, 'token_ids') and generated.token_ids is not None:
                    completion_tokens = len(generated.token_ids)
            except Exception:
                pass
            total_tokens = prompt_tokens + completion_tokens

            # Cost estimation (env-configured per-1k rates, default 0)
            try:
                import os as _os
                rate_prompt = float(_os.getenv('PROMPT_COST_PER_1K_USD', '0') or '0')
                rate_completion = float(_os.getenv('COMPLETION_COST_PER_1K_USD', '0') or '0')
            except Exception:
                rate_prompt = 0.0
                rate_completion = 0.0
            prompt_cost = (prompt_tokens / 1000.0) * rate_prompt
            completion_cost = (completion_tokens / 1000.0) * rate_completion
            est_cost = prompt_cost + completion_cost
            usage_info = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'prompt_cost_usd': prompt_cost,
                'completion_cost_usd': completion_cost,
                'estimated_cost_usd': est_cost,
            }

            # Extract answer
            llm_answer = None
            if generated_text:
                llm_answer = self.extract_answer(generated_text)

            # Save row
            self.save_result_to_csv(qid, correct_answer, llm_answer, generated_text)
            self.save_usage_to_csv(qid, usage_info)

            if llm_answer:
                is_correct = llm_answer == str(correct_answer).upper()
                self.update_accuracy_stats(is_correct)
                results.append({
                    'index': idx,
                    'question': question,
                    'correct_answer': correct_answer,
                    'llm_answer': llm_answer,
                    'is_correct': is_correct,
                    'subject': subject,
                    'subdomain': subdomain,
                    'question_id': qid
                })
        return results
    
    def test_single_question(self, df, index=None, show_details=True, skip_processed=True):
        """Test a single question"""
        if index is None:
            index = random.randint(0, len(df) - 1)
        
        row = df.iloc[index]
        question_id = self._get_row_question_id(row, fallback=index)
        
        # Check if already processed
        if skip_processed:
            processed_ids = self.get_processed_question_ids()
            if str(question_id) in processed_ids:
                if show_details:
                    print(f"⏭️ Question {question_id} already processed, skipping...")
                return None
        
        question = row['question']
        correct_answer = row['correct_answer']
        options_str = row['options']
        subject = row['subject']
        subdomain = row['subdomain_name']
        
        if show_details:
            print(f"\n{'='*60}")
            print(f"📝 TESTING QUESTION #{index + 1}")
            print(f"{'='*60}")
            print(f"Subject: {subject}")
            print(f"Subdomain: {subdomain}")
            print(f"Question: {question}")
        
        # Parse options
        options = self.parse_options(options_str)
        if len(options) != 4:
            if show_details:
                print(f"❌ Invalid options count: {len(options)}")
            return None
        
        if show_details:
            print(f"\nOptions:")
            for i, option in enumerate(options):
                letter = chr(65 + i)  # A, B, C, D
                print(f"  {letter}) {option}")
            
            print(f"\nCorrect Answer: {correct_answer}")
            print(f"\n🤖 Asking TituLLM...")
        
        # Ask LLM
        llm_result = self.ask_llm(question, options)
        if llm_result[0] is not None:
            llm_answer, original_response, usage_info = llm_result
            
            # Check if this was a random answer
            if original_response.startswith("RANDOM_"):
                self.log_random_answer(question_id, question, llm_answer)
        else:
            llm_answer, original_response, usage_info = None, "Failed after 5 attempts", {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'prompt_cost_usd': 0.0,
                'completion_cost_usd': 0.0,
                'estimated_cost_usd': 0.0,
            }
            # Log the failed question
            self.log_failed_question(question_id, question, "Model failed to generate valid response after 5 attempts")
        
        # Save result to CSV immediately
        self.save_result_to_csv(question_id, correct_answer, llm_answer, original_response)
        # Save usage (tokens/cost)
        self.save_usage_to_csv(question_id, usage_info)
        
        if llm_answer:
            is_correct = llm_answer == correct_answer.upper()
            if show_details:
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"LLM Answer: {llm_answer}")
                print(f"Result: {status}")
                print(f"💾 Saved to CSV: {self.results_csv}")
            
            return {
                'index': index,
                'question': question,
                'correct_answer': correct_answer,
                'llm_answer': llm_answer,
                'is_correct': is_correct,
                'subject': subject,
                'subdomain': subdomain,
                'question_id': question_id
            }
        else:
            if show_details:
                print(f"❌ Failed to get LLM response")
                print(f"💾 Saved failure to CSV: {self.results_csv}")
            return None
    
    def process_single_question_threaded(self, df, index):
        """Process a single question for threading (no detailed output)"""
        try:
            row = df.iloc[index]
            question_id = self._get_row_question_id(row, fallback=index)
            question = row['question']
            correct_answer = row['correct_answer']
            options_str = row['options']
            subject = row['subject']
            subdomain = row['subdomain_name']
            
            # Parse options
            options = self.parse_options(options_str)
            if len(options) != 4:
                return None
            
            # Ask LLM
            llm_result = self.ask_llm(question, options)
            if llm_result[0] is not None:
                llm_answer, original_response, usage_info = llm_result
                
                # Check if this was a random answer
                if original_response.startswith("RANDOM_"):
                    self.log_random_answer(question_id, question, llm_answer)
            else:
                llm_answer, original_response, usage_info = None, "Failed after 5 attempts", {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'prompt_cost_usd': 0.0,
                    'completion_cost_usd': 0.0,
                    'estimated_cost_usd': 0.0,
                }
                # Log the failed question
                self.log_failed_question(question_id, question, "Model failed to generate valid response after 5 attempts")
            
            # Save result to CSV immediately (thread-safe)
            self.save_result_to_csv(question_id, correct_answer, llm_answer, original_response)
            # Save usage (tokens/cost)
            self.save_usage_to_csv(question_id, usage_info)
            
            # Clear GPU cache periodically
            self.clear_gpu_cache()
            
            if llm_answer:
                is_correct = llm_answer == correct_answer.upper()
                # Update running stats for live tqdm display
                self.update_accuracy_stats(is_correct)
                return {
                    'index': index,
                    'question': question,
                    'correct_answer': correct_answer,
                    'llm_answer': llm_answer,
                    'is_correct': is_correct,
                    'subject': subject,
                    'subdomain': subdomain,
                    'question_id': question_id
                }
            return None
        except Exception as e:
            print(f"⚠️ Error processing question {index}: {e}")
            return None
    
    def test_multiple_questions(self, df, num_questions=10, random_selection=True, resume=True):
        """Test multiple questions and return results"""
        results = []
        
        # Get already processed questions if resuming
        processed_ids = set()
        if resume:
            processed_ids = self.get_processed_question_ids()
        
        if random_selection:
            # Filter out already processed questions
            available_indices = []
            for i in range(len(df)):
                row_i = df.iloc[i]
                qid_val = self._get_row_question_id(row_i)
                question_id = str(qid_val) if qid_val is not None else str(i)
                if not resume or question_id not in processed_ids:
                    available_indices.append(i)
            
            if len(available_indices) == 0:
                print("✅ All questions already processed!")
                return results
            
            indices = random.sample(available_indices, min(num_questions, len(available_indices)))
        else:
            # Sequential processing, skip already processed
            indices = []
            for i in range(len(df)):
                if len(indices) >= num_questions:
                    break
                row_i = df.iloc[i]
                qid_val = self._get_row_question_id(row_i)
                question_id = str(qid_val) if qid_val is not None else str(i)
                if not resume or question_id not in processed_ids:
                    indices.append(i)
        
        if not indices:
            print("✅ All requested questions already processed!")
            return results
        
        remaining = len(indices)
        print(f"\n🚀 TESTING {remaining} QUESTIONS")
        if resume and processed_ids:
            print(f"📋 Resuming: {len(processed_ids)} already processed, {remaining} remaining")
        print(f"{'='*60}")
        
        # Batch process with vLLM for throughput
        results = []
        pbar = tqdm(range(0, len(indices), actual_batch_size), desc="Processing batches")
        for start in pbar:
            batch = indices[start:start + actual_batch_size]
            batch_results = self.process_questions_batch(df, batch)
            if batch_results:
                results.extend(batch_results)
            # Update pbar text with live stats
            if self.total_processed > 0:
                try:
                    pbar.set_postfix({'acc_%': f'{self.get_current_accuracy():.1f}', 'random': self.random_count, 'done': self.total_processed})
                except Exception:
                    pass
            # Clear GPU cache every few batches (no-op for vLLM but kept)
            if (start // max(1, actual_batch_size)) % 10 == 0:
                self.clear_gpu_cache()
        return results
    
    def process_all_questions(self, df, start_from_beginning=False):
        """Process all questions in the dataset, resuming from where left off"""
        processed_ids = set()
        if not start_from_beginning:
            processed_ids = self.get_processed_question_ids()
        
        # Find unprocessed questions
        unprocessed_indices = []
        for i in range(len(df)):
            row_i = df.iloc[i]
            qid_val = self._get_row_question_id(row_i)
            question_id = str(qid_val) if qid_val is not None else str(i)
            if start_from_beginning or question_id not in processed_ids:
                unprocessed_indices.append(i)
        
        if not unprocessed_indices:
            print("✅ All questions already processed!")
            return []
        
        total_questions = len(df)
        remaining = len(unprocessed_indices)
        processed_count = total_questions - remaining
        
        print(f"\n🚀 PROCESSING ALL QUESTIONS")
        print(f"{'='*60}")
        print(f"Total questions: {total_questions:,}")
        print(f"Already processed: {processed_count:,}")
        print(f"Remaining: {remaining:,}")
        print(f"{'='*60}")
        
        # Process questions individually with Unsloth optimizations
        results = []

        # Process each question one by one with real-time accuracy
        pbar = tqdm(unprocessed_indices, desc="Processing questions")
        pbar.set_description(f"Processing | Acc: 0.0% | Random: {self.random_count}")
        for index in pbar:
            result = self.process_single_question_threaded(df, index)
            if result:
                results.append(result)

            # Update progress bar with real-time accuracy
            if self.total_processed > 0:
                accuracy = self.get_current_accuracy()
                pbar.set_description(f"Processing | Acc: {accuracy:.1f}% | Random: {self.random_count}")
                try:
                    pbar.set_postfix({'acc_%': f'{accuracy:.1f}', 'random': self.random_count, 'done': self.total_processed})
                except Exception:
                    pass

            # Clear GPU cache every 20 questions
            if self.total_processed % 20 == 0:
                self.clear_gpu_cache()

            # Small delay
            time.sleep(0.1)
        
        return results
    
    def analyze_results(self, results):
        """Analyze and display results"""
        if not results:
            print("❌ No results to analyze")
            return
        
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_answers / total_questions) * 100
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTS ANALYSIS")
        print(f"{'='*60}")
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Incorrect Answers: {total_questions - correct_answers}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"💾 All results saved to: {self.results_csv}")
        
        # Analyze by subject
        subject_stats = {}
        for result in results:
            subject = result['subject']
            if subject not in subject_stats:
                subject_stats[subject] = {'total': 0, 'correct': 0}
            subject_stats[subject]['total'] += 1
            if result['is_correct']:
                subject_stats[subject]['correct'] += 1
        
        print(f"\n📚 ACCURACY BY SUBJECT:")
        for subject, stats in subject_stats.items():
            subject_accuracy = (stats['correct'] / stats['total']) * 100
            print(f"  {subject}: {stats['correct']}/{stats['total']} ({subject_accuracy:.1f}%)")
        
        # Show incorrect answers
        incorrect_results = [r for r in results if not r['is_correct']]
        if incorrect_results:
            print(f"\n❌ INCORRECT ANSWERS:")
            for result in incorrect_results:
                print(f"  Q{result['index'] + 1}: Expected {result['correct_answer']}, Got {result['llm_answer']} - {result['subject']}")
        
        return {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'subject_stats': subject_stats,
            'results': results
        }
    
    def save_results(self, results, filename='llm_test_results.json'):
        """Save results to a JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main function"""
    print("🤖 TigerLLM Question Answerer (Unsloth Optimized)")
    print("=" * 50)

    # Initialize the answerer
    answerer = QuestionAnswerer()

    # Load dataset
    # Use the hard overlap dataset (15,074 questions)
    df = answerer.load_dataset("finished/verified_dataset_overlap_with_hard_subset.csv")
    if df is None:
        return

    # Test options
    print(f"\nChoose an option:")
    print(f"1. Test a single random question")
    print(f"2. Test multiple questions (specify number)")
    print(f"3. Test a specific question by index")
    print(f"4. Process ALL questions (resume from where left off)")
    print(f"5. Process ALL questions (start from beginning)")

    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Test single random question
            result = answerer.test_single_question(df)
            if result:
                answerer.analyze_results([result])
        
        elif choice == '2':
            # Test multiple questions
            num_questions = int(input("How many questions to test? "))
            results = answerer.test_multiple_questions(df, num_questions)
            analysis = answerer.analyze_results(results)
            if analysis:
                answerer.save_results(analysis)
        
        elif choice == '3':
            # Test specific question
            index = int(input(f"Enter question index (0-{len(df)-1}): "))
            if 0 <= index < len(df):
                result = answerer.test_single_question(df, index)
                if result:
                    answerer.analyze_results([result])
            else:
                print(f"❌ Invalid index. Must be between 0 and {len(df)-1}")
        
        elif choice == '4':
            # Process all questions (resume)
            print("🔄 Processing all questions, resuming from where left off...")
            results = answerer.process_all_questions(df, start_from_beginning=False)
            analysis = answerer.analyze_results(results)
            if analysis:
                answerer.save_results(analysis, f'final_analysis_{chudling_pong}.json')
        
        elif choice == '5':
            # Process all questions (start over)
            confirm = input("⚠️ This will reprocess ALL questions. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                print("🔄 Processing all questions from the beginning...")
                results = answerer.process_all_questions(df, start_from_beginning=True)
                analysis = answerer.analyze_results(results)
                if analysis:
                    answerer.save_results(analysis, 'final_analysis.json')
            else:
                print("❌ Cancelled")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print(f"\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
