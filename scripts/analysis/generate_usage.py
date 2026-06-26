#!/usr/bin/env python3
"""
Generate token usage CSVs for direct (non-reasoning) benchmarks.

This script reconstructs the prompts that were sent to specific models,
counts prompt/completion/reasoning tokens with the model's own tokenizer,
and writes usage files that mirror the format:

question_id,model_name,timestamp,prompt_tokens,completion_tokens,total_tokens,reasoning_tokens,cached_prompt_tokens

Supported models (extendable):
  - deepseek-chat (local tokenizer in ../deepseek_v3_tokenizer)
  - Qwen/Qwen3-{1.7B,4B,8B,14B,32B} (no thinking)
  - Meta Llama 3.x Instruct, Gemma 3 IT, Unsloth finetunes (mapped to base tokenizers)
  - Proprietary (OpenAI/Anthropic/Google/xAI/Qwen Plus): counts via local fallback tokenizer

Usage example (run inside `0-shot Direct (Non-Reasoning)`):

    python generate_usage.py \
        --results "llm_test_results_*.csv" \
        --questions "../BnMMLU-HARD - DATASET.csv" \
        --output-dir usage

Multiple --results arguments are accepted. Use --limit for a dry run on the
first N rows, and --force to overwrite existing usage files.

Notes:
- Questions not present in the provided questions CSV are skipped.
- For proprietary models without local tokenizers, tokenization falls back to the
  local DeepSeek tokenizer in ../deepseek_v3_tokenizer for consistent counting.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - dependency missing
    raise SystemExit(
        "The generate_usage.py script requires the `transformers` package. "
        "Install it with `pip install transformers`."
    ) from exc

# Allow very large CSV fields (long raw outputs)
try:
    csv.field_size_limit(sys.maxsize)
except Exception:
    try:
        csv.field_size_limit(2**31 - 1)
    except Exception:
        pass

# Optional GPT tokenizer for OpenAI-like models
try:  # pragma: no cover - optional dependency
    from gpt3_tokenizer import GPT3Tokenizer as _GPT3Tokenizer
except Exception:  # pragma: no cover - keep optional
    _GPT3Tokenizer = None  # type: ignore

# Optional tokencost for broad model token counting
try:  # pragma: no cover - optional dependency
    import tokencost as _tokencost  # type: ignore
except Exception:  # pragma: no cover - keep optional
    _tokencost = None  # type: ignore

# Global model hint for tokencost usage per file being processed
CURRENT_MODEL_FOR_TOKENS: Optional[str] = None


def map_model_name_for_tokencost(model_name: str) -> Optional[str]:
    """Map dataset model names to tokencost canonical IDs.

    Returns a string if a confident mapping exists; otherwise None to indicate
    we should fall back to local tokenizers.
    """
    name = (model_name or "").strip()
    low = name.lower()

    # Skip Claude entirely per user request
    if "claude" in low:
        return None

    # DeepSeek
    if low.startswith("deepseek-chat"):
        return "deepseek-chat"

    # OpenAI naming adjustments
    # Map gpt-4.1-nano* to gpt-4o-mini (closest public analogue in tokencost)
    if low.startswith("gpt-4.1-nano") or low.startswith("gpt_4_1_nano"):
        return "gpt-4o-mini"
    # Some internal aliases observed in files
    if low.startswith("gpt-4.1-"):
        return "gpt-4o"
    if low.startswith("gpt-4o-mini"):
        return "gpt-4o-mini"
    if low.startswith("gpt-4o"):
        return "gpt-4o"
    if low.startswith("gpt-4-turbo"):
        return "gpt-4-turbo"
    if low.startswith("gpt-3.5-turbo") or low.startswith("gpt_3_5_turbo"):
        return "gpt-3.5-turbo"
    # Hypothetical gpt-5* → map to gpt-4o as best-effort for counting
    if low.startswith("gpt-5") or low.startswith("gpt_5"):
        return "gpt-4o"

    # Gemini naming
    if "gemini_2_5_flash_lite" in low:
        # No exact 2.5 in tokencost yet; use closest flash
        return "gemini/gemini-1.5-flash"
    if "gemini 2.5" in low or "gemini_2_5" in low:
        return "gemini/gemini-1.5-pro"
    if low.startswith("gemini/") or low.startswith("google/gemini"):
        # Already canonical
        return name

    # Grok (xAI) currently not in tokencost tables → fall back
    if low.startswith("grok-") or "grok" in low:
        return None

    # Qwen Plus cloud SKU
    if "qwen_plus" in low or "qwen-plus" in low:
        # No canonical in tokencost; fall back
        return None

    # Everything else (HF/local) → fall back
    return None

# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionRecord:
    """Container for a question and its options."""

    question: str
    options: Tuple[str, str, str, str]


class QuestionBank:
    """Loads questions/options from a CSV file keyed by question_id.

    This supports both legacy merged CSVs (with `Unique_Serial`) and the
    BnMMLU-HARD dataset (with `question_id`). Questions that are not present in
    the provided CSV will be considered missing and skipped.
    """

    def __init__(self, csv_path: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Question CSV not found: {csv_path}")
        self._data: Dict[str, QuestionRecord] = {}
        self._load(csv_path)

    @staticmethod
    def _parse_options(raw: Any) -> Tuple[str, str, str, str]:
        """Parse options stored as JSON / Python list / delimited text."""
        if raw is None:
            options: List[str] = []
        else:
            text = str(raw).strip()
            if not text:
                options = []
            else:
                options = []
                # Try JSON
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        options = [str(x).strip() for x in parsed]
                except Exception:
                    pass
                # Try literal eval
                if not options:
                    try:
                        parsed = ast.literal_eval(text)
                        if isinstance(parsed, (list, tuple)):
                            options = [str(x).strip() for x in parsed]
                    except Exception:
                        pass
                # Extract quoted segments
                if not options:
                    for match in re.finditer(r"'([^']*)'|\"([^\"]*)\"", text):
                        capture = match.group(1) if match.group(1) is not None else match.group(2)
                        options.append(capture.strip())
                # Fallback comma split
                if not options:
                    options = [part.strip().strip("'\"") for part in text.strip("[]").split(",") if part.strip()]

        # Ensure 4 items (pad/truncate)
        padded = (options + ["", "", "", ""])[:4]
        return tuple(padded)  # type: ignore[return-value]

    def _load(self, csv_path: Path) -> None:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])

            # Support both legacy and hard-dataset schemas
            key_col = None
            if "Unique_Serial" in fieldnames:
                key_col = "Unique_Serial"
            elif "question_id" in fieldnames:
                key_col = "question_id"

            if key_col is None or "question" not in fieldnames or "options" not in fieldnames:
                raise ValueError(
                    "Question CSV missing required columns. Expected either "
                    "{Unique_Serial,question,options} or {question_id,question,options}."
                )

            for row in reader:
                qid = str(row.get(key_col) or "").strip()
                if not qid:
                    continue
                question_text = str(row.get("question") or "").strip()
                options = self._parse_options(row.get("options"))
                self._data[qid] = QuestionRecord(question=question_text, options=options)

    def get(self, question_id: str) -> Optional[QuestionRecord]:
        return self._data.get(str(question_id).strip())


# --------------------------------------------------------------------------------------
# Token counting utilities
# --------------------------------------------------------------------------------------


class TokenizerCache:
    """Loads and caches tokenizers keyed by identifier + kwargs."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}

    def get(self, identifier: Union[str, Path], **kwargs: Any):
        norm_id = str(identifier)
        # Special path for GPT3-BPE virtual tokenizer
        if norm_id.startswith("gpt3-bpe:"):
            key = (norm_id, ())
            if key not in self._cache:
                model_hint = norm_id.split(":", 1)[1]
                self._cache[key] = GPT3BPETokenizerWrapper(model_hint)
            return self._cache[key]

        # default to offline usage; caller can override
        if "local_files_only" not in kwargs:
            kwargs["local_files_only"] = True
        key = (norm_id, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            tokenizer = AutoTokenizer.from_pretrained(norm_id, **kwargs)
            self._cache[key] = tokenizer
        return self._cache[key]


class GPT3BPETokenizerWrapper:
    """Thin wrapper to provide an .encode(text) compatible API for GPT models.

    Preference order:
      1) gpt3_tokenizer (pure-Python GPT-3 BPE)
      2) tokencost (if it exposes a token counting function)
    """

    def __init__(self, model_name_hint: str = "gpt-4.1") -> None:
        self.model_name_hint = model_name_hint
        if _GPT3Tokenizer is not None:
            try:
                self._impl = _GPT3Tokenizer()
                self._mode = "gpt3_tokenizer"
                return
            except Exception:
                pass
        # Fallback to tokencost if available
        if _tokencost is not None:
            self._impl = _tokencost
            self._mode = "tokencost"
            return
        raise RuntimeError(
            "GPT tokenizer not available. Install gpt3_tokenizer (preferred) or tokencost."
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return []
        if getattr(self, "_mode", "") == "gpt3_tokenizer":
            return list(self._impl.encode(text))  # type: ignore[attr-defined]
        # tokencost fallback: best-effort to emulate encode length
        if hasattr(self._impl, "count_string_tokens"):
            n = int(self._impl.count_string_tokens(text, self.model_name_hint))
            # Return dummy token IDs of the right length
            return list(range(n))
        # Final generic fallback if tokencost present but no function
        if hasattr(self._impl, "get_token_count"):
            n = int(self._impl.get_token_count(text, self.model_name_hint))
            return list(range(n))
        # If everything fails, degrade gracefully
        return list(self._impl.encode(text))  # may raise if method missing


def _flatten_token_ids(token_ids: Any) -> List[int]:
    """Normalize tokenizer outputs into a flat list of token ids."""
    if token_ids is None:
        return []
    if isinstance(token_ids, list):
        if token_ids and isinstance(token_ids[0], list):
            return token_ids[0]
        return token_ids
    if hasattr(token_ids, "tolist"):
        values = token_ids.tolist()
        if isinstance(values, list) and values and isinstance(values[0], list):
            return values[0]
        return values
    raise TypeError(f"Unsupported token id container: {type(token_ids)!r}")


def _tokencost_count_string(text: str) -> Optional[int]:
    global CURRENT_MODEL_FOR_TOKENS, _tokencost
    if _tokencost is None or not CURRENT_MODEL_FOR_TOKENS:
        return None
    # Try common tokencost entry points
    try:
        if hasattr(_tokencost, "count_string_tokens"):
            return int(_tokencost.count_string_tokens(text, CURRENT_MODEL_FOR_TOKENS))
        if hasattr(_tokencost, "get_token_count"):
            return int(_tokencost.get_token_count(text, CURRENT_MODEL_FOR_TOKENS))
    except Exception:
        return None
    return None


def count_text_tokens(tokenizer, text: str, *, add_special_tokens: bool = False) -> int:
    """Count tokens for plain text using tokencost when available, else tokenizer."""
    if not text:
        return 0
    n = _tokencost_count_string(text)
    if isinstance(n, int) and n >= 0:
        return n
    # Prefer fast encode when available
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return len(token_ids)
    # Fallback to __call__
    result = tokenizer(text, add_special_tokens=add_special_tokens, return_attention_mask=False)
    token_ids = result.get("input_ids")
    return len(_flatten_token_ids(token_ids))


def _tokencost_count_messages(messages: Sequence[Dict[str, str]]) -> Optional[int]:
    global CURRENT_MODEL_FOR_TOKENS, _tokencost
    if _tokencost is None or not CURRENT_MODEL_FOR_TOKENS:
        return None
    try:
        # Prefer count_message_tokens if available (per tokencost README)
        if hasattr(_tokencost, "count_message_tokens"):
            return int(_tokencost.count_message_tokens(list(messages), model=CURRENT_MODEL_FOR_TOKENS))
        # Some versions may expose pluralized name
        if hasattr(_tokencost, "count_messages_tokens"):
            return int(_tokencost.count_messages_tokens(list(messages), CURRENT_MODEL_FOR_TOKENS))
        # Fallback: approximate by concatenating contents
        joined = "\n\n".join(str(m.get("content", "")) for m in messages)
        return _tokencost_count_string(joined)
    except Exception:
        return None


def count_chat_tokens(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> int:
    """Count tokens for chat prompts: prefer tokencost, then tokenizer template, else serialize to text."""
    # Try tokencost first
    n = _tokencost_count_messages(messages)
    if isinstance(n, int) and n >= 0:
        return n
    # If tokenizer has chat template, try it
    if hasattr(tokenizer, "apply_chat_template"):
        chat_template_kwargs = chat_template_kwargs or {}
        try:
            token_ids = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **chat_template_kwargs,
            )
            return len(_flatten_token_ids(token_ids))
        except Exception:
            # fall back to plain-text serialization if template not set
            pass

    # Final fallback: serialize messages into a plain text prompt
    def _serialize_messages(msgs: Sequence[Dict[str, str]]) -> str:
        parts: List[str] = []
        for m in msgs:
            content = str(m.get("content") or "").strip()
            if content:
                parts.append(content)
        return "\n\n".join(parts)

    text = _serialize_messages(messages)
    return count_text_tokens(tokenizer, text, add_special_tokens=False)


# --------------------------------------------------------------------------------------
# Model-specific handlers
# --------------------------------------------------------------------------------------


THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def extract_reasoning_and_final(raw_text: Optional[str]) -> Tuple[str, str]:
    """Extract reasoning and final content from a raw response.

    Priority:
      1) <think>...</think> blocks for reasoning, everything after last as final.
      2) Lines prefixed with "Reasoning:" and trailing "Final:" JSON or text.
      3) Fallback: return ("", full_text)
    """
    if not raw_text:
        return "", ""
    text = raw_text.strip()
    if not text:
        return "", ""
    # Case 1: DeepSeek/Qwen style tags
    matches = list(THINK_TAG_PATTERN.finditer(text))
    if matches:
        reasoning_sections = [match.group(1).strip() for match in matches if match.group(1)]
        final_start = matches[-1].end()
        final_text = text[final_start:].strip()
        reasoning_text = "\n\n".join(filter(None, reasoning_sections))
        return reasoning_text, final_text
    # Case 2: "Reasoning:" / "Final:" patterns
    if "Final:" in text:
        parts = text.split("Final:", 1)
        reasoning_text = parts[0].strip()
        final_text = parts[1].strip()
        # Strip leading labels like "Reasoning:" if present
        reasoning_text = re.sub(r"^\s*Reasoning:\s*", "", reasoning_text, flags=re.IGNORECASE)
        return reasoning_text, final_text
    # Fallback
    return "", text


class ModelHandler:
    """Base class describing how to build prompts and parse outputs for a model."""

    def __init__(
        self,
        tokenizer_id: Union[str, Path],
        *,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        uses_chat_template: bool,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        prompt_special_tokens: bool = False,
        completion_special_tokens: bool = False,
    ) -> None:
        self.tokenizer_id = tokenizer_id
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.uses_chat_template = uses_chat_template
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.prompt_special_tokens = prompt_special_tokens
        self.completion_special_tokens = completion_special_tokens

    def build_prompt(
        self,
        question: QuestionRecord,
        row: Dict[str, Any],
    ) -> Union[str, Sequence[Dict[str, str]]]:
        raise NotImplementedError

    def extract_outputs(self, row: Dict[str, Any]) -> Tuple[str, str]:
        raise NotImplementedError


class DirectTextHandler(ModelHandler):
    """Plain-text instruction + question for models without chat templates."""

    SYSTEM_PROMPT = 'Respond in JSON format: {"answer": "A"} where A/B/C/D is your choice.'

    def __init__(self, tokenizer_id: Union[str, Path]) -> None:
        super().__init__(
            tokenizer_id=tokenizer_id,
            tokenizer_kwargs={"trust_remote_code": True},
            uses_chat_template=False,
            prompt_special_tokens=False,
            completion_special_tokens=False,
        )

    @staticmethod
    def _format_user(question: QuestionRecord) -> str:
        return (
            "Answer this multiple-choice question. Respond only with A, B, C, or D.\n\n"
            f"{question.question}\n\n"
            f"A) {question.options[0]}\n"
            f"B) {question.options[1]}\n"
            f"C) {question.options[2]}\n"
            f"D) {question.options[3]}\n\n"
            f"{DirectTextHandler.SYSTEM_PROMPT}"
        )

    def build_prompt(self, question: QuestionRecord, row: Dict[str, Any]) -> str:
        return self._format_user(question)

    def extract_outputs(self, row: Dict[str, Any]) -> Tuple[str, str]:
        raw = (
            row.get("original_llm_response")
            or row.get("raw_output")
            or row.get("final_content")
            or ""
        )
        if raw:
            return extract_reasoning_and_final(str(raw))
        # Fallback to the concise letter answer
        final = str(row.get("llm_answer") or "").strip()
        return "", final


class DirectChatHandler(ModelHandler):
    """Chat-style instruction + question for chat template capable models."""

    SYSTEM_PROMPT = 'Respond in JSON format: {"answer": "A"} where A/B/C/D is your choice.'

    def __init__(self, tokenizer_id: str, chat_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            tokenizer_id=tokenizer_id,
            tokenizer_kwargs={"trust_remote_code": True},
            uses_chat_template=True,
            chat_template_kwargs=(chat_kwargs or {}),
            prompt_special_tokens=False,
            completion_special_tokens=False,
        )

    @staticmethod
    def _format_user_content(question: QuestionRecord) -> str:
        return (
            f"Answer this multiple-choice question. Respond only with A, B, C, or D.\n\n"
            f"{question.question}\n\n"
            f"A) {question.options[0]}\n"
            f"B) {question.options[1]}\n"
            f"C) {question.options[2]}\n"
            f"D) {question.options[3]}\n"
        )

    def build_prompt(
        self,
        question: QuestionRecord,
        row: Dict[str, Any],
    ) -> Sequence[Dict[str, str]]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._format_user_content(question)},
        ]

    def extract_outputs(self, row: Dict[str, Any]) -> Tuple[str, str]:
        raw_text = (
            row.get("original_llm_response")
            or row.get("raw_output")
            or row.get("final_content")
            or ""
        )
        if raw_text:
            return extract_reasoning_and_final(str(raw_text))
        final = str(row.get("llm_answer") or "").strip()
        return "", final


def _normalize_model_name(model_name: str) -> str:
    s = (model_name or "").strip().lower()
    s = s.replace(" ", "-")
    s = s.replace("_", "-")
    s = s.replace("--", "-")
    return s


def resolve_model_handler(
    model_name: str,
    *,
    results_path: Path,
) -> ModelHandler:
    """Map model names to handlers for non-reasoning direct runs.

    For proprietary/cloud models without local tokenizers, we will still create a
    handler; token counting will fall back to the local DeepSeek tokenizer if the
    target tokenizer is not available offline (see process_results_file).
    """
    norm = _normalize_model_name(model_name)
    root_dir = Path(__file__).resolve().parent
    deepseek_tok_dir = (root_dir.parent / "deepseek_v3_tokenizer").resolve()
    if not deepseek_tok_dir.exists():
        alt = (root_dir / "deepseek_v3_tokenizer").resolve()
        if alt.exists():
            deepseek_tok_dir = alt

    # DeepSeek chat (local tokenizer provided in repo)
    if "deepseek-chat" in norm or norm == "deepseek":
        return DirectTextHandler(deepseek_tok_dir)

    # Qwen3 sizes and Plus
    qwen_map = {
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen3-4b": "Qwen/Qwen3-4B",
        "qwen3-8b": "Qwen/Qwen3-8B",
        "qwen3-14b": "Qwen/Qwen3-14B",
        "qwen3-32b": "Qwen/Qwen3-32B",
        "qwen-qwen3-plus": "Qwen/Qwen3-32B",  # approximate tokenizer for Plus
        "qwen/qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen/qwen3-4b": "Qwen/Qwen3-4B",
        "qwen/qwen3-8b": "Qwen/Qwen3-8B",
        "qwen/qwen3-14b": "Qwen/Qwen3-14B",
        "qwen/qwen3-32b": "Qwen/Qwen3-32B",
    }
    if norm in qwen_map:
        return DirectChatHandler(tokenizer_id=qwen_map[norm], chat_kwargs={"enable_thinking": False})

    # Meta Llama 3.x Instruct – use proper IDs
    if norm.startswith("meta-llama/llama-3.1-8b-instruct"):
        return DirectChatHandler("unsloth/Meta-Llama-3.1-8B-Instruct")
    if norm.startswith("meta-llama/llama-3.2-3b-instruct"):
        return DirectChatHandler("meta-llama/llama-3.2-3b-instruct")
    if norm.startswith("meta-llama/llama-3.3-70b-instruct") or "llama-3.3-70b-versatile" in norm:
        return DirectChatHandler("meta-llama/llama-3.3-70b-instruct")

    # Gemma 3 IT
    if norm.startswith("google/gemma-3-4b-it"):
        return DirectChatHandler("unsloth/gemma-3-4b-it")
    if norm.startswith("google/gemma-3-12b-it"):
        return DirectChatHandler("unsloth/gemma-3-12b-it")
    if norm.startswith("google/gemma-3-27b-it"):
        return DirectChatHandler("unsloth/gemma-3-27b-it")

    # Unsloth finetunes – when present, use Unsloth tokenizers directly
    if norm.startswith("unsloth/gemma-3-4b-it"):
        return DirectChatHandler("unsloth/gemma-3-4b-it")
    if norm.startswith("unsloth/gemma-3-12b-it"):
        return DirectChatHandler("unsloth/gemma-3-12b-it")
    if norm.startswith("unsloth/gemma-3-27b-it"):
        return DirectChatHandler("unsloth/gemma-3-27b-it")
    if norm.startswith("unsloth/meta-llama-3.1-8b-instruct"):
        return DirectChatHandler("unsloth/Meta-Llama-3.1-8B-Instruct")
    if norm.startswith("unsloth/llama-3.3-70b-instruct"):
        return DirectChatHandler("unsloth/Llama-3.3-70B-Instruct")
    if norm.startswith("unsloth/qwen3-4b-instruct"):
        return DirectChatHandler("unsloth/Qwen3-4B-Instruct-2507")

    # Titulm / Bangla / Tiger – use exact HF IDs
    if norm == "hishab/titulm-llama-3.2-1b-v1.1":
        return DirectChatHandler("hishab/titulm-llama-3.2-1b-v1.1")
    if norm == "hishab/titulm-llama-3.2-3b-v2.0":
        return DirectChatHandler("hishab/titulm-llama-3.2-3b-v2.0")
    if norm == "md-nishat-008/tigerllm-1b-it":
        return DirectChatHandler("md-nishat-008/TigerLLM-1B-it")
    if norm == "md-nishat-008/tigerllm-9b-it":
        return DirectChatHandler("md-nishat-008/TigerLLM-9B-it")
    if norm == "banglallm/banglallama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1":
        return DirectChatHandler("BanglaLLM/BanglaLLama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1")
    if norm == "banglallm/banglallama-3.2-1b-bangla-alpaca-orca-instruct-v0.0.1":
        return DirectChatHandler("BanglaLLM/BanglaLLama-3.2-1b-bangla-alpaca-orca-instruct-v0.0.1")
    if norm == "banglallm/banglallama-3.2-3b-bangla-alpaca-orca-instruct-v0.0.1":
        return DirectChatHandler("BanglaLLM/BanglaLLama-3.2-3b-bangla-alpaca-orca-instruct-v0.0.1")

    # GPT models – count using GPT3 tokenizer
    if norm.startswith("gpt-") or norm.startswith("gpt-") or "gpt-5" in norm or "gpt-4.1-nano" in norm:
        return DirectTextHandler(f"gpt3-bpe:{model_name}")

    # Proprietary clouds (Gemini/Grok/Qwen Plus) – will rely on fallback tokenizer
    if any(x in norm for x in ["grok-", "gemini", "qwen-plus"]):
        return DirectChatHandler(str(deepseek_tok_dir))

    # As a final fallback, use DeepSeek tokenizer with text-only prompt
    return DirectTextHandler(deepseek_tok_dir)


# --------------------------------------------------------------------------------------
# Core processing
# --------------------------------------------------------------------------------------


def compute_usage_filename(results_path: Path, output_dir: Path) -> Path:
    name = results_path.name
    if name.startswith("llm_test_results_"):
        target = "llm_usage_" + name[len("llm_test_results_") :]
    else:
        target = name.replace("llm_test_results", "llm_usage")
    return (output_dir / target).resolve()


def iter_rows(reader: csv.DictReader) -> Iterable[Dict[str, Any]]:
    for row in reader:
        yield {key: value for key, value in row.items()}


def process_results_file(
    results_path: Path,
    *,
    question_bank: QuestionBank,
    output_dir: Path,
    tokenizer_cache: TokenizerCache,
    limit: Optional[int],
    force: bool,
) -> None:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    usage_path = compute_usage_filename(results_path, output_dir)
    if usage_path.exists() and not force:
        print(f"[skip] {usage_path.name} already exists (use --force to overwrite)")
        return

    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            first_row = next(iter_rows(reader))
        except StopIteration:
            print(f"[warn] Results file has no rows: {results_path.name}")
            return

    # Resolve model name from results; prefer filename to avoid mislabeled rows
    inferred_from_filename = results_path.stem.replace("llm_test_results_", "")
    csv_model = first_row.get("model_name") or ""
    model_name = inferred_from_filename or csv_model
    # Explicitly skip Claude runs per user request
    if "claude" in (model_name or "").lower():
        print(f"[skip] Skipping Claude results: {results_path.name}")
        return
    handler = resolve_model_handler(model_name, results_path=results_path)
    # Set global model hint for tokencost (mapped to canonical if possible)
    global CURRENT_MODEL_FOR_TOKENS
    mapped = map_model_name_for_tokencost(model_name)
    CURRENT_MODEL_FOR_TOKENS = mapped or None

    # Try to load the intended tokenizer; if unavailable offline, fall back to DeepSeek local tokenizer
    deepseek_fallback = (Path(__file__).resolve().parent.parent / "deepseek_v3_tokenizer").resolve()
    if not deepseek_fallback.exists():
        alt = (Path(__file__).resolve().parent / "deepseek_v3_tokenizer").resolve()
        if alt.exists():
            deepseek_fallback = alt
    try:
        tokenizer = tokenizer_cache.get(handler.tokenizer_id, **handler.tokenizer_kwargs)
    except Exception as tok_exc:
        print(f"[warn] Tokenizer load failed for '{handler.tokenizer_id}': {tok_exc}")
        if deepseek_fallback.exists():
            print(f"[info] Falling back to DeepSeek tokenizer: {deepseek_fallback}")
            tokenizer = tokenizer_cache.get(str(deepseek_fallback), trust_remote_code=True, local_files_only=True)
        else:
            raise FileNotFoundError(
                "Neither the target tokenizer nor DeepSeek fallback tokenizer is available locally."
            )

    # Re-open reader to stream from beginning (including first row)
    processed = 0
    missing_questions = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_path.open("r", encoding="utf-8", newline="") as in_handle, usage_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        writer = csv.DictWriter(
            out_handle,
            fieldnames=[
                "question_id",
                "model_name",
                "timestamp",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cached_prompt_tokens",
                "accyrqcy",
            ],
        )
        writer.writeheader()

        correct = 0
        scored = 0
        for row_idx, row in enumerate(iter_rows(reader), start=1):
            question_id = str(row.get("question_id") or "").strip()
            if not question_id:
                continue
            question_record = question_bank.get(question_id)
            if question_record is None:
                missing_questions += 1
                if missing_questions <= 5:
                    print(f"[warn] Question ID {question_id} not found in question bank; skipping.")
                continue

            prompt_payload = handler.build_prompt(question_record, row)
            if handler.uses_chat_template:
                prompt_tokens = count_chat_tokens(
                    tokenizer,
                    prompt_payload,  # type: ignore[arg-type]
                    add_generation_prompt=True,
                    chat_template_kwargs=handler.chat_template_kwargs,
                )
            else:
                prompt_tokens = count_text_tokens(
                    tokenizer,
                    prompt_payload,  # type: ignore[arg-type]
                    add_special_tokens=handler.prompt_special_tokens,
                )

            reasoning_text, final_text = handler.extract_outputs(row)
            # Non-reasoning runs: set reasoning tokens to zero regardless of any provided text
            reasoning_tokens = 0
            completion_tokens = count_text_tokens(
                tokenizer, final_text, add_special_tokens=handler.completion_special_tokens
            )
            total_tokens = prompt_tokens + reasoning_tokens + completion_tokens

            # Determine per-row accuracy (1/0) from is_correct or by comparing answers
            acc_val = ""
            try:
                # Prefer explicit is_correct if present
                if "is_correct" in row and str(row.get("is_correct")).strip() != "":
                    is_corr = str(row.get("is_correct")).strip().lower() in ("1", "true", "yes")
                else:
                    gt = str(row.get("correct_answer") or "").strip().upper()
                    pred = str(row.get("llm_answer") or "").strip().upper()
                    is_corr = bool(gt) and (gt == pred)
                acc_val = 1 if is_corr else 0
                scored += 1
                if is_corr:
                    correct += 1
            except Exception:
                acc_val = ""

            writer.writerow(
                {
                    "question_id": question_id,
                    # Always write the resolved model name (filename-based) to avoid per-row mislabels
                    "model_name": model_name,
                    "timestamp": row.get("timestamp", ""),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "cached_prompt_tokens": row.get("cached_prompt_tokens") or 0,
                    "accyrqcy": acc_val,
                }
            )

            processed += 1
            if limit is not None and processed >= limit:
                break

            if processed % 1000 == 0:
                print(f"[info] {results_path.name}: processed {processed:,} rows...", flush=True)

    # Compute overall accuracy (if any rows had valid scoring)
    acc_summary = ""
    if 'correct' in locals() and scored:
        acc_pct = 100.0 * (correct / max(scored, 1))
        acc_summary = f", accuracy={acc_pct:.2f}% (scored={scored:,})"

    summary = f"[done] {usage_path.name}: {processed:,} rows{acc_summary}"
    if missing_questions:
        summary += f" (skipped {missing_questions} missing questions)"
    if limit is not None:
        summary += f" (limit={limit})"
    print(summary)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM usage CSVs from results.")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more result CSV paths (supports globbing if quoted).",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to the question CSV (e.g., ../BnMMLU-HARD - DATASET.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("usage"),
        help="Directory for generated usage CSVs (default: usage).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows per file (useful for smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing usage files instead of skipping.",
    )
    return parser.parse_args(argv)


def expand_results_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        expanded = list(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        if not expanded:
            print(f"[warn] Pattern matched no files: {pattern}")
        paths.extend(expanded)
    unique_paths = sorted(set(path.resolve() for path in paths))
    return unique_paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    question_bank = QuestionBank(args.questions.resolve())
    tokenizer_cache = TokenizerCache()
    results_files = expand_results_paths(args.results)

    if not results_files:
        raise SystemExit("No results files to process.")

    for results_path in results_files:
        try:
            process_results_file(
                results_path=results_path,
                question_bank=question_bank,
                output_dir=args.output_dir.resolve(),
                tokenizer_cache=tokenizer_cache,
                limit=args.limit,
                force=args.force,
            )
        except Exception as exc:
            print(f"[error] Failed to process {results_path}: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
