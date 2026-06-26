# BnMMLU Scripts

This directory contains the public code needed to build evaluation inputs, run BnMMLU evaluations, and reproduce the main analysis tables/figures.

## Layout

- `evaluation/`: model evaluation scripts for API-based models and vLLM-based open-weight models.
- `batch/`: JSONL batch builders, upload helpers, and collectors for OpenAI/Anthropic batch workflows.
- `dataset/`: dataset cleaning, deduplication, verification, and BnMMLU-HARD construction helpers.
- `analysis/`: scoring, error analysis, usage aggregation, and plotting scripts.

Generated CSV/JSONL/PDF/PNG outputs are intentionally not tracked. The dataset is distributed through Hugging Face:

https://huggingface.co/datasets/samanjoy2/BnMMLU

## Environment

Install the common dependencies with:

```bash
pip install -r requirements.txt
```

For vLLM evaluation scripts, use a Linux/CUDA environment and install:

```bash
pip install -r requirements-vllm.txt
```

Copy `.env.example` to `.env` locally and fill in only the keys needed for the scripts you run. Do not commit `.env`.
