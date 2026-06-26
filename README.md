# BnMMLU: Measuring Massive Multitask Language Understanding in Bengali

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![ACL 2026 Findings](https://img.shields.io/badge/ACL%202026-Findings-blue)](https://aclanthology.org/2026.findings-acl.593/)
[![ACL Anthology PDF](https://img.shields.io/badge/PDF-ACL%20Anthology-red)](https://aclanthology.org/2026.findings-acl.593.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2505.18951-b31b1b.svg)](https://arxiv.org/abs/2505.18951)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/samanjoy2/BnMMLU)

## News

- June 2026: BnMMLU is officially published in **Findings of the Association for Computational Linguistics: ACL 2026**. Read it on the [ACL Anthology](https://aclanthology.org/2026.findings-acl.593/) or download the [PDF](https://aclanthology.org/2026.findings-acl.593.pdf).
- The preprint remains available on arXiv: [arXiv:2505.18951](https://arxiv.org/abs/2505.18951).

## Overview

We present BnMMLU, a massive multitask language understanding benchmark for Bengali spanning 41 domains across STEM, humanities, social sciences, and general knowledge, with 134,375 multiple-choice question-option pairs. The dataset preserves mathematical content via MathML, introduces BnMMLU-HARD for stress testing, and benchmarks 24 model variants across 11 LLM families under standardized prompting and context regimes. The benchmark is published in Findings of the Association for Computational Linguistics: ACL 2026.

## Dataset

The BnMMLU dataset consists of 134,375 multiple-choice question-option pairs across 41 domains. It preserves mathematical content via MathML and includes `BnMMLU-HARD`, a compact subset designed for more difficult stress testing. The dataset is available on Hugging Face:

[samanjoy2/BnMMLU](https://huggingface.co/datasets/samanjoy2/BnMMLU)

## Features

- Comprehensive evaluation across 41 domains
- 134,375 multiple-choice question-option pairs
- Mathematical content preserved via MathML
- `BnMMLU-HARD` subset for difficult-case stress testing
- Benchmarks covering 24 model variants across 11 LLM families
- Standardized evaluation under direct vs. chain-of-thought prompting and 0-shot vs. 5-shot settings

## Usage

Evaluation, batch-processing, dataset-construction, and analysis scripts are available in [`scripts/`](scripts/). Install the common dependencies with:

```bash
pip install -r requirements.txt
```

The vLLM-based open-weight model evaluators require a Linux/CUDA environment:

```bash
pip install -r requirements-vllm.txt
```

Use `.env.example` as the local environment template for API keys. Do not commit `.env` files.

## Citation

If you use BnMMLU in your research, please cite:

```bibtex
@inproceedings{joy-shatabda-2026-bnmmlu,
    title = "{B}n{MMLU}: Measuring Massive Multitask Language Understanding in {B}engali",
    author = "Joy, Saman Sarker  and
      Shatabda, Swakkhar",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.593/",
    pages = "12211--12230",
    ISBN = "979-8-89176-395-1",
    abstract = "Large-scale multitask benchmarks have driven rapid progress in language modeling, yet most emphasize high-resource languages such as English, leaving Bengali underrepresented. We present BnMMLU, a comprehensive benchmark for measuring massive multitask language understanding in Bengali. BnMMLU spans 41 domains across STEM, humanities, social sciences, and general knowledge, and contains 134,375 multiple-choice question{--}option pairs-the most extensive Bengali evaluation suite to date. The dataset preserves mathematical content via MathML, and includes BnMMLU-HARD, a compact subset constructed from questions most frequently missed by top systems to stress difficult cases. We benchmark 24 model variants across 11 LLM families, spanning open-weights general/multilingual, Bengali-centric open-weights, and proprietary models, covering multiple parameter scales and instruction-tuned settings. We evaluate models under standardized protocols covering two prompting styles (Direct vs. Chain-of-Thought) and two context regimes (0-shot vs. 5-shot), reporting accuracy consistently across families. Our analysis highlights persistent gaps in reasoning and application skills and indicates sublinear returns to scale across model sizes. We release the dataset and evaluation templates to support rigorous, reproducible assessment of Bengali language understanding and to catalyze progress in multilingual NLP."
}
```

## License

This project is licensed under the CC BY-SA 4.0 license. See the LICENSE file for details.

## Contact

Saman Sarker Joy - saman.sarker.joy@gmail.com
