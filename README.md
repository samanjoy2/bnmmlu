# BnMMLU: Measuring Massive Multitask Language Understanding in Bengali

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![ACL 2026 Findings](https://img.shields.io/badge/ACL%202026-Findings-blue)](#citation)
[![arXiv](https://img.shields.io/badge/arXiv-2505.18951-b31b1b.svg)](https://arxiv.org/abs/2505.18951)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/samanjoy2/BnMMLU)

## News

Our paper, **"BnMMLU: Measuring Massive Multitask Language Understanding in Bengali"**, has been accepted to **Findings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)**.
The preprint is also available on arXiv: [arXiv:2505.18951](https://arxiv.org/abs/2505.18951).

## Overview

We present BnMMLU, a massive multitask language understanding benchmark for Bengali spanning 41 domains across STEM, humanities, social sciences, and general knowledge, with 134,375 multiple-choice question-option pairs. The dataset preserves mathematical content via MathML, introduces BnMMLU-HARD for stress testing, and benchmarks 24 model variants across 11 LLM families under standardized prompting and context regimes. The benchmark is introduced in our accepted ACL 2026 Findings paper.

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

Code for loading and evaluating models on the BnMMLU benchmark will be released and maintained in this repository.

## Citation

If you use BnMMLU in your research, please cite:

```bibtex
@misc{joy2026bnmmlumeasuringmassivemultitask,
  title={BnMMLU: Measuring Massive Multitask Language Understanding in Bengali},
  author={Saman Sarker Joy and Swakkhar Shatabda},
  year={2026},
  eprint={2505.18951},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2505.18951},
}
```

## License

This project is licensed under the CC BY-SA 4.0 license. See the LICENSE file for details.

## Contact

Saman Sarker Joy - saman.sarker.joy@gmail.com
