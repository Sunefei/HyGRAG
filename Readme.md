# HyGRAG: A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3774904.3792720-blue)](https://doi.org/10.1145/3774904.3792720)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The official implementation of **HyGRAG**, accepted at WWW 2026. [[Paper]](https://dl.acm.org/doi/10.1145/3774904.3792720)

## Quick Start

### Installation

```bash
conda env create -f experiment.yml
conda activate RAG
```

Or install dependencies via pip:

```bash
pip install -r requirements.txt
```

### Configuration

Set up your LLM configuration in `Option/Config2.yaml`:

```yaml
llm:
  api_type: "openai"           # Options: "openai", "open_llm" (for vLLM, Ollama, LLaMA-Factory)
  model: "gpt-4o"
  base_url: "https://api.openai.com/v1"
  api_key: "YOUR_API_KEY"
```

For local models (vLLM / Ollama / LLaMA-Factory):

```yaml
llm:
  api_type: "open_llm"
  model: "YOUR_MODEL_NAME"
  base_url: "http://localhost:8000/v1"
  api_key: "not-needed"
```

### Prepare Datasets

Download and place datasets under `Data/<dataset_name>/` with the following structure:

- `Corpus.json` — JSONL file with `title`, `context`, and `id` fields
- `Question.json` — JSONL file with `question`, `answer`, and optional `options` / `answer_idx` fields

Refer to `Data/datasets/README.md` for detailed format specifications.

### Run Methods

1. start HyGRAG
```bash
python main.py -opt Option/Data/multihop-rag.yaml -dataset_name multihop-rag
```
2. incremental test
```bash
python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode incremental -incremental_ratio 0.2
```

## Citation

If you use HyGRAG in your research, please cite our paper:

```bibtex
@inproceedings{10.1145/3774904.3792720,
  author = {Zhong, Haoyang and Sun, Yifei and Zhang, Antong and Wang, Chunping and Chen, Lei and Yang, Yang},
  title = {A Unified Framework for Context-Aware and Relation-Aware Graph Retrieval-Augmented Generation},
  year = {2026},
  doi = {10.1145/3774904.3792720},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  pages = {2477–2488},
  series = {WWW '26}
}
```

## License

This project is licensed under the MIT License.
