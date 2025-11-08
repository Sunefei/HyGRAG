The code implementation of HyGRAG, submitted to WWW' 26.

## HyGRAG

---
Fixed the incorrect reference to Appendix X on line 664. The latest PDF has been uploaded.

---

## Quick Start 🚀

### Dependencies

Ensure you have the required dependencies installed:
```bash
conda env create -f experiment.yml
```

##### Example Configuration (`config.yaml`):
```yaml
llm:
  api_type: "open_llm"  # Options: "openai" or "open_llm" (For Ollama and LlamaFactory) 
  model: "YOUR_LOCAL_MODEL_NAME"
  base_url: "YOUR_LOCAL_URL"  # Change this for local models
  api_key: "YOUR_API_KEY"  # Not required for local models
```

##### For `LlamaFactory` or `Ollama` or `vllm`, ensure the model is correctly installed and running in your local environment.

You can refer to the Readme of [`LlamaFactory`](https://github.com/hiyouga/LLaMA-Factory)
```yaml
llm:
  api_type: "open_llm"  # Options: "openai" or "open_llm" (For Ollama and LlamaFactory) 
  model: "YOUR_LOCAL_MODEL_NAME"
  base_url: "YOUR_LOCAL_URL"  # Change this for local models
  api_key: "ANY_THING_IS_OKAY"  # Not required for local models
```

### Run Methods

#### 1. start HyGRAG
```bash
python main.py -opt Option/Data/multihop-rag.yaml -dataset_name multihop-rag
```

#### 2. incremental test
```bash
python main_incremental.py -opt Option/Ours/HKGraphTreeDynamic.yaml -dataset_name multihop-rag -mode incremental -incremental_ratio 0.2
```

---