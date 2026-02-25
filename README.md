[![arXiv](https://img.shields.io/badge/arXiv-2601.12641-b31b1b.svg)](https://arxiv.org/abs/2601.12641)

# STEP-LLM (official)

Official implementation of our DATE 2026 paper:
**"STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models"**

STEP-LLM fine-tunes compact LLMs (Llama-3.2-3B and Qwen2.5-3B) to generate valid ISO 10303-21 STEP files directly from natural language descriptions, with optional Retrieval-Augmented Generation (RAG) for improved accuracy.

## Overview

| Component | Description |
|---|---|
| **Data Preparation** | Caption ABC dataset with GPT-4V, reorder STEP entities, build RAG index |
| **Training** | Fine-tune with [Unsloth](https://github.com/unslothai/unsloth) + LoRA on RAG-formatted STEP data |
| **Inference** | Generate STEP files with optional FAISS-based retrieval (`generate_step.py`) |
| **Evaluation** | Chamfer distance, Complete Ratio, renderability metrics |

## Project Structure

```
cad_codebased/
├── generate_step.py               # main inference entry point (argparse CLI)
├── retrieval.py                   # FAISS-based semantic retrieval
├── dataset_construct_rag.py       # build RAG training dataset
├── reorder_step.py                # reorder & renumber STEP entities (DFS)
├── render_step.py                 # render STEP files to multi-view images
├── round_step_numbers.py          # normalise STEP entity numbering
├── data_split.py                  # split dataset into train / val / test
├── llama3_SFT_response.py         # training script (reference; see also .ipynb)
├── captioning.ipynb               # GPT-4V captioning notebook
├── llama3_SFT_response.ipynb      # training notebook
├── requirements.txt
├── environment_minimal.yml
├── cad_captions_0-500.csv         # released captions (models 0–500)
├── cad_captions_500-1000.csv      # released captions (models 500–1000)
├── medium_8000_simple.csv         # simple-complexity captions
├── medium_8000_medium.csv         # medium-complexity captions
├── medium_8000_complex.csv        # complex-complexity captions
├── scripts/
│   ├── download_checkpoints.sh    # download released LoRA adapters
│   ├── merge_lora_adapter.py      # merge LoRA adapter into base model
│   ├── setup.sh                   # one-command environment setup
│   ├── download_abc_dataset.sh    # download ABC dataset
│   ├── download_base_models.sh    # download Llama / Qwen base models
│   └── process_dataset.sh         # full data processing pipeline
├── examples/
│   ├── basic_inference.py         # simple text → STEP example
│   └── rag_inference.py           # RAG-augmented generation example
├── eval_ckpt/
│   ├── step_chamfer_reward.py     # Chamfer distance evaluation
│   ├── generate_step_multi_gpu.py # multi-GPU batch generation
│   └── ...
└── data_filter_long/              # token-count filtering tools
```

---

## Quick Start

### 1. Setup Environment

```bash
bash scripts/setup.sh          # creates conda env 'step_llm'
conda activate step_llm
```

Or manually:

```bash
conda create -n step_llm python=3.10 -y
conda activate step_llm
pip install -r requirements.txt
conda install -c conda-forge pythonocc-core -y   # for STEP file processing
```

### 2. Download Base Model

```bash
# Option A — Qwen2.5-3B-Instruct (recommended)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct

# Option B — Llama-3.2-3B-Instruct (requires Meta access)
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./models/Llama-3.2-3B-Instruct
```

### 3. Download Fine-tuned LoRA Adapters

We release two LoRA adapters (100–200 MB each):

| Model | Base | HuggingFace | GitHub Release |
|---|---|---|---|
| STEP-LLM-Llama3B | Llama-3.2-3B-Instruct | [YOUR_HF_USERNAME/step-llm-llama3b] | [v1.0 release] |
| STEP-LLM-Qwen3B  | Qwen2.5-3B-Instruct   | [YOUR_HF_USERNAME/step-llm-qwen3b]  | [v1.0 release] |

```bash
# After filling in the URLs in scripts/download_checkpoints.sh:
bash scripts/download_checkpoints.sh        # both adapters
bash scripts/download_checkpoints.sh qwen   # Qwen only
bash scripts/download_checkpoints.sh llama  # Llama only
```

> **Note:** The adapters can be used directly at inference time (see Step 4).
> Merging into a full model is optional — see [Merge LoRA Adapter](#merge-lora-adapter-optional).

### 4. Run Inference

```bash
# Without RAG (simplest):
python generate_step.py \
    --ckpt_path ./checkpoints/step-llm-qwen3b \
    --caption   "A cylindrical bolt with a hexagonal head" \
    --save_dir  ./generated

# With RAG (better accuracy; requires the processed dataset):
python generate_step.py \
    --ckpt_path    ./checkpoints/step-llm-qwen3b \
    --use_rag \
    --db_csv_path  ./cad_captions_0-500.csv \
    --step_json_dir ./data/abc_rag/20500_dfs \
    --caption      "A cylindrical bolt with a hexagonal head" \
    --save_dir     ./generated \
    --output_name  bolt.step
```

See `python generate_step.py --help` for all options.

---

## Model Checkpoints

### LoRA Adapters (released)

The released checkpoints are LoRA adapters, not full models. This keeps file
sizes small (~150 MB vs ~6 GB) and respects base model licenses.

| Checkpoint | Base model | Dataset | Steps |
|---|---|---|---|
| step-llm-llama3b | Llama-3.2-3B-Instruct | 20k DFS-reordered STEP files | 7200 |
| step-llm-qwen3b  | Qwen2.5-3B-Instruct   | 20k DFS-reordered STEP files | 9000 |

### Merge LoRA Adapter (optional)

Merging produces a single standalone model that loads faster and needs no
PEFT dependency at inference time:

```bash
python scripts/merge_lora_adapter.py \
    --base_model_path Qwen/Qwen2.5-3B-Instruct \
    --adapter_path    ./checkpoints/step-llm-qwen3b \
    --output_path     ./merged_model/step-llm-qwen3b-merged

# Then run inference on the merged model:
python generate_step.py \
    --ckpt_path ./merged_model/step-llm-qwen3b-merged \
    --caption   "A flange with bolt holes"
```

---

## Dataset

### Released Captions

We release the GPT-4V captions we generated for the ABC dataset. The caption
CSV files are included in this repo:

| File | Models | Description |
|---|---|---|
| `cad_captions_0-500.csv`   | ~10k | Captions for ABC chunks 0001–0500 |
| `cad_captions_500-1000.csv`| ~10k | Captions for ABC chunks 0501–1000 |
| `medium_8000_simple.csv`   | ~3k  | Simple-geometry subset |
| `medium_8000_medium.csv`   | ~3k  | Medium-geometry subset |
| `medium_8000_complex.csv`  | ~3k  | Complex-geometry subset |

Columns: `model_id`, `description`, `isDescribable`

### ABC Dataset (download separately)

The STEP files themselves come from the ABC dataset (NYU):

```bash
bash scripts/download_abc_dataset.sh
# or visit: https://archive.nyu.edu/handle/2451/43778
```

Place downloaded chunks under `data/abccad/`.

### Build the Full RAG Dataset

After downloading the ABC dataset and the captions:

```bash
# 1. Reorder STEP entities (DFS order, eliminates forward references)
python reorder_step.py data/abccad/ --out-dir data/dfs_step/

# 2. Normalise entity numbering
python round_step_numbers.py   # (configure paths inside script)

# 3. Build RAG dataset (pairs each STEP file with a retrieved similar example)
python dataset_construct_rag.py

# 4. Split into train / val / test
python data_split.py

# Or run the full pipeline:
bash scripts/process_dataset.sh
```

See `docs/DATASET.md` for details.

---

## Training

Training uses [Unsloth](https://github.com/unslothai/unsloth) for efficient
LoRA fine-tuning (2× faster, ~30% less VRAM than standard PEFT).

### Prompt Template

The model is trained on this format (used at both training and inference time):

```
You are a CAD model generation assistant trained to produce STEP (.step) files
based on textual descriptions. Given the following object description and
relevant retrieved CAD data, generate a STEP file that accurately represents
the described object.

### caption:
{natural language description}

### retrieved relevant step file:
{retrieved STEP DATA section}   ← omitted when USE_RAG=False

### output:
{target STEP DATA section}
```

### Run Training

Edit the configuration block at the top of `llama3_SFT_response.py` (or use
the notebook `llama3_SFT_response.ipynb`), then:

```bash
python llama3_SFT_response.py
```

Key hyperparameters used for the released checkpoints:

| Parameter | Value |
|---|---|
| LoRA rank (`r`) | 16 |
| `lora_alpha` | 16 |
| Learning rate | 5e-5 |
| Batch size | 2 (× 4 grad. accum. = effective 8) |
| Warmup steps | 1200 |
| `max_seq_length` | 16384 |
| Optimiser | adamw_8bit |

---

## Evaluation

```bash
# Chamfer distance (shape similarity)
cd eval_ckpt
python step_chamfer_reward.py

# Multi-GPU batch generation
bash run_multi_gpu.sh
```

See `eval_ckpt/README_eval.md` and `eval_ckpt/README_step_chamfer_reward.md`.

---

## Important Notes

> **Hardcoded paths:** Several scripts (`dataset_construct_rag.py`,
> `retrieval.py`, `render_step.py`, etc.) still contain hardcoded absolute
> paths. Update these to match your local setup before running. A future
> release will add full argparse support to these scripts.

> **CUDA requirement:** Inference and training require an NVIDIA GPU with
> CUDA 11.8+. The model generates sequences up to 14,000 tokens, so at least
> 16 GB VRAM is recommended (24 GB+ for training).

---

## Citation

```bibtex
@article{shi2026step,
  title={STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models},
  author={Shi, Xiangyu and Ding, Junyang and Zhao, Xu and Zhan, Sinong and Mohapatra, Payal
          and Quispe, Daniel and Welbeck, Kojo and Cao, Jian and Chen, Wei and Guo, Ping and others},
  journal={arXiv preprint arXiv:2601.12641},
  year={2026}
}
```

---

## License

See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) — efficient LoRA fine-tuning
- [ABC Dataset](https://archive.nyu.edu/handle/2451/43778) — CAD model source
- [OpenCASCADE](https://www.opencascade.com/) (via `pythonocc-core`) — STEP file processing
- [FAISS](https://github.com/facebookresearch/faiss) — semantic retrieval index

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
