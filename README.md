[![arXiv](https://img.shields.io/badge/arXiv-2601.12641-b31b1b.svg)](https://arxiv.org/abs/2601.12641)

# STEP-LLM (official)

Official implementation of our DATE 2026 paper:
**"STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models"**

STEP-LLM fine-tunes compact LLMs (Llama-3.2-3B and Qwen2.5-3B) to generate valid ISO 10303-21 STEP files directly from natural language descriptions, with optional Retrieval-Augmented Generation (RAG) for improved accuracy.

## Overview

| Component | Description |
|---|---|
| **Data Preparation** | Caption ABC dataset with GPT-4o, reorder STEP entities (DFS), build RAG index |
| **Training** | Fine-tune with [Unsloth](https://github.com/unslothai/unsloth) + LoRA on RAG-formatted STEP data |
| **Inference** | Generate STEP files with optional FAISS-based retrieval (`generate_step.py`) |
| **Evaluation** | Chamfer distance, Complete Ratio, renderability metrics |

## Project Structure

```
cad_codebased/
├── generate_step.py               # main inference entry point (argparse CLI)
├── retrieval.py                   # FAISS-based semantic retrieval
├── llama3_SFT_response.py         # training script (reference; see also .ipynb)
├── llama3_SFT_response.ipynb      # training notebook
├── requirements.txt
├── environment_minimal.yml
├── dataset/                       # captions + place to download ABC dataset / images
│   ├── cad_captions_0-500.csv     # released captions (0–500 entity STEP files)
│   ├── cad_captions_500-1000.csv  # released captions (500–1000 entity STEP files)
│   ├── abccad/                    # download ABC dataset here (not in repo)
│   └── rendered_images/           # download rendered images here (not in repo)
├── data_preparation/              # full data processing pipeline
│   ├── README.md                  # step-by-step pipeline instructions
│   ├── round_step_numbers.py      # normalise STEP floating-point precision
│   ├── step_restructurer.py       # DFS reorder + structural annotations
│   ├── batch_restructure.sh       # batch wrapper for step_restructurer.py
│   ├── restore_step_valid.py      # strip annotations → valid STEP file
│   ├── dataset_construct_rag.py   # build RAG training dataset (FAISS retrieval)
│   ├── data_split.py              # split dataset into train / val / test
│   └── captioning.ipynb           # GPT-4o captioning notebook
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
│   ├── eval_loss_by_ckpt.py       # checkpoint loss evaluation
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

# Option B — Llama-3.2-3B-Instruct (requires Meta access token)
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./models/Llama-3.2-3B-Instruct
```

### 3. Download Fine-tuned LoRA Adapters

We release two LoRA adapters (~150 MB each):

| Model | Base | HuggingFace |
|---|---|---|
| STEP-LLM-Llama3B | Llama-3.2-3B-Instruct | [JasonShiii/step-llm-llama3b](https://huggingface.co/JasonShiii/step-llm-llama3b) *(coming soon)* |
| STEP-LLM-Qwen3B  | Qwen2.5-3B-Instruct   | [JasonShiii/step-llm-qwen3b](https://huggingface.co/JasonShiii/step-llm-qwen3b) *(coming soon)* |

```bash
# Fill in the HuggingFace repo IDs in scripts/download_checkpoints.sh, then:
bash scripts/download_checkpoints.sh        # both adapters
bash scripts/download_checkpoints.sh qwen   # Qwen only
bash scripts/download_checkpoints.sh llama  # Llama only
```

> The adapters can be used directly at inference time without merging.
> See [Merge LoRA Adapter](#merge-lora-adapter-optional) if you prefer a standalone model.

### 4. Run Inference

```bash
# Without RAG (simplest):
python generate_step.py \
    --ckpt_path ./checkpoints/step-llm-qwen3b \
    --caption   "A cylindrical bolt with a hexagonal head" \
    --save_dir  ./generated

# With RAG (retrieves a similar example to guide generation):
python generate_step.py \
    --ckpt_path     ./checkpoints/step-llm-qwen3b \
    --use_rag \
    --db_csv_path   ./dataset/cad_captions_0-500.csv \
    --step_json_dir ./dataset/abc_rag/20500_dfs \
    --caption       "A cylindrical bolt with a hexagonal head" \
    --save_dir      ./generated \
    --output_name   bolt.step
```

See `python generate_step.py --help` for all options.

---

## Model Checkpoints

### LoRA Adapters (released)

The released checkpoints are LoRA adapters, not full models. This keeps file
sizes small (~150 MB vs ~6 GB) and respects base model licenses.

| Checkpoint | Base model | Training data | Steps |
|---|---|---|---|
| step-llm-llama3b | Llama-3.2-3B-Instruct | ~20k STEP files, 0–500 entities | 7200 |
| step-llm-qwen3b  | Qwen2.5-3B-Instruct   | ~20k STEP files, 0–500 entities | 9000 |

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

### STEP File Complexity and Token Length

STEP files can be extremely long — a single file often contains thousands of
entity lines, each consuming significant context window. To make fine-tuning
feasible within a 16k token context window, we filter the ABC dataset by
**entity count**:

| Entity range | Approx. token length | Used in |
|---|---|---|
| 0–500 entities    | ~500–8k tokens  | DATE 2026 paper **(this release)** |
| 500–1000 entities | ~8k–16k tokens  | Ongoing journal extension |

For the DATE submission, we use the **first 10 chunks** of the ABC dataset
(`abc_0001_step_v00` to `abc_0010_step_v00`, ~20k models) filtered to STEP
files with **0–500 entities**. The 500–1000 entity range is ongoing work and
not part of this release.

### Released Captions

We release the GPT-4o generated captions for the ABC dataset:

| File | Entity range | Description |
|---|---|---|
| `dataset/cad_captions_0-500.csv`    | 0–500 entities    | Captions used in the DATE paper |
| `dataset/cad_captions_500-1000.csv` | 500–1000 entities | Captions for ongoing journal extension |

Columns: `model_id`, `isDescribable`, `description`

### Rendered CAD Images

We provide rendered multi-view images of the ABC Dataset STEP files, hosted on HuggingFace:

**[JasonShiii/STEP-LLM-dataset](https://huggingface.co/datasets/JasonShiii/STEP-LLM-dataset)**

| Folder | Entity range | # Models | Description |
|---|---|---|---|
| `step_under500_image/` | 0–500 entities | ~20k | Images for DATE 2026 paper (this release) |
| `step_500-1000_image/` | 500–1000 entities | ~17k | Images for ongoing journal extension |

Each folder has 10 per-chunk zip files (~25 MB each for under-500, ~75–85 MB for 500–1000).
These images were used with GPT-4o to generate captions (`data_preparation/captioning.ipynb`).
The released caption CSVs are included in this repo — you do **not** need these images to run inference or reproduce training.

```python
# Download a single chunk via Python:
from huggingface_hub import hf_hub_download
hf_hub_download("JasonShiii/STEP-LLM-dataset",
                "step_under500_image/abc_0001_step_v00_under500_image.zip",
                repo_type="dataset", local_dir="./dataset/rendered_images")
```

### ABC Dataset (download separately)

The STEP files come from the [ABC Dataset](https://archive.nyu.edu/handle/2451/43778) (NYU).
We do **not** redistribute the STEP files — download them directly:

```bash
bash scripts/download_abc_dataset.sh
# or visit: https://archive.nyu.edu/handle/2451/43778
```

Place downloaded chunks under `dataset/abccad/`.

### Build the Full RAG Dataset

After downloading the ABC dataset and captions, the data preparation pipeline is:

```bash
# 1. Normalise floating-point precision in raw STEP files
python data_preparation/round_step_numbers.py dataset/abccad/step_under500/ \
    --output-dir dataset/rounded_step/

# 2. DFS reorder + annotate (eliminates forward references for LLM training)
#    Annotated files are valid STEP and training-ready — no stripping needed.
#    Edit SRC_BASE / DEST_BASE at the top of the script first:
bash data_preparation/batch_restructure.sh

# 3. Build RAG dataset (pairs each STEP file with a FAISS-retrieved similar example)
python data_preparation/dataset_construct_rag.py

# 4. Split into train / val / test
python data_preparation/data_split.py
```

See `data_preparation/README.md` for full step-by-step instructions.

---

## Training

Training uses [Unsloth](https://github.com/unslothai/unsloth) for efficient
LoRA fine-tuning (2× faster, ~30% less VRAM than standard PEFT).
All training in the DATE submission was performed on a **single GPU**.

### Prompt Template

The model is trained on this format (used consistently at both training and inference time):

**With RAG** (recommended):
```
You are a CAD model generation assistant trained to produce STEP (.step) files
based on textual descriptions. Given the following object description and
relevant retrieved CAD data, generate a STEP file that accurately represents
the described object.

### caption:
{natural language description}

### retrieved relevant step file:
{retrieved STEP DATA section}

### output:
{target STEP DATA section}
```

**Without RAG:**
```
You are a CAD model generation assistant trained to produce STEP (.step) files
based on textual descriptions. Given the following object description, generate
a STEP file that accurately represents the described object.

### caption:
{natural language description}

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
python eval_ckpt/step_chamfer_reward.py

# Complete Ratio (STEP file validity)
python eval_ckpt/CR/CR_calculate.py

# Renderability check
python eval_ckpt/renderability/check_renderability.py
```

See `eval_ckpt/README_eval.md` and `eval_ckpt/README_step_chamfer_reward.md`.

---

## Important Notes

> **CUDA requirement:** Inference and training require an NVIDIA GPU.
> STEP files can be very long (up to 16k tokens), so at least 16 GB VRAM
> is recommended for inference, and 24 GB+ for training.

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
