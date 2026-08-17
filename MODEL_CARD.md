# Model Card: STEP-LLM

## Model Details

### Model Description

STEP-LLM fine-tunes compact LLMs to generate valid ISO 10303-21 STEP files
directly from natural language descriptions, with optional Retrieval-Augmented
Generation (RAG) for improved accuracy.

- **Developed by**: Xiangyu Shi, Junyang Ding, Xu Zhao, Sinong Zhan, Payal Mohapatra, Daniel Quispe, Kojo Welbeck, Jian Cao, Wei Chen, Ping Guo, Qi Zhu (Northwestern University)
- **Model type**: Autoregressive language model with LoRA fine-tuning
- **Language**: English (input), STEP / ISO 10303-21 (output)
- **License**: MIT (code). Model weights are LoRA adapters and inherit the
  license of their base model — [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
  for the Llama variants, [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE)
  for the Qwen variant (note: Qwen2.5-**3B** is not Apache 2.0, unlike most other Qwen2.5 sizes).
- **Base models**: Llama-3.2-3B-Instruct / Qwen2.5-3B
- **Fine-tuning method**: LoRA (Low-Rank Adaptation) via [Unsloth](https://github.com/unslothai/unsloth)
- **Training data**: ABC CAD Dataset with GPT-4o generated captions

### Model Sources

- **Repository**: https://github.com/JasonShiii/STEP-LLM
- **Paper**: [DATE 2026](https://past.date-conference.com/proceedings-archive/2026/DATA/1319.pdf) · [arXiv:2601.12641](https://arxiv.org/abs/2601.12641)
- **LoRA adapters**: [step-llm-llama3b](https://huggingface.co/JasonShiii/step-llm-llama3b) · [step-llm-llama3b-no_rag](https://huggingface.co/JasonShiii/step-llm-llama3b-no_rag) · [step-llm-qwen3b](https://huggingface.co/JasonShiii/step-llm-qwen3b)

### Released Checkpoints

| Checkpoint | Mode | Base model | Training data | Steps |
|---|---|---|---|---|
| step-llm-llama3b        | RAG    | Llama-3.2-3B-Instruct | ~20k STEP files, 0–500 entities | 7200 |
| step-llm-llama3b-no_rag | no-RAG | Llama-3.2-3B-Instruct | ~20k STEP files, 0–500 entities | 6300 |
| step-llm-qwen3b         | RAG    | Qwen2.5-3B            | ~20k STEP files, 0–500 entities | 9000 |

The released checkpoints are LoRA adapters (~150 MB each), not full models.

> **RAG and no-RAG checkpoints are not interchangeable.** They were trained with
> different prompt templates. Running a RAG checkpoint without retrieval (or
> vice versa) puts the model off-distribution and degrades output quality.

## Uses

### Direct Use

Generate STEP files from natural language descriptions. Use the provided entry
point rather than calling the model directly — it applies the exact prompt
template the checkpoints were trained on and reattaches the STEP header:

```bash
# Without RAG — use the no-RAG adapter:
python generate_step.py \
    --ckpt_path ./checkpoints/step-llm-llama3b-no_rag \
    --caption   "A cylindrical bolt with a hexagonal head" \
    --save_dir  ./generated

# With RAG — use a RAG adapter:
python generate_step.py \
    --ckpt_path     ./checkpoints/step-llm-qwen3b \
    --use_rag \
    --db_csv_path   ./dataset/cad_captions_0-500.csv \
    --step_json_dir ./dataset/abc_rag/train_val_test \
    --caption       "A cylindrical bolt with a hexagonal head" \
    --save_dir      ./generated
```

The prompt templates live in `generate_step.py` and are byte-identical to those
in `llama3_SFT_response.py`. If you build your own inference loop, reuse them:

```python
from generate_step import ABC_PROMPT_RAG, ABC_PROMPT_NO_RAG, STEP_HEADER
```

### Downstream Use

- CAD model prototyping from text
- Design automation
- Text-to-3D workflows
- Engineering documentation to CAD conversion

### Out-of-Scope Use

- High-precision engineering applications (token-based representation limits precision)
- Safety-critical CAD design (requires human verification)
- Production use without validation
- Non-mechanical domains (trained on mechanical parts)

## Bias, Risks, and Limitations

### Known Limitations

1. **Geometric precision**: Token-based representation may introduce small numerical errors.
2. **Complexity**: Trained on models with 0–500 STEP entities; very complex assemblies may fail.
3. **Sequence length**: STEP files are long. Generation is capped at 16384 tokens of context; longer models get truncated.
4. **Domain**: Optimised for mechanical parts; architecture and organic shapes are out of distribution.
5. **Dimensions**: Generated dimensions may not match the exact specifications in the prompt.
6. **Validation**: Generated STEP files should be validated in CAD software before use.

### Bias

- Training data is biased toward mechanical parts and common geometric primitives.
- Simple shapes (cubes, cylinders, brackets) are over-represented.
- Complex assemblies and organic shapes are under-represented.
- May reflect biases in the GPT-4o caption generation process.

### Risks

- Generated models may not meet engineering specifications.
- Potential for invalid or non-manufacturable geometry.
- Generated geometry may resemble training data.

### Recommendations

- Validate all generated STEP files in CAD software.
- Verify dimensions and tolerances for engineering applications.
- Do not use for safety-critical or production applications without expert review.

## Training Details

### Training Data

- **Base dataset**: [ABC CAD Dataset](https://archive.nyu.edu/handle/2451/43778)
- **Captions**: generated with GPT-4o
- **Subset**: ~20k STEP files with 0–500 entities
- **Preprocessing**: floating-point rounding → DFS entity reorder (removes forward references) → RAG pairing → split
- **Split**: 70% train / 10% validation / 20% test (see `data_preparation/data_split.py`)

See [`docs/DATASET.md`](docs/DATASET.md) and [`data_preparation/README.md`](data_preparation/README.md) for details.

### Prompt Template

Training and inference use the same templates. See the "Prompt Template"
section of [`README.md`](README.md#prompt-template) for the exact strings.

### Training Hyperparameters

**LoRA configuration**:

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0 |
| `bias` | none |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

**Training configuration**:

| Parameter | Value |
|---|---|
| Optimiser | adamw_8bit |
| Learning rate | 5e-5 |
| Per-device batch size | 2 |
| Gradient accumulation | 4 (effective batch size 8) |
| Warmup steps | 1200 |
| `max_seq_length` | 16384 |
| Quantisation | none (`load_in_4bit=False`) |
| Hardware | single GPU |

Training steps per released checkpoint are listed in the checkpoint table above.

**Framework**: Unsloth 2025.3.18 · PyTorch 2.x · Transformers 4.35+

## Evaluation

### Metrics

- **Chamfer Distance** — geometric similarity between generated and reference models (`eval_ckpt/step_chamfer_reward.py`)
- **Complete Ratio (CR)** — STEP file structural validity (`eval_ckpt/CR/CR_calculate.py`)
- **Renderability** — whether the file loads and tessellates in OpenCASCADE (`eval_ckpt/renderability/check_renderability.py`)
- **Validation loss** by checkpoint (`eval_ckpt/eval_loss_by_ckpt.py`)

### Results

Quantitative results are reported in the DATE 2026 paper — see
[the proceedings PDF](https://past.date-conference.com/proceedings-archive/2026/DATA/1319.pdf)
or [arXiv:2601.12641](https://arxiv.org/abs/2601.12641).

See [`eval_ckpt/README_eval.md`](eval_ckpt/README_eval.md) for how to reproduce the evaluation.

## Technical Specifications

### Model Architecture

- **Base**: Llama-3.2-3B-Instruct or Qwen2.5-3B
- **Adaptation**: LoRA
- **Context length**: 16384 tokens (RoPE scaling handled by Unsloth)

### Compute Requirements

**Training**: single NVIDIA GPU, 24 GB+ VRAM recommended.

**Inference**: NVIDIA GPU required. At least 16 GB VRAM recommended — STEP files
can run to 16k tokens.

### Software

- Unsloth 2025.3.18
- PyTorch 2.0+
- Transformers 4.35+
- Python 3.10+
- CUDA 11.8+

## How to Get Started

```bash
# 1. Set up the environment
bash scripts/setup.sh
conda activate step_llm

# 2. Download the LoRA adapters (~150 MB each)
bash scripts/download_checkpoints.sh

# 3. Generate
python generate_step.py \
    --ckpt_path ./checkpoints/step-llm-llama3b-no_rag \
    --caption   "A cylindrical bolt with a hexagonal head"
```

### Merging a LoRA Adapter (optional)

Merging produces a standalone model that loads faster and needs no PEFT
dependency at inference time:

```bash
python scripts/merge_lora_adapter.py \
    --base_model_path Qwen/Qwen2.5-3B \
    --adapter_path    ./checkpoints/step-llm-qwen3b \
    --output_path     ./merged_model/step-llm-qwen3b-merged
```

## Citation

```bibtex
@inproceedings{shi2026stepllm,
  title={STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models},
  author={Shi, Xiangyu and Ding, Junyang and Zhao, Xu and Zhan, Sinong and Mohapatra, Payal and Quispe, Daniel and Welbeck, Kojo and Cao, Jian and Chen, Wei and Guo, Ping and Zhu, Qi},
  booktitle={Proceedings of the 2026 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  year={2026},
  organization={IEEE}
}
```

## Model Card Contact

Open an issue at https://github.com/JasonShiii/STEP-LLM/issues
