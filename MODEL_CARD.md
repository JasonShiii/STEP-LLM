# Model Card: CAD Text-to-STEP Generation

## Model Details

### Model Description

A fine-tuned language model for generating CAD models in STEP format from natural language descriptions.

- **Developed by**: [Your Name/Organization]
- **Model type**: Autoregressive Language Model with LoRA fine-tuning
- **Language**: English (input), STEP (output)
- **License**: MIT (code), [Specify for model weights]
- **Base Model**: Qwen2.5-3B-Instruct / Llama-3.2-3B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: ABC CAD Dataset with generated captions

### Model Sources

- **Repository**: [GitHub URL]
- **Paper**: [ArXiv/Paper link if available]
- **Demo**: [Demo link if available]
- **LoRA Adapter**: [HuggingFace/Release URL]

## Uses

### Direct Use

Generate STEP files (CAD models) from natural language descriptions:

```python
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/merged_model",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True
)

# Generate
prompt = "A rectangular bracket with four mounting holes"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048)
step_file = tokenizer.decode(outputs[0])
```

### Downstream Use

- CAD model prototyping from text
- Design automation
- Text-to-3D workflows
- Engineering documentation to CAD conversion

### Out-of-Scope Use

- High-precision engineering applications (token-based representation limits precision)
- Safety-critical CAD design (requires human verification)
- Production-ready models without validation
- Non-mechanical domains (trained on mechanical parts)

## Bias, Risks, and Limitations

### Known Limitations

1. **Geometric Precision**: Token-based representation may introduce small numerical errors
2. **Complexity**: Best for models with < 2000 tokens; very complex assemblies may fail
3. **Domain**: Optimized for mechanical parts; other domains (architecture, organic shapes) may be suboptimal
4. **Dimensions**: Generated dimensions may not match exact specifications in prompt
5. **Validation**: Generated STEP files should be validated before use

### Bias

- Training data biased toward mechanical parts and common geometric primitives
- Over-representation of simple shapes (cubes, cylinders, brackets)
- Limited representation of complex assemblies and organic shapes
- May reflect biases in caption generation process

### Risks

- Generated models may not meet engineering specifications
- Potential for generating invalid or non-manufacturable geometries
- Should not be used in safety-critical applications without verification
- Generated IP may inadvertently resemble training data

### Recommendations

Users should:
- Validate all generated STEP files in CAD software
- Verify dimensions and tolerances for engineering applications
- Check for geometric validity and manufacturability
- Not use for safety-critical or production applications without expert review

## Training Details

### Training Data

- **Base Dataset**: ABC CAD Dataset (~1M CAD models)
- **Captions**: GPT-4 Vision generated descriptions
- **Samples**: ~XXX,XXX training examples (update with actual number)
- **Token Length**: Filtered to < 2000 tokens
- **Split**: 70% train, 20% validation, 10% test

See `docs/DATASET.md` for details.

### Training Procedure

#### Preprocessing

1. STEP file reordering for consistency
2. Entity renumbering
3. Caption generation using vision models
4. Token length filtering
5. Dataset splitting

#### Training Hyperparameters

**LoRA Configuration**:
- Rank (r): 16
- Alpha: 16
- Dropout: 0
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Trainable parameters: ~1-2% of base model

**Training Configuration**:
- Optimizer: AdamW
- Learning rate: 2e-4
- Batch size: [Update with actual value]
- Gradient accumulation: [Update]
- Max sequence length: 4096
- Training epochs: [Update]
- Hardware: [Update: e.g., 4x A100 40GB]

**Framework**:
- Unsloth: 2025.3.18
- PyTorch: 2.x
- Transformers: 4.35+

#### Speeds, Sizes, Times

**Training Time**: ~XX hours on [hardware]
**Model Size**:
- LoRA adapter: ~500 MB
- Base model: ~6 GB
- Merged model: ~6 GB

**Inference Speed**:
- With 4-bit quantization: ~XX tokens/sec
- Without quantization: ~XX tokens/sec

## Evaluation

### Testing Data & Metrics

#### Test Data

- Test split: 10% of dataset (~X,XXX examples)
- Held-out, unseen during training
- Same distribution as training data

#### Metrics

1. **Generation Quality**
   - STEP file validity: XX%
   - Geometric validity (OpenCASCADE): XX%
   - Parse success rate: XX%

2. **Geometric Accuracy**
   - Chamfer Distance: X.XX ± X.XX
   - Hausdorff Distance: X.XX ± X.XX
   - Point cloud IoU: X.XX

3. **Language Understanding**
   - Caption relevance: [Qualitative/Quantitative metric]
   - Dimension accuracy: [Metric]

4. **Training Metrics**
   - Training loss: X.XX
   - Validation loss: X.XX
   - Perplexity: XX.X

### Results

[Update with actual results]

**Qualitative Examples**:
- Successfully generates simple geometric primitives
- Handles dimensional specifications reasonably well
- Can produce multi-feature models (holes, grooves, etc.)
- Struggles with very complex assemblies

See `eval_ckpt/README_eval.md` for detailed evaluation results.

## Environmental Impact

**Hardware**: [e.g., 4x NVIDIA A100 40GB]
**Training Time**: [e.g., 24 hours]
**Cloud Provider**: [If applicable]
**Carbon Footprint**: [Estimate if available]

Tools for estimation:
- [ML CO2 Impact Calculator](https://mlco2.github.io/impact/)
- [Code Carbon](https://codecarbon.io/)

## Technical Specifications

### Model Architecture

- **Base**: Qwen2.5-3B-Instruct or Llama-3.2-3B
- **Adaptation**: LoRA (Low-Rank Adaptation)
- **Context length**: 4096 tokens
- **Vocabulary size**: [Base model vocabulary]

### Compute Infrastructure

**Training**:
- GPU: [e.g., 4x A100 40GB]
- CPU: [e.g., 64 cores]
- RAM: [e.g., 256 GB]
- Storage: [e.g., 2 TB SSD]

**Inference**:
- Minimum: 1x GPU with 8GB VRAM (with 4-bit quantization)
- Recommended: 1x GPU with 16GB+ VRAM

### Software

- Unsloth: 2025.3.18
- PyTorch: 2.0+
- Transformers: 4.35+
- Python: 3.10+
- CUDA: 11.8+

## Model Card Authors

[Your Name]

## Model Card Contact

[Your Email/GitHub]

## Citation

```bibtex
@misc{cad_text_to_step_2025,
  title={Text-to-CAD: Generating STEP Files from Natural Language},
  author={Your Name},
  year={2025},
  url={your-github-url}
}
```

## How to Get Started

### Download and Merge LoRA Adapter

```bash
# Download base model
python scripts/download_base_models.sh

# Download LoRA adapter from releases
# Link: [your-release-url]

# Merge adapter with base model
python scripts/merge_lora_adapter.py \
    --base_model_path ./Qwen2.5-3B-Instruct \
    --adapter_path ./lora_adapter \
    --output_path ./merged_model
```

### Basic Inference

```bash
python examples/basic_inference.py
```

### With RAG

```bash
python examples/rag_inference.py
```

## More Information

- **Repository**: [GitHub URL]
- **Documentation**: See `README.md` and `docs/`
- **Dataset**: See `docs/DATASET.md`
- **Training**: See `docs/TRAINING.md`
- **Paper**: [Link if available]
