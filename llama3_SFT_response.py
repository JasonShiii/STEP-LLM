"""
STEP-LLM Training Script
========================
Fine-tunes a Llama-3.2 or Qwen-2.5 model on the STEP-CAD dataset using
Unsloth's efficient LoRA implementation.

This script is based on the Unsloth SFT template:
  https://github.com/unslothai/unsloth

For a notebook version see: llama3_SFT_response.ipynb

Quick start
-----------
1. Set the configuration variables in the "── Configuration ──" section below.
2. Run: python llama3_SFT_response.py
3. The LoRA adapter will be saved to LORA_SAVE_PATH.

The adapter can then be used directly in generate_step.py, or merged into a
full model with: python scripts/merge_lora_adapter.py
"""

# ── Configuration ──────────────────────────────────────────────────────────────
# UPDATE these paths before running.

# Base model — one of:
#   "meta-llama/Llama-3.2-3B-Instruct"   (requires HF access token)
#   "Qwen/Qwen2.5-3B-Instruct"
#   "/local/path/to/downloaded/model"
BASE_MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"  # UPDATE THIS PATH

# RAG dataset directory (produced by dataset_construct_rag.py)
# Must contain train.json (and optionally test.json / val.json in HF arrow format)
DATASET_PATH = "./data/abc_rag/20500_dfs"  # UPDATE THIS PATH

# Where to save the LoRA adapter after training
LORA_SAVE_PATH = "./lora_adapter"  # UPDATE THIS PATH

# Where to save intermediate checkpoints during training
OUTPUT_DIR = "./ckpt_outputs"  # UPDATE THIS PATH

# Whether to include retrieved STEP examples in the prompt (recommended: True)
USE_RAG = True
# ── End configuration ──────────────────────────────────────────────────────────


from unsloth import FastLanguageModel
import torch

max_seq_length = 16384  # RoPE Scaling is handled automatically by Unsloth
dtype = None            # None = auto-detect (bfloat16 on Ampere+, float16 on older GPUs)
load_in_4bit = False    # Set True to use 4-bit quantisation (reduces VRAM, slight quality loss)

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token="hf_...",   # needed for gated models (e.g. Llama)
)

# Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                               # LoRA rank — 8/16/32/64 are common choices
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,                     # 0 is optimised in Unsloth
    bias="none",                        # "none" is optimised in Unsloth
    use_gradient_checkpointing="unsloth",  # saves 30% VRAM; also fits 2× larger batches
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ── Prompt templates ────────────────────────────────────────────────────────────
# These templates MUST be used consistently at both training and inference time.

ABC_PROMPT_RAG = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description and relevant retrieved CAD data, generate a STEP file that accurately represents the described object.


### caption:
{}

### retrieved relevant step file:
{}

### output:
{}"""

ABC_PROMPT_NO_RAG = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description, generate a STEP file that accurately represents the described object.

### caption:
{}

### output:
{}"""

EOS_TOKEN = tokenizer.eos_token  # must be appended to every training example


def formatting_prompts_func(examples):
    """Format dataset examples into the training prompt."""
    instructions = examples["caption"]
    outputs = examples["output"]
    texts = []

    if USE_RAG:
        inputs = examples["relavant_step_file"]
        for instruction, input_, output in zip(instructions, inputs, outputs):
            text = ABC_PROMPT_RAG.format(instruction, input_, output) + EOS_TOKEN
            texts.append(text)
    else:
        for instruction, output in zip(instructions, outputs):
            text = ABC_PROMPT_NO_RAG.format(instruction, output) + EOS_TOKEN
            texts.append(text)

    return {"text": texts}


# ── Dataset ─────────────────────────────────────────────────────────────────────
from datasets import load_dataset

dataset = load_dataset(path=DATASET_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

test_dataset = load_dataset(path=DATASET_PATH, split="test")
test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

# ── Training ─────────────────────────────────────────────────────────────────────
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=1200,
        num_train_epochs=40,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",       # set to "wandb" for experiment tracking
        save_strategy="steps",
        save_steps=300,
    ),
)

# Train only on model outputs (mask the prompt so loss is only on the STEP data)
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="### caption:\n",
    response_part="### output:\n",
)

print("\nTraining started...\n")
trainer_stats = trainer.train()
print("\nTraining completed!")
print(trainer_stats)

# Save LoRA adapter
print(f"\nSaving LoRA adapter to {LORA_SAVE_PATH} ...")
model.save_pretrained(LORA_SAVE_PATH)
tokenizer.save_pretrained(LORA_SAVE_PATH)
print("Done!")
