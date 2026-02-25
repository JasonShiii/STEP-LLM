import unsloth
import os
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
import pandas as pd
import logging
import sys

max_seq_length = 16384 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

checkpoint_path = "/home/group/cad_codebased/data/SFT_ckpt/ckpt_outputs_improved_caption/ckpt_colab/checkpoint-9000" # 
log_csv = "/home/group/cad_codebased/training_log/eval_loss_new_caption.csv"

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "/home/group/cad_codebased/llama_3.2/Llama_3.2_3B", # or choose "unsloth/Llama-3.2-1B-Instruct"
    model_name = checkpoint_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# Data Preparation
use_rag = True  # Set to False if not using RAG

ABC_prompt_rag = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description and relevant retrieved CAD data, generate a STEP file that accurately represents the described object.


### caption:
{}

### retrieved relevant step file:
{}

### output:
{}"""

ABC_prompt_no_rag = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description, generate a STEP file that accurately represents the described object.

### caption:
{}

### output:
{}"""



EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

dataset = load_dataset(path="/home/group/cad_codebased/data/abc_rag/20500", split="train")
test_dataset = load_dataset(path="/home/group/cad_codebased/data/abc_rag/20500", split="test")

# Process dataset without calling a separate function
if use_rag:
    dataset = dataset.map(
        lambda examples: {
            "text": [
                ABC_prompt_rag.format(inst, inp, out) + EOS_TOKEN
                for inst, inp, out in zip(examples["caption"], examples["relavant_step_file"], examples["output"])
            ]
        },
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda examples: {
            "text": [
                ABC_prompt_rag.format(inst, inp, out) + EOS_TOKEN
                    for inst, inp, out in zip(examples["caption"], examples["relavant_step_file"], examples["output"])
            ]
        },
        batched=True,
    )
else:
    dataset = dataset.map(
        lambda examples: {
             "text": [
                ABC_prompt_no_rag.format(inst, out) + EOS_TOKEN
                for inst, out in zip(examples["caption"], examples["output"])
            ]
        },
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda examples: {
            "text": [
                ABC_prompt_no_rag.format(inst, out) + EOS_TOKEN
                for inst, out in zip(examples["caption"], examples["output"])
            ]
        },
        batched=True,
    )

# Train the model
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        # warmup_steps = 40,
        # warmup_steps = 3,
        # num_train_epochs = 1, # Set this for 1 full training run.
        # num_train_epochs = 5,
        max_steps = 10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "ckpt_outputs_response",
        report_to = "none", # Use this for WandB etc
        eval_strategy="steps",  # Run test loss calculation during training
        eval_steps=5,  # Calculate test loss every 5 steps
    ),
)

# only train on the outputs and ignore the loss on the user's inputs.
# from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "### caption:\n",
    response_part = "### output:\n",
)
# print("trainer.train_dataset",trainer.train_dataset)


eval_results = trainer.evaluate()
print("eval_results",eval_results)
eval_loss = eval_results.get("eval_loss", None)

# Append eval_loss and checkpoint path to CSV file
log_data = pd.DataFrame([{"eval_loss": eval_loss,"checkpoint": checkpoint_path}])

if os.path.exists(log_csv):
    log_data.to_csv(log_csv, mode="a", header=False, index=False)  # Append to existing file
else:
    log_data.to_csv(log_csv, index=False)  # Create new file if it doesn't exist

print(f"Evaluation results saved to {log_csv}")