#!/usr/bin/env python3
"""
Merge a LoRA adapter with its base model to produce a standalone merged model.

The released STEP-LLM checkpoints are LoRA adapters trained on top of:
  - Llama-3.2-3B-Instruct  (checkpoint-7200)
  - Qwen2.5-3B-Instruct    (checkpoint-9000)

Merging is optional — you can also load the adapter directly at inference time
via generate_step.py by pointing --ckpt_path at the adapter directory.

Usage:
    python scripts/merge_lora_adapter.py \\
        --base_model_path meta-llama/Llama-3.2-3B-Instruct \\
        --adapter_path    ./checkpoints/step-llm-llama3b \\
        --output_path     ./merged_model/step-llm-llama3b-merged

    python scripts/merge_lora_adapter.py \\
        --base_model_path Qwen/Qwen2.5-3B-Instruct \\
        --adapter_path    ./checkpoints/step-llm-qwen3b \\
        --output_path     ./merged_model/step-llm-qwen3b-merged
"""

import argparse
import os

try:
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import torch
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install unsloth peft torch transformers")
    exit(1)


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    max_seq_length: int = 16384,
):
    """
    Load a base model, apply a LoRA adapter, merge weights, and save.

    Args:
        base_model_path: HuggingFace model ID or local path to base model.
        adapter_path:    Path to the LoRA adapter directory.
        output_path:     Where to save the merged full-precision model.
        max_seq_length:  Must match the value used during training (default 16384).
    """
    print("=" * 60)
    print("LoRA Adapter Merge")
    print("=" * 60)

    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter not found: {adapter_path}")

    os.makedirs(output_path, exist_ok=True)

    print(f"  Base model  : {base_model_path}")
    print(f"  LoRA adapter: {adapter_path}")
    print(f"  Output path : {output_path}")
    print(f"  Max seq len : {max_seq_length}")
    print()

    # Load base model via Unsloth (enables RoPE scaling, etc.)
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=None,        # auto-detect (bfloat16 on Ampere+)
        load_in_4bit=False,
    )

    # Load LoRA adapter weights on top of the base model
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Fuse adapter weights into the base model (no runtime overhead at inference)
    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    # Save merged model + tokenizer
    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print()
    print("=" * 60)
    print("Merge complete!")
    print("=" * 60)
    print()
    print("You can now run inference with the merged model:")
    print(f"  python generate_step.py --ckpt_path {output_path} --caption 'A bolt'")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter with its base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base_model_path",
        required=True,
        help="HuggingFace model ID or local path to the base model "
             "(e.g. 'meta-llama/Llama-3.2-3B-Instruct' or './Qwen2.5-3B-Instruct').",
    )
    parser.add_argument(
        "--adapter_path",
        required=True,
        help="Path to the LoRA adapter directory (e.g. './checkpoints/step-llm-llama3b').",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory where the merged model will be saved.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=16384,
        help="Maximum sequence length — must match training value (default: 16384).",
    )

    args = parser.parse_args()
    merge_lora_adapter(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
