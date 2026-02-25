#!/usr/bin/env python3
"""
Basic STEP file generation example (no RAG).

Demonstrates simple text-to-CAD generation from a trained checkpoint.
For RAG-augmented generation see examples/rag_inference.py.

Usage:
    # Using a LoRA adapter directly:
    python examples/basic_inference.py \
        --ckpt_path ./checkpoints/step-llm-qwen3b \
        --caption "A cylindrical bolt with a hexagonal head"

    # Using a merged full model:
    python examples/basic_inference.py \
        --ckpt_path ./merged_model/step-llm-qwen3b-merged \
        --caption "A simple cube"
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path so we can import generate_step helpers
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from unsloth import FastLanguageModel
    from transformers import TextStreamer
    import torch
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install unsloth torch transformers")
    sys.exit(1)


# ── Prompt template ────────────────────────────────────────────────────────────
# This MUST match the format used during training (see llama3_SFT_response.py).
PROMPT_TEMPLATE = """You are a CAD model generation assistant trained to produce STEP (.step) files based on textual descriptions. Given the following object description and relevant retrieved CAD data, generate a STEP file that accurately represents the described object.


### caption:
{}

### output:
{}"""

STEP_HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION( ( '' ), ' ' );
FILE_NAME( '/vol/tmp/translate-8579754438183730235/5ae5839f3947920fcf80d878.step', '2018-04-29T08:34:40', ( '' ), ( '' ), ' ', ' ', ' ' );
FILE_SCHEMA( ( 'AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }' ) );
ENDSEC;"""


def generate_step(caption: str, ckpt_path: str, max_new_tokens: int = 14000) -> str:
    """
    Generate a STEP file from a text description (no RAG).

    Args:
        caption:        Natural language description of the CAD model.
        ckpt_path:      Path to the checkpoint or LoRA adapter directory.
        max_new_tokens: Maximum tokens to generate (default: 14000).

    Returns:
        Full STEP file content as a string.
    """
    print(f"Loading model from: {ckpt_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ckpt_path,
        max_seq_length=16384,   # must match training value
        dtype=None,             # auto-detect (bfloat16 on Ampere+)
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    formatted_prompt = PROMPT_TEMPLATE.format(caption, "")
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer)

    print(f"\nGenerating STEP for: '{caption}'")
    print("-" * 50)

    generated = model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    # Extract the model's output (everything after '### output:')
    step_data = output_text.split("### output:")[-1].strip()
    return STEP_HEADER + "\n" + step_data


def main():
    parser = argparse.ArgumentParser(description="Basic (no-RAG) STEP file generation.")
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the checkpoint or LoRA adapter directory.",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Natural language description of the CAD model. "
             "If omitted, an interactive prompt will be shown.",
    )
    parser.add_argument(
        "--save_dir",
        default="./generated_outputs",
        help="Directory to save the result (default: ./generated_outputs).",
    )
    parser.add_argument(
        "--output_name",
        default="output.step",
        help="Filename for the generated STEP file (default: output.step).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=14000,
        help="Maximum tokens to generate (default: 14000).",
    )
    args = parser.parse_args()

    # Prompt interactively if no caption given
    caption = args.caption
    if not caption:
        print("Example descriptions:")
        examples = [
            "A simple cube with dimensions 10x10x10 mm",
            "A cylinder with radius 5mm and height 20mm",
            "A rectangular bracket with mounting holes",
        ]
        for i, ex in enumerate(examples, 1):
            print(f"  {i}) {ex}")
        print(f"  {len(examples)+1}) Enter custom description")
        choice = input(f"\nChoice [1-{len(examples)+1}]: ").strip()
        try:
            n = int(choice)
            caption = examples[n - 1] if 1 <= n <= len(examples) else input("Description: ").strip()
        except (ValueError, IndexError):
            caption = input("Description: ").strip()

    if not caption:
        print("Error: empty caption.")
        sys.exit(1)

    step_content = generate_step(
        caption=caption,
        ckpt_path=args.ckpt_path,
        max_new_tokens=args.max_new_tokens,
    )

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    out_path.write_text(step_content)

    print("\n" + "=" * 50)
    print("Generation complete!")
    print("=" * 50)
    print(f"\nSaved to: {out_path}")
    print("\nPreview (first 500 chars):")
    print("-" * 50)
    print(step_content[:500] + ("..." if len(step_content) > 500 else ""))


if __name__ == "__main__":
    main()
