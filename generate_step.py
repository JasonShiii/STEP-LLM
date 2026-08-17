#!/usr/bin/env python3
"""
STEP file generation from natural language, with or without RAG.

This is the inference entry point for STEP-LLM. The prompt templates below
must stay byte-identical to the ones in llama3_SFT_response.py — the
checkpoints were trained on those exact strings.

See README.md ("Run Inference") for usage, or run with --help.
"""

import os
import json
import argparse
from unsloth import FastLanguageModel
from transformers import TextStreamer


def load_model(ckpt_path, max_seq_length=16384, dtype=None, load_in_4bit=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ckpt_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    print(f"Model {id(model)} is set for inference.")
    return model, tokenizer


def retrieve_step_text(caption, db_csv_path, step_json_dir, top_k=5):
    # Imported lazily so that no-RAG generation does not require the retrieval stack.
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    df = pd.read_csv(db_csv_path)
    df = df[df["isDescribable"] == True]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    descriptions = df["description"].tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=False)

    # Use sklearn cosine similarity (no FAISS dependency required)
    caption_embedding = model.encode([caption], convert_to_tensor=False)
    similarities = cosine_similarity(caption_embedding, embeddings)[0]

    # Get top_k most similar indices (descending)
    top_indices = similarities.argsort()[-top_k:][::-1]

    for rank, idx in enumerate(top_indices):
        model_id = str(df.iloc[idx]["model_id"]).zfill(8)
        description = df.iloc[idx]["description"]
        similarity_score = similarities[idx]
        print(
            f"Trying top-{rank+1} match: {description} "
            f"(model_id: {model_id}, similarity: {similarity_score:.4f})"
        )

        for fname in ["train.json", "test.json", "val.json"]:
            path = os.path.join(step_json_dir, fname)
            with open(path, "r") as f:
                data = json.load(f)
                for item in data:
                    if str(item["id_original"]).zfill(8) == model_id:
                        print(f"Found STEP file in {fname} for model_id {model_id}")
                        return item["output"], model_id, description

        print(f"Top-{rank+1} retrieved model_id {model_id} not found in JSON directory.")

    raise FileNotFoundError(
        f"No STEP file found for any of the top-{top_k} matches."
    )


# Prompt templates. These MUST stay byte-identical to the ones in
# llama3_SFT_response.py — the checkpoints were trained on those exact strings,
# and even a one-word difference degrades generation quality.
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

# Standard STEP header prepended to the generated DATA section. The model is
# trained to emit only the DATA section, so the header is added back here.
STEP_HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION( ( '' ), ' ' );
FILE_NAME( '/vol/tmp/translate-8579754438183730235/5ae5839f3947920fcf80d878.step', '2018-04-29T08:34:40', ( '' ), ( '' ), ' ', ' ', ' ' );
FILE_SCHEMA( ( 'AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }' ) );
ENDSEC;"""


def generate_step_file(
    ckpt_path,
    db_csv_path,
    step_dir,
    use_rag,
    caption,
    save_dir,
    output_name="output.step",
    max_new_tokens=14000,
):
    model, tokenizer = load_model(ckpt_path)

    if use_rag:
        rel_step_text, model_id, retrieved_caption = retrieve_step_text(
            caption, db_csv_path, step_dir
        )
        formatted_prompt = ABC_PROMPT_RAG.format(caption, rel_step_text, "")
    else:
        formatted_prompt = ABC_PROMPT_NO_RAG.format(caption, "")

    # Tokenize and generate
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    streamer = TextStreamer(tokenizer)
    print(f"Generating STEP file with model object id: {id(model)}...")
    generated = model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    # Extract the STEP DATA section (after '### output:')
    step_data = output_text.split("### output:")[-1].strip()

    # Prepend standard STEP header
    full_step_file = STEP_HEADER + "\n" + step_data

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, output_name)
    with open(save_path, "w") as f:
        f.write(full_step_file)

    print(f"STEP file saved to {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a STEP CAD file from a natural language description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Without RAG (simpler, no retrieval database needed):
  python generate_step.py \\
      --ckpt_path ./checkpoints/step-llm-qwen3b \\
      --caption "A cylindrical bolt with a hexagonal head" \\
      --save_dir ./generated

  # With RAG (retrieves a similar example from the training set):
  python generate_step.py \\
      --ckpt_path ./checkpoints/step-llm-qwen3b \\
      --use_rag \\
      --db_csv_path ./dataset/cad_captions_0-500.csv \\
      --step_json_dir ./dataset/abc_rag/train_val_test \\
      --caption "A cylindrical bolt with a hexagonal head" \\
      --save_dir ./generated \\
      --output_name bolt.step
""",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the model checkpoint or LoRA adapter directory.",
    )
    parser.add_argument(
        "--caption",
        required=True,
        help="Natural language description of the CAD model to generate.",
    )
    parser.add_argument(
        "--save_dir",
        default="./generated",
        help="Directory to save the generated STEP file (default: ./generated).",
    )
    parser.add_argument(
        "--output_name",
        default="output.step",
        help="Filename for the generated STEP file (default: output.step).",
    )
    parser.add_argument(
        "--use_rag",
        action="store_true",
        help="Enable Retrieval-Augmented Generation (requires --db_csv_path and --step_json_dir).",
    )
    parser.add_argument(
        "--db_csv_path",
        default=None,
        help="Path to captions CSV file used for RAG retrieval (required when --use_rag).",
    )
    parser.add_argument(
        "--step_json_dir",
        default=None,
        help="Directory containing train/val/test JSON files for RAG retrieval "
             "(required when --use_rag).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=14000,
        help="Maximum number of tokens to generate (default: 14000).",
    )

    args = parser.parse_args()

    if args.use_rag:
        if not args.db_csv_path or not args.step_json_dir:
            parser.error("--use_rag requires both --db_csv_path and --step_json_dir")

    generate_step_file(
        ckpt_path=args.ckpt_path,
        db_csv_path=args.db_csv_path,
        step_dir=args.step_json_dir,
        use_rag=args.use_rag,
        caption=args.caption,
        save_dir=args.save_dir,
        output_name=args.output_name,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
