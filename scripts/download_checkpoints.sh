#!/bin/bash
# Download STEP-LLM LoRA adapter checkpoints
#
# Three checkpoints are released:
#   - step-llm-llama3b         : Llama-3.2-3B-Instruct, RAG    (checkpoint-7200)
#   - step-llm-llama3b-no_rag  : Llama-3.2-3B-Instruct, no-RAG (checkpoint-6300)
#   - step-llm-qwen3b          : Qwen2.5-3B,             RAG    (checkpoint-9000)
#
# RAG and no-RAG variants were trained with different prompt templates and are
# NOT interchangeable — pick the variant that matches how you'll run inference.
# The Qwen no-RAG variant is not released; use the Llama no-RAG adapter for
# RAG-free generation.
#
# After downloading, you can either:
#   (a) Use the adapter directly — point --ckpt_path at the adapter dir
#   (b) Merge into a full model  — run scripts/merge_lora_adapter.py
#
# Usage:
#   bash scripts/download_checkpoints.sh                   # downloads all three
#   bash scripts/download_checkpoints.sh llama             # Llama RAG only
#   bash scripts/download_checkpoints.sh llama-no-rag      # Llama no-RAG only
#   bash scripts/download_checkpoints.sh qwen              # Qwen RAG only

set -e

MODEL=${1:-"all"}   # "llama", "llama-no-rag", "qwen", or "all"
DEST_DIR="./checkpoints"

mkdir -p "$DEST_DIR"

# ── Option A: HuggingFace Hub ─────────────────────────────────────────────────
# pip install huggingface_hub
#
# Replace JasonShiii with your actual HuggingFace username after uploading.

download_hf() {
    local repo_id="$1"
    local local_dir="$2"
    echo "Downloading $repo_id from HuggingFace..."
    huggingface-cli download "$repo_id" --local-dir "$local_dir"
    echo "Saved to $local_dir"
}

# ── Option B: GitHub Releases ─────────────────────────────────────────────────
# Replace JasonShiii/STEP-LLM with your actual repo path after release.

download_gh() {
    local url="$1"
    local dest="$2"
    echo "Downloading from GitHub Releases: $url"
    wget -q --show-progress -O "${dest}.zip" "$url"
    unzip -q "${dest}.zip" -d "$dest"
    rm "${dest}.zip"
    echo "Saved to $dest"
}

# ─────────────────────────────────────────────────────────────────────────────

if [[ "$MODEL" == "llama" || "$MODEL" == "all" || "$MODEL" == "both" ]]; then
    download_hf "JasonShiii/step-llm-llama3b" "$DEST_DIR/step-llm-llama3b"
fi

if [[ "$MODEL" == "llama-no-rag" || "$MODEL" == "all" || "$MODEL" == "both" ]]; then
    download_hf "JasonShiii/step-llm-llama3b-no_rag" "$DEST_DIR/step-llm-llama3b-no_rag"
fi

if [[ "$MODEL" == "qwen" || "$MODEL" == "all" || "$MODEL" == "both" ]]; then
    download_hf "JasonShiii/step-llm-qwen3b" "$DEST_DIR/step-llm-qwen3b"
fi

echo ""
echo "Checkpoints are in $DEST_DIR/"
echo ""
echo "Next steps:"
echo "  # Run inference directly with the adapter:"
echo "  python generate_step.py --ckpt_path $DEST_DIR/step-llm-qwen3b --caption 'A bolt'"
echo ""
echo "  # Or merge with the base model first:"
echo "  python scripts/merge_lora_adapter.py \\"
echo "      --base_model_path Qwen/Qwen2.5-3B \\"
echo "      --adapter_path    $DEST_DIR/step-llm-qwen3b \\"
echo "      --output_path     ./merged_model/step-llm-qwen3b-merged"
