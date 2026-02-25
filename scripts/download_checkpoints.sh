#!/bin/bash
# Download STEP-LLM LoRA adapter checkpoints
#
# Two checkpoints are released:
#   - step-llm-llama3b : trained on Llama-3.2-3B-Instruct  (checkpoint-7200)
#   - step-llm-qwen3b  : trained on Qwen2.5-3B-Instruct    (checkpoint-9000)
#
# After downloading, you can either:
#   (a) Use the adapter directly — point --ckpt_path at the adapter dir
#   (b) Merge into a full model  — run scripts/merge_lora_adapter.py
#
# Usage:
#   bash scripts/download_checkpoints.sh            # downloads both
#   bash scripts/download_checkpoints.sh llama      # Llama only
#   bash scripts/download_checkpoints.sh qwen       # Qwen only

set -e

MODEL=${1:-"both"}   # "llama", "qwen", or "both"
DEST_DIR="./checkpoints"

mkdir -p "$DEST_DIR"

# ── Option A: HuggingFace Hub ─────────────────────────────────────────────────
# pip install huggingface_hub
#
# Replace YOUR_HF_USERNAME with your actual HuggingFace username after uploading.

download_hf() {
    local repo_id="$1"
    local local_dir="$2"
    echo "Downloading $repo_id from HuggingFace..."
    huggingface-cli download "$repo_id" --local-dir "$local_dir"
    echo "Saved to $local_dir"
}

# ── Option B: GitHub Releases ─────────────────────────────────────────────────
# Replace YOUR_GITHUB_USER/step-llm with your actual repo path after release.

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

if [[ "$MODEL" == "llama" || "$MODEL" == "both" ]]; then
    # --- HuggingFace (uncomment and fill in after uploading) ---
    # download_hf "YOUR_HF_USERNAME/step-llm-llama3b" "$DEST_DIR/step-llm-llama3b"

    # --- GitHub Releases (uncomment and fill in after creating release) ---
    # download_gh \
    #   "https://github.com/YOUR_GITHUB_USER/step-llm/releases/download/v1.0/step-llm-llama3b.zip" \
    #   "$DEST_DIR/step-llm-llama3b"

    echo "[Llama] TODO: fill in HuggingFace or GitHub release URL above."
fi

if [[ "$MODEL" == "qwen" || "$MODEL" == "both" ]]; then
    # --- HuggingFace (uncomment and fill in after uploading) ---
    # download_hf "YOUR_HF_USERNAME/step-llm-qwen3b" "$DEST_DIR/step-llm-qwen3b"

    # --- GitHub Releases (uncomment and fill in after creating release) ---
    # download_gh \
    #   "https://github.com/YOUR_GITHUB_USER/step-llm/releases/download/v1.0/step-llm-qwen3b.zip" \
    #   "$DEST_DIR/step-llm-qwen3b"

    echo "[Qwen] TODO: fill in HuggingFace or GitHub release URL above."
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
echo "      --base_model_path Qwen/Qwen2.5-3B-Instruct \\"
echo "      --adapter_path    $DEST_DIR/step-llm-qwen3b \\"
echo "      --output_path     ./merged_model/step-llm-qwen3b-merged"
