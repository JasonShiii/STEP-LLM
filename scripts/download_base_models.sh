#!/bin/bash
# Download base models (Llama and Qwen)

set -e  # Exit on error

echo "=========================================="
echo "Base Model Download Script"
echo "=========================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}}"

# Function to download from Hugging Face
download_hf_model() {
    local model_name=$1
    local local_dir=$2
    
    echo "Downloading ${model_name}..."
    
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download ${model_name} --local-dir ${local_dir}
    else
        echo "Installing huggingface-cli..."
        pip install -q huggingface_hub[cli]
        huggingface-cli download ${model_name} --local-dir ${local_dir}
    fi
}

echo "Select models to download:"
echo "1) Llama 3.2 3B"
echo "2) Qwen2.5 3B Instruct"
echo "3) Both"
echo "4) Skip (models already downloaded)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "Downloading Llama 3.2 3B..."
        download_hf_model "meta-llama/Llama-3.2-3B" "${MODEL_DIR}/llama_3.2/Llama_3.2_3B"
        echo "✓ Llama 3.2 3B downloaded to ${MODEL_DIR}/llama_3.2/Llama_3.2_3B"
        ;;
    2)
        echo "Downloading Qwen2.5 3B Instruct..."
        download_hf_model "Qwen/Qwen2.5-3B-Instruct" "${MODEL_DIR}/Qwen2.5-3B-Instruct"
        echo "✓ Qwen2.5 3B downloaded to ${MODEL_DIR}/Qwen2.5-3B-Instruct"
        ;;
    3)
        echo "Downloading both models..."
        download_hf_model "meta-llama/Llama-3.2-3B" "${MODEL_DIR}/llama_3.2/Llama_3.2_3B"
        download_hf_model "Qwen/Qwen2.5-3B-Instruct" "${MODEL_DIR}/Qwen2.5-3B-Instruct"
        echo "✓ Both models downloaded"
        ;;
    4)
        echo "Skipping model download"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Note: You may need a Hugging Face token for some models"
echo "Set it with: huggingface-cli login"
echo "=========================================="
