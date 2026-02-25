#!/bin/bash
# Process ABC Dataset with generated captions
# This script applies the generated captions to the ABC dataset

set -e  # Exit on error

echo "=========================================="
echo "ABC Dataset Processing Script"
echo "=========================================="
echo ""

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
ABC_DIR="${DATA_DIR}/abccad"
CAPTION_FILE="${CAPTION_FILE:-${DATA_DIR}/cad_captions.csv}"

# Check prerequisites
if [ ! -d "${ABC_DIR}" ]; then
    echo "Error: ABC dataset not found at ${ABC_DIR}"
    echo "Please run: bash scripts/download_abc_dataset.sh"
    exit 1
fi

if [ ! -f "${CAPTION_FILE}" ]; then
    echo "Error: Caption file not found at ${CAPTION_FILE}"
    echo "Please ensure you have downloaded the captions from our release"
    exit 1
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Data directory: ${DATA_DIR}"
echo "ABC directory: ${ABC_DIR}"
echo "Caption file: ${CAPTION_FILE}"
echo ""

# Step 1: Reorder STEP files
echo "Step 1/4: Reordering STEP files..."
cd "${PROJECT_ROOT}"
python reorder_step.py --input_dir "${ABC_DIR}" --output_dir "${DATA_DIR}/reordered_step"

# Step 2: Render STEP files to images (optional, can be slow)
read -p "Do you want to render STEP files to images? This may take a long time. (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Step 2/4: Rendering STEP files to images..."
    python render_step.py --step_dir "${DATA_DIR}/reordered_step" --output_dir "${DATA_DIR}/step_images"
else
    echo "Step 2/4: Skipping image rendering"
fi

# Step 3: Construct RAG dataset
echo "Step 3/4: Constructing RAG dataset..."
python dataset_construct_rag.py \
    --caption_file "${CAPTION_FILE}" \
    --step_dir "${DATA_DIR}/reordered_step" \
    --output_dir "${DATA_DIR}/abc_rag"

# Step 4: Split dataset
echo "Step 4/4: Splitting dataset into train/val/test..."
python data_split.py \
    --input_file "${DATA_DIR}/abc_rag/dataset.json" \
    --output_dir "${DATA_DIR}/abc_rag" \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1

echo ""
echo "=========================================="
echo "✓ Dataset processing complete!"
echo "=========================================="
echo ""
echo "Processed datasets are in: ${DATA_DIR}/abc_rag/"
echo "  - train.json"
echo "  - val.json"
echo "  - test.json"
echo ""
echo "Next steps:"
echo "  1. Review the processed data"
echo "  2. Start training: python llama_finetuning_unsloth.py"
echo ""
