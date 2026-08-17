#!/bin/bash
# Build the RAG training dataset from the raw ABC dataset + captions.
#
# This mirrors the pipeline documented in README.md ("Build the Full RAG Dataset").
# Steps 2-4 are driven by configuration blocks at the top of the respective
# scripts rather than by command-line flags, so this script checks prerequisites,
# points you at what needs editing, and runs each stage in order.

set -e  # Exit on error

echo "=========================================="
echo "STEP-LLM Dataset Processing"
echo "=========================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/dataset}"
ABC_DIR="${ABC_DIR:-${DATA_DIR}/abccad/step_under500}"
CAPTION_FILE="${CAPTION_FILE:-${DATA_DIR}/cad_captions_0-500.csv}"
ROUNDED_DIR="${ROUNDED_DIR:-${DATA_DIR}/rounded_step}"

# Check prerequisites
if [ ! -d "${ABC_DIR}" ]; then
    echo "Error: ABC STEP files not found at ${ABC_DIR}"
    echo "Please run: bash scripts/download_abc_dataset.sh"
    echo "(or set ABC_DIR to the directory holding the raw .step files)"
    exit 1
fi

if [ ! -f "${CAPTION_FILE}" ]; then
    echo "Error: Caption file not found at ${CAPTION_FILE}"
    exit 1
fi

echo "Project root:  ${PROJECT_ROOT}"
echo "ABC directory: ${ABC_DIR}"
echo "Caption file:  ${CAPTION_FILE}"
echo ""

# ── Step 1: Normalise floating-point precision ────────────────────────────────
echo "Step 1/4: Rounding floating-point numbers in raw STEP files..."
python data_preparation/round_step_numbers.py "${ABC_DIR}" --output-dir "${ROUNDED_DIR}"
echo ""

# ── Step 2: DFS reorder + annotate ────────────────────────────────────────────
echo "Step 2/4: DFS-restructuring STEP files..."
echo ""
echo "  batch_restructure.sh is configured by the SRC_BASE / DEST_BASE variables"
echo "  at the top of the script. Set them to:"
echo "    SRC_BASE=\"${ROUNDED_DIR}\""
echo "    DEST_BASE=\"${DATA_DIR}/dfs_step\""
echo ""
read -p "Have you updated data_preparation/batch_restructure.sh? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash data_preparation/batch_restructure.sh
else
    echo "Skipping. Edit the script and re-run, or run it manually:"
    echo "  bash data_preparation/batch_restructure.sh"
    exit 1
fi
echo ""

# ── Step 3: Build the RAG dataset ─────────────────────────────────────────────
echo "Step 3/4: Constructing RAG dataset..."
echo ""
echo "  dataset_construct_rag.py is configured by the CSV_FILE / STEP_FILE_DIRS /"
echo "  OUTPUT_JSON_PATH constants at the top of the script. Make sure"
echo "  STEP_FILE_DIRS lists the chunk directories under ${DATA_DIR}/dfs_step/."
echo ""
read -p "Have you updated data_preparation/dataset_construct_rag.py? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python data_preparation/dataset_construct_rag.py
else
    echo "Skipping. Edit the script and re-run, or run it manually:"
    echo "  python data_preparation/dataset_construct_rag.py"
    exit 1
fi
echo ""

# ── Step 4: Split into train / val / test ─────────────────────────────────────
echo "Step 4/4: Splitting dataset into train/val/test (70/10/20)..."
python data_preparation/data_split.py
echo ""

echo "=========================================="
echo "✓ Dataset processing complete!"
echo "=========================================="
echo ""
echo "Processed datasets are in: ${DATA_DIR}/abc_rag/train_val_test/"
echo "  - train.json"
echo "  - val.json"
echo "  - test.json"
echo ""
echo "Next steps:"
echo "  1. Review the processed data"
echo "  2. Edit the configuration block at the top of llama3_SFT_response.py"
echo "  3. Start training: python llama3_SFT_response.py"
echo ""
