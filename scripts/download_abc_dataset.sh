#!/bin/bash
# Download ABC Dataset and setup directory structure
# This script downloads the ABC dataset from the official source

set -e  # Exit on error

echo "=========================================="
echo "ABC Dataset Download Script"
echo "=========================================="
echo ""

# Configuration
ABC_DATASET_URL="https://archive.nyu.edu/handle/2451/43778"  # Update with actual download link
DATA_DIR="${DATA_DIR:-./data}"
ABC_DIR="${DATA_DIR}/abccad"

# Create directories
echo "Creating directory structure..."
mkdir -p "${ABC_DIR}"
mkdir -p "${DATA_DIR}/abc_rag"
mkdir -p "${DATA_DIR}/STEP_generated"

echo ""
echo "=========================================="
echo "IMPORTANT: ABC Dataset Download Instructions"
echo "=========================================="
echo ""
echo "The ABC dataset must be downloaded manually from:"
echo "${ABC_DATASET_URL}"
echo ""
echo "Download steps:"
echo "1. Visit the URL above and accept the terms"
echo "2. Download the dataset (STEP files)"
echo "3. Extract to: ${ABC_DIR}/"
echo "4. Expected structure:"
echo "   ${ABC_DIR}/"
echo "   ├── abc_0000_step/"
echo "   ├── abc_0001_step/"
echo "   └── ..."
echo ""
echo "After downloading, run the processing script:"
echo "  bash scripts/process_dataset.sh"
echo ""
echo "=========================================="

# Check if dataset already exists
if [ -d "${ABC_DIR}/abc_0000_step" ] || [ -d "${ABC_DIR}/abc_0001_step" ]; then
    echo ""
    echo "✓ ABC dataset appears to be already downloaded!"
    echo "  Location: ${ABC_DIR}"
else
    echo ""
    echo "⚠ ABC dataset not found in ${ABC_DIR}"
    echo "  Please download manually as instructed above"
    exit 1
fi
