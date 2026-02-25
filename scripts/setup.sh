#!/bin/bash
# One-click setup script for CAD Code-Based Generation

set -e  # Exit on error

echo "=========================================="
echo "CAD Code-Based Generation Setup"
echo "=========================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: ${python_version}"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found"
    USE_CONDA=true
else
    echo "Conda not found, using pip/venv"
    USE_CONDA=false
fi

echo ""
echo "Select setup option:"
echo "1) Full setup (conda environment + dependencies + models + dataset)"
echo "2) Dependencies only (skip environment creation)"
echo "3) Minimal setup (just install Python packages)"
echo ""
read -p "Enter choice [1-3]: " setup_choice

# Environment setup
if [ "$setup_choice" = "1" ] && [ "$USE_CONDA" = true ]; then
    echo ""
    echo "Creating conda environment 'step_llm'..."
    
    # Check if environment.yml exists
    if [ -f "environment_minimal.yml" ]; then
        echo "Using environment_minimal.yml..."
        conda env create -f environment_minimal.yml -y
    else
        echo "Creating environment manually..."
        conda create -n step_llm python=3.10 -y
    fi
    
    echo "Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate step_llm
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."

# Install PyTorch first (CUDA version)
read -p "Do you have a CUDA-capable GPU? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Installing PyTorch with CUDA support..."
    if [ "$USE_CONDA" = true ]; then
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio
fi

# Install other requirements
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install pythonocc-core via conda if available
if [ "$USE_CONDA" = true ]; then
    echo "Installing pythonocc-core..."
    conda install -c conda-forge pythonocc-core -y || echo "Warning: pythonocc-core installation failed"
fi

echo "✓ Dependencies installed"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""

if [ "$USE_CONDA" = true ]; then
    echo "To activate the environment:"
    echo "  conda activate step_llm"
    echo ""
fi

echo "Next steps:"
echo "1. Update .env file with your API keys and paths"
echo "2. Download base model from Hugging Face (see README.md)"
echo "3. Download dataset: bash scripts/download_abc_dataset.sh"
echo "4. Process dataset: bash scripts/process_dataset.sh"
echo "5. Download LoRA adapter from our release (see README.md)"
echo "6. Try inference: python examples/basic_inference.py"
echo ""
echo "For more information, see README.md"
echo ""
