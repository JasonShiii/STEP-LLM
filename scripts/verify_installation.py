#!/usr/bin/env python3
"""
Environment verification script for CAD Text-to-STEP Generation project.

This script checks if all required packages are installed and configured correctly.
"""

import sys

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)

def print_check(name, status, details=""):
    status_symbol = "✓" if status else "✗"
    print(f"{status_symbol} {name:<30} {details}")
    return status

def main():
    print_header("CAD Text-to-STEP Environment Verification")
    
    all_passed = True
    
    # Python version
    print_header("Python Environment")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = sys.version_info >= (3, 10)
    all_passed &= print_check("Python version", python_ok, python_version)
    
    # Core packages
    print_header("Core Packages")
    
    # PyTorch
    try:
        import torch
        torch_version = torch.__version__
        torch_ok = True
        all_passed &= print_check("PyTorch", torch_ok, torch_version)
        
        # CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            all_passed &= print_check("CUDA", cuda_available, f"{cuda_version} ({gpu_name})")
        else:
            print_check("CUDA", False, "Not available (CPU only)")
    except ImportError:
        all_passed &= print_check("PyTorch", False, "Not installed")
    
    # Transformers
    try:
        import transformers
        tf_version = transformers.__version__
        all_passed &= print_check("Transformers", True, tf_version)
    except ImportError:
        all_passed &= print_check("Transformers", False, "Not installed")
    
    # Unsloth
    try:
        import unsloth
        all_passed &= print_check("Unsloth", True, "Installed")
    except ImportError:
        all_passed &= print_check("Unsloth", False, "Not installed")
    
    # Data processing packages
    print_header("Data Processing")
    
    # NumPy
    try:
        import numpy as np
        np_version = np.__version__
        all_passed &= print_check("NumPy", True, np_version)
    except ImportError:
        all_passed &= print_check("NumPy", False, "Not installed")
    
    # Pandas
    try:
        import pandas as pd
        pd_version = pd.__version__
        all_passed &= print_check("Pandas", True, pd_version)
    except ImportError:
        all_passed &= print_check("Pandas", False, "Not installed")
    
    # Datasets
    try:
        import datasets
        ds_version = datasets.__version__
        all_passed &= print_check("Datasets", True, ds_version)
    except ImportError:
        all_passed &= print_check("Datasets", False, "Not installed")
    
    # Retrieval packages
    print_header("Retrieval & Embeddings")
    
    # FAISS
    try:
        import faiss
        all_passed &= print_check("FAISS", True, "Installed")
    except ImportError:
        all_passed &= print_check("FAISS", False, "Not installed")
    
    # Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
        st_version = sentence_transformers.__version__
        all_passed &= print_check("Sentence Transformers", True, st_version)
    except ImportError:
        all_passed &= print_check("Sentence Transformers", False, "Not installed")
    
    # Training packages
    print_header("Training Utilities")
    
    # TRL
    try:
        import trl
        trl_version = trl.__version__
        all_passed &= print_check("TRL", True, trl_version)
    except ImportError:
        all_passed &= print_check("TRL", False, "Not installed")
    
    # PEFT
    try:
        import peft
        peft_version = peft.__version__
        all_passed &= print_check("PEFT", True, peft_version)
    except ImportError:
        all_passed &= print_check("PEFT", False, "Not installed")
    
    # Accelerate
    try:
        import accelerate
        acc_version = accelerate.__version__
        all_passed &= print_check("Accelerate", True, acc_version)
    except ImportError:
        all_passed &= print_check("Accelerate", False, "Not installed")
    
    # bitsandbytes
    try:
        import bitsandbytes
        bnb_version = bitsandbytes.__version__
        all_passed &= print_check("BitsAndBytes", True, bnb_version)
    except ImportError:
        all_passed &= print_check("BitsAndBytes", False, "Not installed")
    
    # Optional packages
    print_header("Optional Packages")
    
    # Jupyter
    try:
        import jupyter
        print_check("Jupyter", True, "Installed")
    except ImportError:
        print_check("Jupyter", False, "Not installed (optional)")
    
    # WandB
    try:
        import wandb
        wandb_version = wandb.__version__
        print_check("WandB", True, f"{wandb_version} (optional)")
    except ImportError:
        print_check("WandB", False, "Not installed (optional)")
    
    # OpenCASCADE (pythonocc-core)
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        print_check("pythonocc-core", True, "Installed")
    except ImportError:
        print_check("pythonocc-core", False, "Not installed (optional)")
    
    # Trimesh (alternative)
    try:
        import trimesh
        print_check("Trimesh", True, "Installed (optional)")
    except ImportError:
        print_check("Trimesh", False, "Not installed (optional)")
    
    # Final summary
    print_header("Summary")
    
    if all_passed:
        print("✓ All required packages are installed correctly!")
        print("\nYou can now:")
        print("  1. Download dataset: bash scripts/download_abc_dataset.sh")
        print("  2. Try inference: python examples/basic_inference.py")
        print("  3. Start training: See continuous_training_junyang/")
        return 0
    else:
        print("✗ Some required packages are missing or incorrectly configured.")
        print("\nTo fix:")
        print("  1. Review ENVIRONMENT_SETUP.md")
        print("  2. Run: bash scripts/setup.sh")
        print("  3. Or install manually: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
