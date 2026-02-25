#!/bin/bash

# Script to run render_step.py with virtual X server
# This solves the "qt.qpa.xcb: could not connect to display" issue

# Check if xvfb-run is available
if ! command -v xvfb-run &> /dev/null; then
    echo "xvfb-run not found. Trying to install xvfb..."
    
    # Try to install without sudo first (user might have permissions)
    if command -v apt &> /dev/null; then
        echo "Attempting to install xvfb using apt..."
        apt update && apt install -y xvfb 2>/dev/null || {
            echo "Failed to install xvfb without sudo. Please run:"
            echo "sudo apt update && sudo apt install -y xvfb"
            echo "Then run this script again."
            exit 1
        }
    else
        echo "Package manager 'apt' not found. Please install xvfb manually."
        exit 1
    fi
fi

# Activate conda environment and run the script with virtual display
conda activate /home/group/cad_codebased/brepgen_env

echo "Starting virtual X server and running render_step.py..."
xvfb-run -a -s "-screen 0 1024x768x24" python /home/group/cad_codebased/render_step.py

echo "Rendering complete."










