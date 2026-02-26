#!/usr/bin/env python3
"""
STEP File Number Rounding Script
================================

This script rounds floating-point numbers in STEP files to 6 decimal places.
It processes both single STEP files and directories containing STEP files.

Usage:
    python round_step_numbers.py <input_path>

Input path can be:
1. A single .step file
2. A directory containing subdirectories with .step files

# Process a single file
python round_step_numbers.py data/reordered_step/0002/00020000/00020000_7259f13b84f34cce843db1f8_step_043.step

# Process an entire directory
python round_step_numbers.py data/reordered_step/0002

# Process the entire reordered_step directory
python round_step_numbers.py data/reordered_step

Output:
    Processed files are saved under ./data/reorder_round_step (configurable via --output-dir)
    maintaining the same directory structure as input.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Union, List

def round_float_numbers(content: str) -> str:
    """
    Round floating-point numbers in STEP file content to 6 decimal places.
    
    Args:
        content: The STEP file content as a string
        
    Returns:
        Modified content with rounded numbers
    """
    # Pattern to match floating-point numbers in various formats:
    # - Regular decimals: 0.123456789
    # - Scientific notation: 1.23E-17
    # - Negative numbers: -0.123456789
    # - Numbers in parentheses: ( 0.123456789, -0.456789 )
    # - Numbers with trailing zeros: 0.123456789000000
    
    # This regex matches floating-point numbers but excludes entity numbers (#123)
    float_pattern = r'\b(-?\d+\.\d+E?[+-]?\d*)\b'
    
    def round_match(match):
        number_str = match.group(1)
        
        # Skip if this looks like an entity number (starts with #)
        if match.start() > 0 and content[match.start()-1] == '#':
            return number_str
        
        try:
            # Check if it's scientific notation
            if 'E' in number_str or 'e' in number_str:
                # Split into mantissa and exponent
                if 'E' in number_str:
                    mantissa, exponent = number_str.split('E', 1)
                else:
                    mantissa, exponent = number_str.split('e', 1)
                
                # Round the mantissa to 6 decimal places
                mantissa_float = float(mantissa)
                rounded_mantissa = round(mantissa_float, 6)
                
                # Format mantissa to exactly 6 decimal places
                formatted_mantissa = f"{rounded_mantissa:.6f}"
                
                # Reconstruct the scientific notation
                return f"{formatted_mantissa}E{exponent}"
            else:
                # Regular decimal - parse and round
                number = float(number_str)
                rounded = round(number, 6)
                
                # Format to exactly 6 decimal places
                return f"{rounded:.6f}"
                
        except ValueError:
            # If we can't parse it as a float, return unchanged
            return number_str
    
    # Apply the rounding to the content
    return re.sub(float_pattern, round_match, content)

def process_single_step_file(input_path: str, output_dir: str) -> None:
    """
    Process a single STEP file.
    
    Args:
        input_path: Path to the input STEP file
        output_dir: Output directory path
    """
    input_path = Path(input_path)
    output_path = Path(output_dir) / input_path.name
    
    print(f"Processing: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Round the numbers
        rounded_content = round_float_numbers(content)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the processed content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(rounded_content)
        
        print(f"Saved: {output_path}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Process all STEP files in a directory and its subdirectories.
    
    Args:
        input_dir: Path to the input directory
        output_dir: Output directory path
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print(f"Processing directory: {input_path}")
    
    # Find all .step files recursively
    step_files = list(input_path.rglob("*.step"))
    
    if not step_files:
        print(f"No .step files found in {input_path}")
        return
    
    print(f"Found {len(step_files)} STEP files")
    
    for step_file in step_files:
        # Calculate relative path from input directory
        rel_path = step_file.relative_to(input_path)
        
        # Create corresponding output path
        output_file_path = output_path / rel_path
        
        print(f"Processing: {step_file}")
        
        try:
            with open(step_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Round the numbers
            rounded_content = round_float_numbers(content)
            
            # Create output directory if it doesn't exist
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the processed content
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(rounded_content)
            
            print(f"Saved: {output_file_path}")
            
        except Exception as e:
            print(f"Error processing {step_file}: {e}")

def main():
    """Main function to handle command line arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Round floating-point numbers in STEP files to 6 decimal places"
    )
    parser.add_argument(
        "input_path",
        help="Path to a single STEP file or directory containing STEP files"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/reorder_round_step",
        help="Output directory (default: ./data/reorder_round_step)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    # Validate input path
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process based on input type
    if input_path.is_file():
        if input_path.suffix.lower() != '.step':
            print(f"Error: Input file is not a .step file: {input_path}")
            sys.exit(1)
        process_single_step_file(str(input_path), str(output_dir))
    elif input_path.is_dir():
        process_directory(str(input_path), str(output_dir))
    else:
        print(f"Error: Input path is neither a file nor directory: {input_path}")
        sys.exit(1)
    
    print("Processing completed!")

if __name__ == "__main__":
    main() 