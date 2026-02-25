#!/usr/bin/env python3
"""
Script to filter ABC RAG dataset by token threshold.
Removes data entries whose specified field exceeds the token threshold
and saves the deleted IDs to a CSV file.
"""

import argparse
import json
import os
import sys
import csv
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
import time
from tqdm import tqdm

def load_tokenizer(model_path):
    """Load the tokenizer from the specified model path."""
    try:
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Tokenizer loaded successfully: {tokenizer.__class__.__name__}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer from {model_path}: {e}")
        return None

def count_tokens(text, tokenizer):
    """Count tokens in the given text using the specified tokenizer."""
    if not text or text.strip() == "":
        return 0
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return 0

def get_field_content(sample, field_type):
    """Extract the content for the specified field type."""
    if field_type == "caption-step":
        caption = sample.get('caption', '')
        relevant_step = sample.get('relavant_step_file', '')
        return caption + relevant_step
    elif field_type == "output":
        return sample.get('output', '')
    elif field_type == "whole-data":
        caption = sample.get('caption', '')
        relevant_step = sample.get('relavant_step_file', '')
        output = sample.get('output', '')
        return caption + relevant_step + output
    else:
        raise ValueError(f"Invalid field type: {field_type}")

def filter_dataset(data_file, tokenizer, field_type, threshold):
    """Filter the dataset based on token threshold."""
    print(f"Filtering dataset: {data_file}")
    print(f"Field type: {field_type}")
    print(f"Token threshold: {threshold:,}")
    
    # Initialize lists
    filtered_data = []
    deleted_items = []
    deleted_ids = []
    token_counts = []
    
    # Load and process the dataset
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Dataset contains {len(data)} samples")
        
        # Process each sample
        for i, sample in enumerate(tqdm(data, desc="Processing samples")):
            # Get the content for the specified field
            field_content = get_field_content(sample, field_type)
            
            # Count tokens
            token_count = count_tokens(field_content, tokenizer)
            token_counts.append(token_count)
            
            # Check if token count exceeds threshold
            if token_count <= threshold:
                filtered_data.append(sample)
            else:
                # Record deleted item and ID
                deleted_items.append(sample)
                original_id = sample.get('id_original', f'unknown_{i}')
                deleted_ids.append({
                    'id_original': original_id,
                    'token_count': token_count,
                    'threshold': threshold,
                    'field_type': field_type
                })
            
            # Progress update every 1000 samples
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} samples...")
    
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None, None, None, None
    
    return filtered_data, deleted_items, deleted_ids, token_counts

def save_deleted_items_json(deleted_items, output_file):
    """Save the deleted items to a JSON file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deleted_items, f, indent=2, ensure_ascii=False)
        print(f"Deleted items saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving deleted items: {e}")
        return False

def update_original_file(data_file, filtered_data):
    """Update the original JSON file with filtered data."""
    try:
        # Create a backup of the original file
        backup_file = f"{data_file}.backup"
        import shutil
        shutil.copy2(data_file, backup_file)
        print(f"Original file backed up to: {backup_file}")
        
        # Write the filtered data back to the original file
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        print(f"Original file updated with filtered data: {data_file}")
        return True
    except Exception as e:
        print(f"Error updating original file: {e}")
        return False

def save_deleted_ids(deleted_ids, output_file):
    """Save the deleted IDs to a CSV file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if deleted_ids:
                fieldnames = deleted_ids[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(deleted_ids)
        
        print(f"Deleted IDs saved to: {output_file}")
        return True
    except Exception as e:
        print(f"Error saving deleted IDs: {e}")
        return False

def print_summary(original_count, filtered_count, deleted_count, token_counts, threshold):
    """Print a summary of the filtering operation."""
    print(f"\n{'='*60}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*60}")
    
    print(f"Original dataset size: {original_count:,}")
    print(f"Filtered dataset size: {filtered_count:,}")
    print(f"Deleted entries: {deleted_count:,}")
    print(f"Retention rate: {(filtered_count/original_count)*100:.2f}%")
    print(f"Deletion rate: {(deleted_count/original_count)*100:.2f}%")
    
    if token_counts:
        token_counts = np.array(token_counts)
        print(f"\nToken count statistics:")
        print(f"  Min: {int(np.min(token_counts)):,}")
        print(f"  Max: {int(np.max(token_counts)):,}")
        print(f"  Mean: {float(np.mean(token_counts)):.2f}")
        print(f"  Median: {float(np.median(token_counts)):.2f}")
        print(f"  Std: {float(np.std(token_counts)):.2f}")
        print(f"  Threshold: {threshold:,}")
        
        # Count entries above threshold
        above_threshold = np.sum(token_counts > threshold)
        print(f"  Entries above threshold: {above_threshold:,} ({above_threshold/original_count*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="Filter ABC RAG dataset by token threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_by_token_threshold.py /path/to/dataset.json llama 8000 "caption-step"
  python filter_by_token_threshold.py /path/to/dataset.json qwen 12000 "output"
  python filter_by_token_threshold.py /path/to/dataset.json llama 20000 "whole-data"
        """
    )
    
    parser.add_argument(
        'data_file',
        help='Path to the JSON data file to filter'
    )
    
    parser.add_argument(
        'tokenizer_type',
        choices=['llama', 'qwen'],
        help='Choose tokenizer: llama (Llama 3.2) or qwen (Qwen2.5)'
    )
    
    parser.add_argument(
        'threshold',
        type=int,
        help='Token threshold (entries exceeding this will be deleted)'
    )
    
    parser.add_argument(
        'field_type',
        choices=['caption-step', 'output', 'whole-data'],
        help='Field to check: caption-step, output, or whole-data'
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
    
    # Set model path based on tokenizer type
    if args.tokenizer_type == 'llama':
        model_path = '/home/group/cad_codebased/llama_3.2/Llama_3.2_3B'
        model_name = 'llama_3.2_3b'
    else:  # qwen
        # Try local path first, then HuggingFace model ID
        local_path = '/home/group/cad_codebased/Qwen2_5_3B'
        if os.path.exists(local_path):
            model_path = local_path
        else:
            # Use HuggingFace model ID (will download if needed)
            model_path = 'unsloth/Qwen2.5-3B'
        model_name = 'qwen2.5_3b'
    
    # Check if model path exists (skip check for HuggingFace model IDs)
    # HuggingFace model IDs are in format "org/model-name" (single slash)
    is_huggingface_id = '/' in model_path and not os.path.exists(model_path) and model_path.count('/') == 1
    if not is_huggingface_id and not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        sys.exit(1)
    
    print(f"Processing data file: {args.data_file}")
    print(f"Using {args.tokenizer_type.upper()} tokenizer from: {model_path}")
    print(f"Field type: {args.field_type}")
    print(f"Token threshold: {args.threshold:,}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_path)
    if not tokenizer:
        print("Error: Failed to load tokenizer. Cannot proceed.")
        sys.exit(1)
    
    # Filter dataset
    start_time = time.time()
    filtered_data, deleted_items, deleted_ids, token_counts = filter_dataset(
        args.data_file, tokenizer, args.field_type, args.threshold
    )
    processing_time = time.time() - start_time
    
    if filtered_data is None:
        print("Error: Failed to filter dataset.")
        sys.exit(1)
    
    # Generate output filenames
    data_path = Path(args.data_file)
    folder_name = data_path.parent.name
    file_name = data_path.stem
    base_filename = f"{folder_name}_{file_name}"
    
    # Save deleted items to JSON
    deleted_items_output = f"/home/group/cad_codebased/data_filter_long/filtered_datasets/{base_filename}_deleted_{args.tokenizer_type}_{args.threshold}.json"
    save_deleted_items_json(deleted_items, deleted_items_output)
    
    # Save deleted IDs to CSV
    deleted_ids_output = f"/home/group/cad_codebased/data_filter_long/deleted_list/{base_filename}_deleted_{args.tokenizer_type}_{args.threshold}.csv"
    save_deleted_ids(deleted_ids, deleted_ids_output)
    
    # Update original file with filtered data
    update_original_file(args.data_file, filtered_data)
    
    # Print summary
    print_summary(
        len(filtered_data) + len(deleted_ids),
        len(filtered_data),
        len(deleted_ids),
        token_counts,
        args.threshold
    )
    
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print("Dataset filtering completed successfully!")

if __name__ == "__main__":
    main()
