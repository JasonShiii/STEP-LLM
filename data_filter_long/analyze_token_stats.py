#!/usr/bin/env python3
"""
Script to analyze token statistics for ABC RAG dataset.
Calculates max, min, average, 95th percentile, and 99th percentile token counts
for caption + relevant_step_file, output, and caption + relevant_step_file + output
using both Llama 3.2 and Qwen2.5 tokenizers.
"""

import argparse
import json
import os
import sys
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

def calculate_percentile(data, percentile):
    """Calculate the specified percentile of the data."""
    if len(data) == 0:
        return 0
    return np.percentile(data, percentile)

def calculate_statistics(token_counts):
    """Calculate comprehensive statistics for token counts."""
    if len(token_counts) == 0:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std': 0,
            'p95': 0,
            'p99': 0
        }
    
    token_counts = np.array(token_counts)
    return {
        'count': len(token_counts),
        'min': int(np.min(token_counts)),
        'max': int(np.max(token_counts)),
        'mean': float(np.mean(token_counts)),
        'median': float(np.median(token_counts)),
        'std': float(np.std(token_counts)),
        'p95': int(calculate_percentile(token_counts, 95)),
        'p99': int(calculate_percentile(token_counts, 99))
    }

def process_dataset(data_file, llama_tokenizer, qwen_tokenizer):
    """Process the dataset and calculate token statistics for both tokenizers."""
    print(f"Processing dataset: {data_file}")
    
    # Initialize token count lists for both tokenizers
    llama_stats = {
        'caption_relevant_step': [],
        'output': [],
        'caption_relevant_step_output': []
    }
    
    qwen_stats = {
        'caption_relevant_step': [],
        'output': [],
        'caption_relevant_step_output': []
    }
    
    # Load and process the dataset
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Dataset contains {len(data)} samples")
        
        # Process each sample
        for i, sample in enumerate(tqdm(data, desc="Processing samples")):
            # Extract fields
            caption = sample.get('caption', '')
            relevant_step_file = sample.get('relavant_step_file', '')
            output = sample.get('output', '')
            
            # Combine fields
            caption_relevant_step = caption + relevant_step_file
            caption_relevant_step_output = caption + relevant_step_file + output
            
            # Count tokens for Llama tokenizer
            if llama_tokenizer:
                llama_stats['caption_relevant_step'].append(
                    count_tokens(caption_relevant_step, llama_tokenizer)
                )
                llama_stats['output'].append(
                    count_tokens(output, llama_tokenizer)
                )
                llama_stats['caption_relevant_step_output'].append(
                    count_tokens(caption_relevant_step_output, llama_tokenizer)
                )
            
            # Count tokens for Qwen tokenizer
            if qwen_tokenizer:
                qwen_stats['caption_relevant_step'].append(
                    count_tokens(caption_relevant_step, qwen_tokenizer)
                )
                qwen_stats['output'].append(
                    count_tokens(output, qwen_tokenizer)
                )
                qwen_stats['caption_relevant_step_output'].append(
                    count_tokens(caption_relevant_step_output, qwen_tokenizer)
                )
            
            # Progress update every 1000 samples
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} samples...")
    
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return None, None
    
    # Calculate statistics for both tokenizers
    print("\nCalculating statistics...")
    
    llama_results = {}
    if llama_tokenizer:
        for field, counts in llama_stats.items():
            llama_results[field] = calculate_statistics(counts)
    
    qwen_results = {}
    if qwen_tokenizer:
        for field, counts in qwen_stats.items():
            qwen_results[field] = calculate_statistics(counts)
    
    return llama_results, qwen_results

def save_results(llama_results, qwen_results, output_file):
    """Save the token statistics to a text file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TOKEN STATISTICS ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        # Llama 3.2 results
        if llama_results:
            f.write("LLAMA 3.2 TOKENIZER RESULTS:\n")
            f.write("-" * 30 + "\n\n")
            
            for field, stats in llama_results.items():
                field_name = field.replace('_', ' + ').upper()
                f.write(f"{field_name}:\n")
                f.write(f"  Count: {stats['count']:,}\n")
                f.write(f"  Min: {stats['min']:,}\n")
                f.write(f"  Max: {stats['max']:,}\n")
                f.write(f"  Average: {stats['mean']:.2f}\n")
                f.write(f"  Median: {stats['median']:.2f}\n")
                f.write(f"  Std: {stats['std']:.2f}\n")
                f.write(f"  95th Percentile: {stats['p95']:,}\n")
                f.write(f"  99th Percentile: {stats['p99']:,}\n")
                f.write("\n")
        
        # Qwen2.5 results
        if qwen_results:
            f.write("QWEN2.5 TOKENIZER RESULTS:\n")
            f.write("-" * 30 + "\n\n")
            
            for field, stats in qwen_results.items():
                field_name = field.replace('_', ' + ').upper()
                f.write(f"{field_name}:\n")
                f.write(f"  Count: {stats['count']:,}\n")
                f.write(f"  Min: {stats['min']:,}\n")
                f.write(f"  Max: {stats['max']:,}\n")
                f.write(f"  Average: {stats['mean']:.2f}\n")
                f.write(f"  Median: {stats['median']:.2f}\n")
                f.write(f"  Std: {stats['std']:.2f}\n")
                f.write(f"  95th Percentile: {stats['p95']:,}\n")
                f.write(f"  99th Percentile: {stats['p99']:,}\n")
                f.write("\n")
        
        # Summary comparison
        if llama_results and qwen_results:
            f.write("SUMMARY COMPARISON:\n")
            f.write("-" * 20 + "\n\n")
            
            for field in ['caption_relevant_step', 'output', 'caption_relevant_step_output']:
                if field in llama_results and field in qwen_results:
                    field_name = field.replace('_', ' + ').upper()
                    f.write(f"{field_name}:\n")
                    f.write(f"  Llama 3.2 - 95th percentile: {llama_results[field]['p95']:,}\n")
                    f.write(f"  Qwen2.5 - 95th percentile: {qwen_results[field]['p95']:,}\n")
                    f.write(f"  Llama 3.2 - 99th percentile: {llama_results[field]['p99']:,}\n")
                    f.write(f"  Qwen2.5 - 99th percentile: {qwen_results[field]['p99']:,}\n")
                    f.write("\n")
    
    print(f"Results saved to: {output_file}")

def print_summary(llama_results, qwen_results):
    """Print a summary of the token statistics to console."""
    print(f"\n{'='*80}")
    print(f"TOKEN STATISTICS SUMMARY")
    print(f"{'='*80}")
    
    # Llama 3.2 results
    if llama_results:
        print(f"\nLLAMA 3.2 TOKENIZER RESULTS:")
        print(f"{'-'*30}")
        
        for field, stats in llama_results.items():
            field_name = field.replace('_', ' + ').upper()
            print(f"\n{field_name}:")
            print(f"  Count: {stats['count']:,}")
            print(f"  Min: {stats['min']:,}")
            print(f"  Max: {stats['max']:,}")
            print(f"  Average: {stats['mean']:.2f}")
            print(f"  Median: {stats['median']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  95th Percentile: {stats['p95']:,}")
            print(f"  99th Percentile: {stats['p99']:,}")
    
    # Qwen2.5 results
    if qwen_results:
        print(f"\nQWEN2.5 TOKENIZER RESULTS:")
        print(f"{'-'*30}")
        
        for field, stats in qwen_results.items():
            field_name = field.replace('_', ' + ').upper()
            print(f"\n{field_name}:")
            print(f"  Count: {stats['count']:,}")
            print(f"  Min: {stats['min']:,}")
            print(f"  Max: {stats['max']:,}")
            print(f"  Average: {stats['mean']:.2f}")
            print(f"  Median: {stats['median']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  95th Percentile: {stats['p95']:,}")
            print(f"  99th Percentile: {stats['p99']:,}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze token statistics for ABC RAG dataset using both Llama 3.2 and Qwen2.5 tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_token_stats.py /home/group/cad_codebased/data/abc_rag/20500_dfs/val.json
  python analyze_token_stats.py /path/to/your/dataset.json
        """
    )
    
    parser.add_argument(
        'data_file',
        help='Path to the JSON data file to analyze'
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
    
    # Set model paths
    llama_path = '/home/group/cad_codebased/llama_3.2/Llama_3.2_3B'
    qwen_path = '/home/group/cad_codebased/Qwen2_5_3B'
    
    # Check if model paths exist
    if not os.path.exists(llama_path):
        print(f"Warning: Llama 3.2 model path not found: {llama_path}")
        llama_path = None
    
    if not os.path.exists(qwen_path):
        print(f"Warning: Qwen2.5 model path not found: {qwen_path}")
        print(f"Using HuggingFace model ID instead: unsloth/Qwen2.5-3B")
        qwen_path = 'unsloth/Qwen2.5-3B'
    
    if not llama_path and not qwen_path:
        print("Error: Neither model path exists. Cannot proceed.")
        sys.exit(1)
    
    print(f"Processing data file: {args.data_file}")
    if llama_path:
        print(f"Llama 3.2 model path: {llama_path}")
    if qwen_path:
        print(f"Qwen2.5 model path: {qwen_path}")
    
    # Load tokenizers
    llama_tokenizer = None
    qwen_tokenizer = None
    
    if llama_path:
        llama_tokenizer = load_tokenizer(llama_path)
    
    if qwen_path:
        qwen_tokenizer = load_tokenizer(qwen_path)
    
    if not llama_tokenizer and not qwen_tokenizer:
        print("Error: Failed to load any tokenizer. Cannot proceed.")
        sys.exit(1)
    
    # Process dataset
    start_time = time.time()
    llama_results, qwen_results = process_dataset(args.data_file, llama_tokenizer, qwen_tokenizer)
    processing_time = time.time() - start_time
    
    if llama_results is None or qwen_results is None:
        print("Error: Failed to process dataset.")
        sys.exit(1)
    
    # Print summary
    print_summary(llama_results, qwen_results)
    
    # Generate output filename
    data_path = Path(args.data_file)
    folder_name = data_path.parent.name
    file_name = data_path.stem
    output_filename = f"{folder_name}_{file_name}.txt"
    output_path = Path("/home/group/cad_codebased/data_filter_long/token_stats") / output_filename
    
    # Save results
    save_results(llama_results, qwen_results, output_path)
    
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print("Token statistics have been calculated and saved successfully!")

if __name__ == "__main__":
    main()

