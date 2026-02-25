#!/usr/bin/env python3
"""
STEP File Entity Number Statistics Analyzer (Gen/GT Version)

This script analyzes STEP files in a directory structure with "gen" and "gt" subfolders
and provides statistics about entity numbers in each file.

Directory structure expected:
input_path/
├── subdir1/
│   ├── gen/
│   │   └── gen_subdir1.step
│   └── gt/
│       └── subdir1.step
├── subdir2/
│   ├── gen/
│   │   └── gen_subdir2.step
│   └── gt/
│       └── subdir2.step
└── ...

Usage:
    python step_entity_analyzer_gengt.py <input_path> --subfolder <gen|gt> [options]
"""

import os
import sys
import re
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


def extract_max_entity_number(step_file_path: str) -> int:
    """
    Extract the maximum entity number from a STEP file.
    
    Args:
        step_file_path: Path to the STEP file
        
    Returns:
        Maximum entity number found in the file, or 0 if none found
    """
    max_entity = 0
    entity_pattern = re.compile(r'^#(\d+)\s*=')
    
    try:
        with open(step_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                match = entity_pattern.match(line)
                if match:
                    entity_num = int(match.group(1))
                    max_entity = max(max_entity, entity_num)
    except Exception as e:
        print(f"Error reading file {step_file_path}: {e}")
        return 0
    
    return max_entity


def find_step_files(input_path: str, subfolder: str) -> List[str]:
    """
    Find all STEP files in the specified subfolder (gen or gt) of each subdirectory.
    
    Args:
        input_path: Path to the directory containing numbered subdirectories
        subfolder: Subfolder name ("gen" or "gt")
        
    Returns:
        List of paths to STEP files
    """
    step_files = []
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    
    if subfolder not in ['gen', 'gt']:
        raise ValueError(f"Subfolder must be 'gen' or 'gt', got: {subfolder}")
    
    # Look for subdirectories containing gen/gt folders
    processed_count = 0
    skipped_count = 0
    
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir():
            # Look for the specified subfolder (gen or gt)
            target_subfolder = subdir / subfolder
            if target_subfolder.exists() and target_subfolder.is_dir():
                # Look for STEP files in the target subfolder
                step_files_in_subfolder = list(target_subfolder.glob("*.step"))
                if step_files_in_subfolder:
                    step_files.extend([str(f) for f in step_files_in_subfolder])
                    processed_count += 1
                else:
                    print(f"Warning: No STEP files found in {target_subfolder}")
                    skipped_count += 1
            else:
                print(f"Warning: Subfolder '{subfolder}' not found in {subdir}")
                skipped_count += 1
    
    print(f"Found {len(step_files)} STEP files in '{subfolder}' subfolders")
    print(f"Processed: {processed_count} directories, Skipped: {skipped_count} directories")
    
    return step_files


def categorize_entity_count(entity_count: int) -> str:
    """
    Categorize entity count into predefined ranges.
    
    Args:
        entity_count: Number of entities
        
    Returns:
        Category string
    """
    if entity_count < 100:
        return "<100"
    elif entity_count < 200:
        return "100-200"
    elif entity_count < 300:
        return "200-300"
    elif entity_count < 400:
        return "300-400"
    elif entity_count < 500:
        return "400-500"
    else:
        return ">500"


def analyze_step_files(input_path: str, subfolder: str) -> Dict:
    """
    Analyze all STEP files in the specified subfolder and compute statistics.
    
    Args:
        input_path: Path to the directory containing numbered subdirectories
        subfolder: Subfolder name ("gen" or "gt")
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"Analyzing STEP files in '{subfolder}' subfolders of: {input_path}")
    
    # Find all STEP files
    step_files = find_step_files(input_path, subfolder)
    
    if not step_files:
        print("No STEP files found!")
        return {}
    
    # Extract entity counts
    entity_counts = []
    file_details = []
    
    for i, step_file in enumerate(step_files):
        if (i + 1) % 50 == 0:
            print(f"Processing file {i + 1}/{len(step_files)}")
        
        entity_count = extract_max_entity_number(step_file)
        entity_counts.append(entity_count)
        file_details.append({
            'file': step_file,
            'entity_count': entity_count,
            'category': categorize_entity_count(entity_count)
        })
    
    # Calculate statistics
    if entity_counts:
        max_entities = max(entity_counts)
        min_entities = min(entity_counts)
        avg_entities = statistics.mean(entity_counts)
        median_entities = statistics.median(entity_counts)
    else:
        max_entities = min_entities = avg_entities = median_entities = 0
    
    # Count by categories
    category_counts = {
        "<100": 0,
        "100-200": 0,
        "200-300": 0,
        "300-400": 0,
        "400-500": 0,
        ">500": 0
    }
    
    for detail in file_details:
        category_counts[detail['category']] += 1
    
    # Prepare results
    results = {
        'input_path': input_path,
        'subfolder': subfolder,
        'total_files': len(step_files),
        'max_entities': max_entities,
        'min_entities': min_entities,
        'avg_entities': avg_entities,
        'median_entities': median_entities,
        'category_counts': category_counts,
        'file_details': file_details
    }
    
    return results


def print_results(results: Dict):
    """
    Print the analysis results in a formatted manner.
    
    Args:
        results: Dictionary containing analysis results
    """
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("STEP FILE ENTITY ANALYSIS RESULTS (GEN/GT VERSION)")
    print("="*80)
    print(f"Input Path: {results['input_path']}")
    print(f"Subfolder Analyzed: {results['subfolder']}")
    print(f"Total STEP Files Analyzed: {results['total_files']}")
    print()
    
    print("ENTITY COUNT STATISTICS:")
    print("-"*40)
    print(f"Maximum Entity Count: {results['max_entities']}")
    print(f"Minimum Entity Count: {results['min_entities']}")
    print(f"Average Entity Count: {results['avg_entities']:.2f}")
    print(f"Median Entity Count: {results['median_entities']:.2f}")
    print()
    
    print("ENTITY COUNT DISTRIBUTION:")
    print("-"*40)
    total_files = results['total_files']
    for category, count in results['category_counts'].items():
        percentage = (count / total_files * 100) if total_files > 0 else 0
        print(f"{category:>10}: {count:>6} files ({percentage:>5.1f}%)")
    print()
    
    # Show some examples of files with highest and lowest entity counts
    file_details = results['file_details']
    if file_details:
        print("EXAMPLES:")
        print("-"*40)
        
        # Sort by entity count
        sorted_files = sorted(file_details, key=lambda x: x['entity_count'])
        
        print("Files with lowest entity counts:")
        for detail in sorted_files[:5]:
            filename = os.path.basename(detail['file'])
            parent_dir = os.path.basename(os.path.dirname(os.path.dirname(detail['file'])))
            print(f"  {parent_dir}/{results['subfolder']}/{filename}: {detail['entity_count']} entities")
        
        print("\nFiles with highest entity counts:")
        for detail in sorted_files[-5:]:
            filename = os.path.basename(detail['file'])
            parent_dir = os.path.basename(os.path.dirname(os.path.dirname(detail['file'])))
            print(f"  {parent_dir}/{results['subfolder']}/{filename}: {detail['entity_count']} entities")


def save_detailed_results(results: Dict, output_file: str):
    """
    Save detailed results to a CSV file.
    
    Args:
        results: Dictionary containing analysis results
        output_file: Path to output CSV file
    """
    try:
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['parent_dir', 'subfolder', 'filename', 'filepath', 'entity_count', 'category']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for detail in results['file_details']:
                parent_dir = os.path.basename(os.path.dirname(os.path.dirname(detail['file'])))
                writer.writerow({
                    'parent_dir': parent_dir,
                    'subfolder': results['subfolder'],
                    'filename': os.path.basename(detail['file']),
                    'filepath': detail['file'],
                    'entity_count': detail['entity_count'],
                    'category': detail['category']
                })
        
        print(f"Detailed results saved to: {output_file}")
    except ImportError:
        print("CSV module not available, skipping detailed output file")
    except Exception as e:
        print(f"Error saving detailed results: {e}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze STEP file entity numbers in gen/gt directory structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze 'gen' subfolder
  python step_entity_analyzer_gengt.py /path/to/checkpoint-9000 --subfolder gen
  
  # Analyze 'gt' subfolder with CSV output
  python step_entity_analyzer_gengt.py /path/to/checkpoint-9000 --subfolder gt --output gt_results.csv
  
Directory structure expected:
  input_path/
  ├── 00010154/
  │   ├── gen/
  │   │   └── gen_00010154.step
  │   └── gt/
  │       └── 00010154.step
  └── 00010413/
      ├── gen/
      │   └── gen_00010413.step
      └── gt/
          └── 00010413.step
        """
    )
    
    parser.add_argument('input_path', help='Path to directory containing numbered subdirectories with gen/gt folders')
    parser.add_argument('--subfolder', '-s', required=True, choices=['gen', 'gt'], 
                       help='Specify which subfolder to analyze: "gen" or "gt"')
    parser.add_argument('--output', '-o', help='Output CSV file for detailed results')
    
    args = parser.parse_args()
    
    try:
        # Run analysis
        results = analyze_step_files(args.input_path, args.subfolder)
        
        # Print results
        print_results(results)
        
        # Save detailed results if requested
        if args.output and results:
            save_detailed_results(results, args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
