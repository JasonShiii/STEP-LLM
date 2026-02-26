#!/usr/bin/env python3
"""
Restore Valid STEP File

This script restores a valid STEP file from a restructured STEP file by:
  1) Replacing MDGPR placeholder tokens (#ID_n) with their mapped numeric IDs
  2) Removing all training annotations/comments (/* ... */)

Run this after step_restructurer.py to produce a valid (loadable) STEP file.
It supports processing a single file path or a directory (recursively).

Usage:
  python restore_step_valid.py <path> [-o OUTPUT_DIR]

Examples:
  # Single file (outputs to ./restored_output by default)
  python restore_step_valid.py restructured_output/00010007_restructured.step

  # Directory (outputs to ./restored_output by default)
  python restore_step_valid.py restructured_output/

  # Custom output directory
  python restore_step_valid.py restructured_output/ -o /path/to/custom/output
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


MDGPR_TYPE = 'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION'


def read_text(path: Path) -> List[str]:
    try:
        return path.read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1').splitlines()


def write_text(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding='utf-8')


def find_mdgpr_root_indices(lines: List[str]) -> List[int]:
    pattern = re.compile(r'^\s*#\d+\s*=\s*' + re.escape(MDGPR_TYPE) + r'\s*\(')
    indices = []
    for i, line in enumerate(lines):
        if pattern.search(line):
            indices.append(i)
    return indices


def collect_placeholder_mapping(lines: List[str], start_idx: int, end_idx: int) -> Dict[str, int]:
    """Collect mapping lines /* ID_n = <num> */ between start and end indices (exclusive)."""
    mapping: Dict[str, int] = {}
    map_re = re.compile(r'^\s*/\*\s*ID_(\d+)\s*=\s*(\d+)\s*\*/\s*$')
    for i in range(start_idx + 1, end_idx):
        m = map_re.match(lines[i])
        if m:
            key = f"ID_{m.group(1)}"
            val = int(m.group(2))
            mapping[key] = val
    return mapping


def replace_placeholders_in_mdgpr_line(line: str, mapping: Dict[str, int]) -> str:
    # Replace #ID_n with #<mapped_number>
    def repl(m: re.Match) -> str:
        key = f"ID_{m.group(1)}"
        if key in mapping:
            return f"#{mapping[key]}"
        return m.group(0)

    return re.sub(r'#ID_(\d+)\b', repl, line)


def strip_annotations(lines: List[str]) -> List[str]:
    """Remove all annotations of the form /* ... */ anywhere in a line.
    After removal, drop lines that become empty or whitespace-only.
    """
    res: List[str] = []
    comment_re = re.compile(r'/\*.*?\*/')
    for line in lines:
        new_line = comment_re.sub('', line)
        if new_line.strip():
            res.append(new_line)
    return res


def restore_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """Restore a single file. Returns (num_placeholders_replaced, num_annotations_removed_lines)."""
    lines = read_text(input_path)

    mdgpr_indices = find_mdgpr_root_indices(lines)
    placeholders_replaced = 0

    # For each MDGPR root, gather placeholder mapping from subsequent lines until next MDGPR or EOF
    for idx_i, start_idx in enumerate(mdgpr_indices):
        end_idx = mdgpr_indices[idx_i + 1] if idx_i + 1 < len(mdgpr_indices) else len(lines)

        # If no placeholders on the root line, skip
        if '#ID_' not in lines[start_idx]:
            continue

        mapping = collect_placeholder_mapping(lines, start_idx, end_idx)
        old_line = lines[start_idx]
        new_line = replace_placeholders_in_mdgpr_line(old_line, mapping)
        if new_line != old_line:
            placeholders_replaced += 1
            lines[start_idx] = new_line

    # Count annotation lines before stripping (lines containing /* */)
    annotation_lines = sum(1 for ln in lines if '/*' in ln and '*/' in ln)

    # Strip all annotations
    clean_lines = strip_annotations(lines)

    write_text(output_path, clean_lines)

    return placeholders_replaced, annotation_lines


def find_step_files(root: Path) -> List[Path]:
    exts = {'.step', '.stp', '.STEP', '.STP'}
    return [p for p in root.rglob('*') if p.suffix in exts and p.is_file()]


def main():
    parser = argparse.ArgumentParser(description='Restore valid STEP files from restructured STEP files.')
    parser.add_argument('path', help='File or directory to process')
    parser.add_argument('-o', '--output-dir', help='Output directory (default: ./restored_output)')

    args = parser.parse_args()

    src_path = Path(args.path)
    if not src_path.exists():
        print(f"Error: Path not found: {src_path}")
        exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path('./restored_output')

    files: List[Path]
    if src_path.is_file():
        files = [src_path]
        base_root = src_path.parent
    else:
        files = find_step_files(src_path)
        base_root = src_path

    if not files:
        print("No STEP files found to process.")
        return

    total = len(files)
    replaced_sum = 0
    annotations_sum = 0

    for f in files:
        # Mirror directory structure under output_dir and add _restored suffix
        rel = f.relative_to(base_root)
        out_file = output_dir / rel
        out_file = out_file.with_name(out_file.stem + '_restored' + out_file.suffix)

        replaced, ann = restore_file(f, out_file)
        replaced_sum += replaced
        annotations_sum += ann
        print(f"Restored: {f} -> {out_file} (mdgpr_lines:{replaced}, annotation_lines:{ann})")

    print(f"\nProcessed {total} files. MDGPR lines updated: {replaced_sum}. Annotation lines removed (lines containing /*...*/): {annotations_sum}.")


if __name__ == '__main__':
    main()
