#!/usr/bin/env bash
# Batch-restructure STEP files using step_restructurer.py
#
# UPDATE the SRC_BASE and DEST_BASE paths below to match your local data layout.
# The RESTRUCTURER path is resolved relative to this script automatically.
#
# Usage:
#   bash data_preparation/batch_restructure.sh

set -euo pipefail

# ── Configurable paths ─────────────────────────────────────────────────────────
# UPDATE these to match your local data directory layout:
SRC_BASE="./dataset/rounded_step"   # source: rounded STEP files (per-model subdirs)
DEST_BASE="./dataset/dfs_step"      # destination: DFS-restructured STEP files
# ────────────────────────────────────────────────────────────────────────────────

# Resolve restructurer path relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESTRUCTURER="$SCRIPT_DIR/step_restructurer.py"

if [[ ! -f "$RESTRUCTURER" ]]; then
  echo "Error: restructurer not found at $RESTRUCTURER" >&2
  exit 1
fi

echo "Starting batch restructuring..."
echo "Source base: $SRC_BASE"
echo "Destination base: $DEST_BASE"

# Source layout: either numbered chunk buckets (SRC_BASE/0001 .. SRC_BASE/0010,
# the layout used for the paper) or per-model directories directly under
# SRC_BASE (the layout produced by round_step_numbers.py on a raw ABC chunk).
SRC_DIRS=()
for i in $(seq -w 0001 0010); do
  [[ -d "$SRC_BASE/$i" ]] && SRC_DIRS+=("$SRC_BASE/$i")
done
if [[ ${#SRC_DIRS[@]} -eq 0 ]]; then
  echo "No numbered chunk dirs (0001..0010) under $SRC_BASE — processing $SRC_BASE directly."
  SRC_DIRS=("$SRC_BASE")
fi

PROCESSED=0
for SRC_DIR in "${SRC_DIRS[@]}"; do
  echo "Scanning: $SRC_DIR"
  # Find .step and .stp files (case-insensitive)
  while IFS= read -r -d '' STEP_FILE; do
    # Determine destination directory mirroring source structure
    SRC_DIRNAME="$(dirname "$STEP_FILE")"
    REL_DIR="${SRC_DIRNAME#"$SRC_BASE"/}"
    DEST_DIR="$DEST_BASE/$REL_DIR"
    mkdir -p "$DEST_DIR"

    echo "Processing: $STEP_FILE"
    echo "  -> Output dir: $DEST_DIR"
    python3 "$RESTRUCTURER" "$STEP_FILE" -o "$DEST_DIR"
    PROCESSED=$((PROCESSED + 1))
  done < <(find "$SRC_DIR" -type f \( -iname '*.step' -o -iname '*.stp' \) -print0)
done

if [[ $PROCESSED -eq 0 ]]; then
  echo "Error: no .step/.stp files found under $SRC_BASE — nothing was restructured." >&2
  echo "Check SRC_BASE and that round_step_numbers.py ran first." >&2
  exit 1
fi
echo "Batch restructuring complete: $PROCESSED files. Output at: $DEST_BASE"
