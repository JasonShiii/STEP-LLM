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
SRC_BASE="./data/abccad/step_500-1000"   # source: raw STEP files from ABC dataset
DEST_BASE="./data/dfs_step_500-1000"     # destination: DFS-restructured STEP files
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

# Process directories 0001 to 0010
for i in $(seq -w 0001 0010); do
  SRC_DIR="$SRC_BASE/$i"
  if [[ ! -d "$SRC_DIR" ]]; then
    echo "Skip missing dir: $SRC_DIR"
    continue
  fi

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
  done < <(find "$SRC_DIR" -type f \( -iname '*.step' -o -iname '*.stp' \) -print0)
done

echo "Batch restructuring complete. Output at: $DEST_BASE"
