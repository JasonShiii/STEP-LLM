# STEP File Renderability Checker

This tool checks whether STEP files can be successfully rendered (read, meshed, and exported to STL).

## Definition of "Renderable"

A STEP file is considered "renderable" if:
- The STEP file can be read successfully by OpenCASCADE.
- The resulting shape is non-null.
- The shape can be meshed (triangulated) and exported to a non-empty STL file.
- Optionally (if `trimesh` is installed): the exported STL loads with non-zero face count.

### Why Subprocess Isolation?
Malformatted STEP files can cause OCC to crash the hosting process. To prevent batch processing from stopping, the heavy OCC operations are executed in a short-lived subprocess with a timeout.

**Batch Processing Features:**
- Dynamic progress bar showing current file and completion percentage
- Hierarchical statistics (overall → subdirectory level)
- Smart directory filtering (STEP_generated → only 'gen' folders)
- Focused failure reporting (lists non-renderable files with reasons)

### Script
- `check_renderability.py`: Runs the renderability test and prints a single JSON object to stdout.

### Usage

**Single file mode:**
```bash
python check_renderability.py /absolute/path/to/file.step [options]
```

**Batch mode:**
```bash
python check_renderability.py --batch /path/to/directory [options]
```

### Options
- `--batch DIRECTORY`: Batch process all STEP files in directory recursively
- `--output FILE`: Output file for batch report (default: auto-generated)
- `--timeout N`: Timeout in seconds (default: 30)
- `--require-all-roots`: Check every ADVANCED_BREP_SHAPE_REPRESENTATION root; fail if any root fails to mesh/export
- `--verify-trimesh-faces`: Require exported STL to load in trimesh with non-zero face count (automatically enabled in batch mode)
- `--treat-reader-errors-as-failure`: Treat OCC StepReaderData error messages as failure (otherwise they're reported but don't fail the check)

### Examples

**Single file basic check:**
```bash
python check_renderability.py /path/to/file.step
```

**Single file strict validation:**
```bash
python check_renderability.py /path/to/file.step --require-all-roots --verify-trimesh-faces
```

**Batch check dfs_step directory (checks all STEP files):**
```bash
python check_renderability.py --batch ./data/dfs_step
```

**Batch check STEP_generated directory (only checks 'gen' subfolders):**
```bash
python check_renderability.py --batch ./data/STEP_generated/eval_output
```

**Batch check with custom output file:**
```bash
python check_renderability.py --batch ./data/dfs_step --output step_analysis.json
```

**Batch check with custom timeout:**
```bash
python check_renderability.py --batch /path/to/directory --timeout 60
```

### Output Fields Explained

| Field | Description |
|-------|-------------|
| `renderable` | True if file can be rendered |
| `status` | "ok", "error", or "timeout" |
| `reason` | Error details (null on success) |
| `num_roots_expected` | Number of ADVANCED_BREP_SHAPE_REPRESENTATION found in file |
| `num_roots_found` | Number of roots OCC reader can transfer |
| `roots` | Per-root details (only with --require-all-roots) |
| `stl_ok` | Whether STL export succeeded |
| `trimesh_verified` | Whether trimesh validation was requested |
| `trimesh_faces` | Face count from trimesh (null if not verified) |
| `reader_errors_present` | Whether OCC printed StepReaderData warnings |
| `reader_error_lines` | Array of OCC warning messages |
| `duration_seconds` | Time taken for the check |

### Sample Output

**Single file basic check:**
```json
{
  "renderable": true,
  "status": "ok", 
  "reason": null,
  "num_roots_expected": 2,
  "num_roots_found": 2,
  "roots": [],
  "stl_ok": true,
  "trimesh_verified": false,
  "trimesh_faces": null,
  "duration_seconds": 0.25,
  "reader_errors_present": true,
  "reader_error_lines": ["*** ERR StepReaderData *** Pour Entite #13"]
}
```

**Single file with --require-all-roots --verify-trimesh-faces:**
```json
{
  "renderable": true,
  "status": "ok",
  "num_roots_expected": 2,
  "num_roots_found": 2,
  "roots": [
    {
      "root_index": 1,
      "transfer_ok": true,
      "shape_null": false,
      "stl_ok": true,
      "trimesh_verified": true,
      "trimesh_faces": 142,
      "ok": true
    },
    {
      "root_index": 2,
      "transfer_ok": true,
      "shape_null": false,
      "stl_ok": true,
      "trimesh_verified": true,
      "trimesh_faces": 156,
      "ok": true
    }
  ],
  "duration_seconds": 0.84,
  "reader_errors_present": true,
  "reader_error_lines": ["*** ERR StepReaderData *** Pour Entite #13"]
}
```

**Batch mode report structure:**
```json
{
  "batch_info": {
    "input_directory": "./data/dfs_step",
    "start_time": "2025-08-09T18:20:09.160058",
    "end_time": "2025-08-09T18:20:10.818536",
    "total_duration_minutes": 0.028,
    "files_processed": 152,
    "settings": {
      "timeout": 30,
      "verify_trimesh_faces": true
    }
  },
  "summary_statistics": {
    "total_files": 152,
    "renderable": 128,
    "not_renderable": 24,
    "timeout": 2,
    "error": 22,
    "with_reader_errors": 45,
    "avg_duration": 0.83,
    "failure_reasons": {
      "read_failed": 12,
      "null_shape": 8,
      "stl_empty_or_export_failed": 4
    },
    "trimesh_stats": {
      "total_with_faces": 120,
      "total_verified": 128,
      "min_faces": 24,
      "max_faces": 1842,
      "avg_faces": 287.5
    }
  },
  "subdirectory_statistics": {
    "0001": {
      "total": 76,
      "renderable": 68,
      "not_renderable": 8,
      "with_trimesh_faces": 60,
      "failed_files": [
        {
          "file_path": "./data/dfs_step/0001/00010007/file.step",
          "relative_path": "0001/00010007/file.step",
          "reason": "read_failed",
          "status": "error"
        }
      ]
    },
    "0002": {
      "total": 76,
      "renderable": 60,
      "not_renderable": 16,
      "with_trimesh_faces": 55,
      "failed_files": [...]
    }
  }
}
```

### Exit Codes
- `0`: renderable
- `2`: not renderable (read/mesh/export failed or empty geometry)
- `3`: timeout exceeded
- `4`: internal error

### Dependencies
- `python-opencascade` (OCC): For reading STEP files and meshing
- `trimesh` (optional): For additional STL validation

### Installation
```bash
# Install OCC (example for conda)
conda install -c conda-forge python-opencascade

# Optional: Install trimesh for enhanced validation
pip install trimesh
```