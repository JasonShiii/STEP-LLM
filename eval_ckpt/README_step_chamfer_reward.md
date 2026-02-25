# STEP File Chamfer Distance and Reward Calculator

This script provides a comprehensive tool for computing chamfer distance between two STEP files and calculating a reward based on user-defined thresholds. It wraps the complete pipeline from STEP file processing to chamfer distance calculation with proper alignment.

## Features

- **Complete STEP Processing Pipeline**: STEP → STL → Point Cloud conversion
- **Deterministic Sampling**: Uses content-based seeding for reproducible point cloud generation
- **Robust Alignment**: Multi-stage alignment (center + global registration + ICP)
- **Scale Normalization**: Optional scale normalization for fair comparison across different object sizes
- **Flexible Reward System**: Configurable threshold-based reward calculation
- **Error Handling**: Graceful handling of invalid STEP files and conversion failures
- **Both Library and CLI**: Can be used as a Python library or command-line tool

## Requirements

### Environment
```bash
conda activate brepgen_env  # Contains pyocc and required libraries
```

### Dependencies
- OpenCASCADE (OCC.Core) - STEP file processing
- trimesh - STL processing and point cloud sampling
- open3d - Point cloud alignment
- chamferdist - Chamfer distance calculation
- torch - GPU acceleration (optional)
- numpy, plyfile - Data processing

## Installation

The script is ready to use in the `brepgen_env` conda environment. All required dependencies should be pre-installed.

## Usage

### Command Line Interface

```bash
python step_chamfer_reward.py step_file1.step step_file2.step --lower-bound 0.5 --upper-bound 2.0
```

#### Arguments
- `step_file1`, `step_file2`: Paths to STEP files to compare
- `--lower-bound`: Lower threshold for reward calculation (required)
- `--upper-bound`: Upper threshold for reward calculation (required)
- `--no-scale-normalize`: Disable scale normalization (default: enabled)
- `--use-gt-scale-only`: Use only first file's scale for normalization (recommended for GT vs Generated evaluation)
- `--verbose`, `-v`: Enable detailed output

#### Example
```bash
# For GT vs Generated evaluation (recommended)
python step_chamfer_reward.py ground_truth.step generated.step --lower-bound 0.3 --upper-bound 1.5 --use-gt-scale-only --verbose

# For general comparison
python step_chamfer_reward.py file1.step file2.step --lower-bound 0.3 --upper-bound 1.5 --verbose
```

### Python Library Usage

```python
from step_chamfer_reward import process_step_files

# Read STEP file contents
with open('file1.step', 'r') as f:
    content1 = f.read()
with open('file2.step', 'r') as f:
    content2 = f.read()

# Process files
chamfer_distance, reward = process_step_files(
    content1, content2,
    lower_bound=0.5,
    upper_bound=2.0,
    scale_normalize=True,
    use_gt_scale_only=True,  # Recommended for GT vs Generated evaluation
    verbose=False
)

print(f"Chamfer Distance: {chamfer_distance}")
print(f"Reward: {reward}")
```

## Pipeline Overview

### 1. STEP File Processing
- **Input**: Two STEP file contents as strings
- **STEP → STL**: Uses OpenCASCADE for reliable conversion
- **Error Handling**: Validates STEP file integrity and STL output

### 2. Point Cloud Generation
- **STL → Point Cloud**: Uses trimesh surface sampling (2000 points)
- **Deterministic Sampling**: Content-based seeding ensures reproducible results
- **Validation**: Checks for empty meshes and sampling failures

### 3. Point Cloud Alignment
Multi-stage alignment for rigid transformation invariance:

1. **Center Alignment**: Translates both clouds to common centroid
2. **Global Registration**: RANSAC-based feature matching for large rotations
3. **Iterative Closest Point (ICP)**: Fine-tunes alignment with progressive thresholds

### 4. Chamfer Distance Calculation
- **Scale Normalization**: Optional RMS-based scale normalization with two modes:
  - **Average Mode** (default): Uses average of both point clouds' scales
  - **GT-only Mode** (`--use-gt-scale-only`): Uses only first file's scale (recommended for GT vs Generated evaluation)
- **GPU Acceleration**: Automatic fallback to CPU if CUDA unavailable
- **Mean Reduction**: Returns average squared distance per point

### 5. Reward Calculation

The reward system provides a linear interpolation between thresholds:

```
if chamfer_distance > upper_bound:
    reward = 0.0
elif chamfer_distance <= lower_bound:
    reward = 1.0
else:
    reward = 1.0 - (chamfer_distance - lower_bound) / (upper_bound - lower_bound)
```

#### Reward Examples
With `lower_bound=0.5` and `upper_bound=2.0`:

| Chamfer Distance | Reward | Description |
|------------------|--------|-------------|
| 0.3 | 1.0 | Excellent match |
| 0.5 | 1.0 | At lower threshold |
| 1.0 | 0.67 | Good match |
| 1.5 | 0.33 | Acceptable match |
| 2.0 | 0.0 | At upper threshold |
| 2.5 | 0.0 | Poor match |

## Output Format

### Standard Output
```
Chamfer Distance: 1.234567
Reward: 0.567890
```

### Verbose Output
```
Using deterministic seed: 1234567
Converting STEP files to point clouds...
Generated point clouds: (2000, 3), (2000, 3)
Computing chamfer distance with alignment...
Chamfer Distance: 1.234567
Reward: 0.567890
```

### Error Cases
```
Chamfer Distance: FAILED
Reward: 0.000000
```

## Scale Normalization Options

The script provides two scale normalization strategies to handle different evaluation scenarios:

### 1. Average Scale Normalization (Default)
- **Formula**: `scale_factor = (scale1 + scale2) / 2`
- **Use Case**: General comparison between two objects of unknown relative importance
- **Behavior**: Both files contribute equally to the scale factor
- **Example**: Comparing two design alternatives

### 2. Ground Truth Scale Normalization (`--use-gt-scale-only`)
- **Formula**: `scale_factor = scale1` (first file only)
- **Use Case**: **Evaluating generated content against ground truth** (recommended)
- **Behavior**: Only the ground truth file's scale is used for normalization
- **Example**: Evaluating AI-generated CAD models against reference designs

### Why GT-only is Better for Evaluation

**Problem with averaging**: If GT is small (scale=1) and Generated is large (scale=100), the average scale factor (50.5) makes poor generations look artificially good.

**Example Scenario**:
```
Ground Truth: Small gear (RMS scale = 1.0)
Generated:    Large building (RMS scale = 100.0)

Average approach:  scale_factor = 50.5  → artificially low CD
GT-only approach:  scale_factor = 1.0   → realistic CD relative to GT
```

**Recommendation**: Always use `--use-gt-scale-only` when evaluating generated content against ground truth.

## Key Features

### Deterministic Sampling
- Uses MD5 hash of file contents to generate reproducible seeds
- Ensures consistent point cloud generation across runs
- Eliminates randomness in evaluation

### Robust Error Handling
- Graceful handling of invalid STEP files
- Automatic cleanup of temporary files
- Detailed error reporting in verbose mode

### Scale Invariance
- Optional scale normalization using RMS distance
- Fair comparison across different object sizes
- Consistent evaluation regardless of scale

### Performance Optimization
- GPU acceleration when available
- Efficient alignment algorithms
- Minimal memory footprint

## Testing

### Run Test Suite
```bash
python test_step_chamfer.py
```

### View Examples
```bash
python example_usage.py
```

## File Structure

```
eval_ckpt/
├── step_chamfer_reward.py          # Main script
├── test_step_chamfer.py            # Test suite
├── example_usage.py                # Usage examples
└── README_step_chamfer_reward.md   # This documentation
```

## Integration with Existing Evaluation Pipeline

This script integrates with the existing evaluation framework described in `README_eval.md`:

1. **Replaces**: Manual STEP → STL → Point Cloud conversion steps
2. **Maintains**: Compatibility with existing chamfer distance methodology
3. **Enhances**: Adds deterministic sampling and reward calculation
4. **Simplifies**: Single-command evaluation from STEP files

## Common Use Cases

### Model Evaluation
```bash
# Evaluate generated STEP against ground truth
python step_chamfer_reward.py gt_model.step generated_model.step --lower-bound 0.3 --upper-bound 1.5
```

### Batch Processing
```python
import os
from step_chamfer_reward import process_step_files

results = []
for gt_file, gen_file in file_pairs:
    with open(gt_file, 'r') as f1, open(gen_file, 'r') as f2:
        cd, reward = process_step_files(f1.read(), f2.read(), 0.5, 2.0)
        results.append((gt_file, gen_file, cd, reward))
```

### Hyperparameter Tuning
Test different threshold combinations to optimize reward distribution for your specific use case.

## Troubleshooting

### Common Issues

1. **"Cannot read STEP file"**: Check STEP file format and completeness
2. **"STL conversion failed"**: STEP geometry may be invalid or unsupported
3. **"Empty mesh"**: STEP file contains no valid geometric entities
4. **CUDA errors**: Script automatically falls back to CPU processing

### Environment Issues
Ensure you're using the correct conda environment:
```bash
conda activate brepgen_env
which python  # Should point to brepgen_env
```

### Memory Issues
For large batches, consider processing files sequentially rather than loading all into memory.

## Performance Notes

- **Typical Processing Time**: 10-30 seconds per file pair
- **Memory Usage**: ~500MB peak per process
- **GPU Acceleration**: 2-3x speedup when available
- **Deterministic**: Same inputs always produce same outputs

## Future Enhancements

- Multi-threading support for batch processing
- Additional alignment strategies
- More sophisticated reward functions
- Integration with MLflow for experiment tracking
