# Evaluation of checkpoints

To evaluate the ckpts trained on new dataset, here're some steps to follow:

## 1. Filter the data according to token number
When the token number of a prompt exceed a certain amount, the `eval_loss` will get *nan* value.
Through experiment, the current threshold is set to **16455**

- For previous case that using STEP file & rendered image for captioning, the script `captioning_old.ipynb` has already calculated the token number and fill the value in `cad_captions.csv`
- While now we only use rendered image for captioning, the captioning prompt no longer contains STEP file. Thus, script `recalculate_token_count.py` recalculates the token number of data in `cad_captions.csv` and replace the *token_count* value.
- Script `debug_find_delete_longest_data.py` then processes the test dataset filters out the data whose token number exceeds the threshold, and records the filtered data in `debug_testset_deleted.csv`
- Run `eval_loss_by_ckpt.py` to calculate the *eval_loss* of different ckpts. `eval_loss_by_ckptn.py` are just copies of `eval_loss_by_ckpt.py` for parallel running. 


## 2. Compute *Complete_Ratio* metric
**eval_ckpt/CR**: Measure the capability of a model to generate “complete” step files (not stuck in repetition). Following is the calculation steps:
select a ckpt model
- inference on 400 data randomly selected from test dataset, and save the generated STEP files
- select other ckpt models to do inference on the same 400 data, save the generated files namely
- caculate the CR of different ckpts
  - CR = (#complete STEP file)/(#generated STEP file)
  - how to define “complete”: check whether the generated STEP file ends with “END-ISO-10303-21;”


## 3. Compute Shape and Geometric Metrics

Comprehensive Chamfer Distance evaluation with rigid transformation invariance and scale normalization for robust shape comparison.

### Overview

The shape evaluation pipeline computes Chamfer Distance (CD) between point clouds to measure geometric similarity. Key innovations include:

- **Rigid Transformation Invariance**: Identical shapes have consistent CD regardless of position/orientation
- **Scale Normalization**: Fair comparison across different object sizes
- **Deterministic Sampling**: Consistent point cloud generation eliminates random variation

### Evaluation Pipeline

1. **STEP → STL → Point Cloud Conversion**
2. **Multi-Stage Point Cloud Alignment**
3. **Chamfer Distance Calculation with Optional Scale Normalization**

### Step 1: Data Preparation

#### Environment Setup
```bash
conda activate brepgen_env
cd ./eval_ckpt/Shape/
```

#### Convert STEP files to STL format
```bash
jupyter notebook step_to_stl.ipynb
```

#### Generate Point Clouds from STL files
```bash
python sample_points.py --in_dir <STL_DIRECTORY> --out_dir <POINTCLOUD_DIRECTORY>
```

**Key Feature**: Uses **deterministic random seeding** based on filename to ensure consistent sampling across runs, eliminating variability from random point selection.

### Step 2: Rigid Transformation Analysis

#### Understanding Transformation Behavior

The `rigid_trans_step.py` script applies transformations with specific behavior:

| Transformation | Behavior | Critical Note |
|---------------|----------|---------------|
| **Translation** | Direct translation by vector | Moves object by exact amount |
| **Rotation** | Around **world origin (0,0,0)** | Objects not at origin will translate significantly |
| **Combined** | Rotation first, then translation | Net displacement can be much larger than expected |

**Example**: Object at `(0.3, -63.5, 0)` rotated 90° around Z-axis moves center to `(63.5, 0.3, 0)` - a 90+ unit displacement from a simple rotation!

#### Usage
```bash
python rigid_trans_step.py input.step output.step [tx ty tz] [rx ry rz angle_deg]
```

### Step 3: Chamfer Distance Calculation with Alignment

#### Basic Usage
```bash
python CD_cal_ICP_only.py path/to/pc1.ply path/to/pc2.ply
```

#### With Scale Normalization (Recommended)
```bash
python CD_cal_ICP_only.py path/to/pc1.ply path/to/pc2.ply --scale-normalize --scale-method rms_distance
```

#### Advanced Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--icp-threshold` | 10.0 | Distance threshold for ICP convergence |
| `--icp-iterations` | 3 | Number of ICP iterations with decreasing thresholds |
| `--disable-global-registration` | False | Disable RANSAC-based global registration |
| `--scale-normalize` | False | Enable scale normalization |
| `--scale-method` | `bbox_diagonal` | Scale normalization method |
| `--detailed-output` | False | Show alignment metrics and scale factors |

#### Multi-Stage Alignment Strategy

The alignment process uses a **3-stage approach** for robust rigid transformation invariance:

1. **Center Alignment** (O(n)): Translates both point clouds to common centroid
   - Handles large translations efficiently
   - Essential first step for all alignments

2. **Global Registration** (RANSAC + FPFH): Finds initial alignment using feature matching
   - Uses Fast Point Feature Histograms for robust feature description
   - Handles large rotations that ICP alone cannot resolve
   - Critical for rotated objects

3. **Iterative Closest Point (ICP)**: Fine-tunes alignment with progressive thresholds
   - 3 iterations with decreasing thresholds: 10.0 → 5.0 → 2.5
   - Coarse-to-fine alignment for optimal precision

#### Performance Results

Testing on various transformations shows dramatic improvement:

| Transformation | Raw CD | Aligned CD | Improvement |
|---------------|--------|------------|-------------|
| **Translation only** | ~1.5 | ~1.6 | Maintained excellence |
| **90° Z rotation** | 173.81 | 3.24 | **98% improvement** |
| **45° X rotation** | 190.74 | 3.06 | **98% improvement** |
| **60° Y rotation** | 7.20 | 3.07 | **52% improvement** |

### Step 4: Scale Normalization

#### Why Scale Normalization Matters

Raw CD values are scale-dependent - larger objects inherently have larger CD values. Scale normalization enables fair comparison across different object sizes.

#### Available Scale Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `bbox_diagonal` | Bounding box diagonal (default) | General purpose, robust |
| `avg_distance` | Average distance from centroid | Balanced, less sensitive to outliers |
| `rms_distance` | Root Mean Square distance | Mathematically principled |
| `max_distance` | Maximum distance from centroid | Sensitive to outliers |
| `volume_scale` | Cube root of bounding box volume | Volume-based scaling |

#### Scale Normalization Formula
```
Normalized_CD = Raw_CD / (Scale_Factor²)
```

#### Example: Scale Impact
```bash
# Without normalization - CD varies by 10,000x across scales
Raw CD: 12.34 (small object) vs 123,456 (large object)

# With normalization - CD remains consistent
Normalized CD: ~0.0012 (both objects)
```

#### Custom Configuration Example
```bash
# Fine-tuned parameters for specific cases
python CD_cal_ICP_only.py pc1.ply pc2.ply \
    --icp-threshold 5.0 \
    --icp-iterations 5 \
    --scale-normalize \
    --scale-method rms_distance \
    --detailed-output
```

### Key Technical Notes

#### Chamfer Distance Interpretation
- **Point Reduction**: Uses `mean` reduction (average squared distance per point)
- **Typical Values**: Well-aligned identical shapes should have CD ≤ 3.5
- **Scale Dependency**: Addressed through optional scale normalization

#### Transformation Invariance Validation
Pipeline validated on:
- Pure translations (various magnitudes)
- Rotations around different axes (X, Y, Z)
- Combined rotation + translation transformations
- Different object geometries and scales

#### Computational Efficiency
- **Center alignment**: O(n) - very fast
- **Global registration**: O(n²) - moderate cost, essential for large rotations
- **ICP iterations**: O(n²) per iteration - most expensive but provides precision


### Recommended Usage

For most evaluation scenarios, use:
```bash
python CD_cal_ICP_only.py pc1.ply pc2.ply --scale-normalize --scale-method rms_distance
```
### [IMPORTANT] Evaluate on Ground-Truth Dataset and Ckpt-Generated Dataset
```bash
cd ./eval_ckpt/Shape

./calculate_median_cd.sh <ground_truth_dir> <generated_dir> [output_file]
```
- The `output_file` should be under `cad_codebased/eval_ckpt/Shape/output_CD`
- The point cloud directory should be under `cad_codebased/eval_ckpt/Shape/pointcloud_eval`



## 4. Generate bacthes of STEP file of different ckpt for evaluation
- Switch to conda env `cad_llm3`. (*cad_llm3* is set up from a copy of *cad_llm2*.)
  
   Previously *cad_llm2* works for */cad_codebased/generate_step.py* where *faiss* was replaced by *sklearn* for cosine-similarity calculation due to env compatibility issues. *cad_llm3* debugged the issue of faiss and worked fine with */cad_codebased/eval_ckpt/generate_step_initial.py*

- Run `generate_step_initial.py` first on a designated ckpt to generate the first batch of STEP file.
- Run `generate_step_ckpt.py` later on the rest of ckpts to generate STEP files that have the same model_id of the first batch.

## 5. Batch Process the STEP Files to Point Clouds
Refer to `/home/group/cad_codebased/eval_ckpt/Shape/README_batch_conversion.md` to check the details of converting ckpt's generated STEP file to point clouds for ckpt evaluation.



## 6. Reserialize the STEP file
The non-sequential feature and cross-references make it hard for transformer to generate a STEP file.

Thus, we restructrue the step file to realize:
- **Eliminate forward references**: Ensure that each entity is defined before it is first referenced.
- **Similar Entity Clustering**: Under the premise of satisfying the dependency topology, try to cluster entities of the same type together as much as possible (optional).
- **renumber entity ID**: remap entity IDs starting with '#' to a continuous increasing sequence starting from '# 1'.

See: `/home/group/cad_codebased/reorder_step.py`



