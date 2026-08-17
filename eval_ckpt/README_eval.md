# Evaluation of checkpoints

To evaluate the ckpts trained on new dataset, here're some steps to follow:

> **What ships with this repository.** The evaluation code released here is:
>
> | Script | Metric |
> |---|---|
> | [`eval_loss_by_ckpt.py`](eval_loss_by_ckpt.py) | Validation loss per checkpoint |
> | [`CR/CR_calculate.py`](CR/CR_calculate.py) | Complete Ratio |
> | [`step_chamfer_reward.py`](step_chamfer_reward.py) | Chamfer distance between two STEP files |
> | [`renderability/check_renderability.py`](renderability/check_renderability.py) | Renderability |
> | [`ACE/step_entity_analyzer.py`](ACE/step_entity_analyzer.py) | STEP entity statistics |
>
> The point-cloud shape-evaluation pipeline described in section 3 (`eval_ckpt/Shape/`)
> and the batch generation helpers in section 4 are internal research scripts that
> are **not included** in this release. Those sections are kept as a description of
> the methodology used in the paper; `step_chamfer_reward.py` is the released,
> self-contained equivalent for Chamfer distance — see
> [`README_step_chamfer_reward.md`](README_step_chamfer_reward.md).

## 1. Filter the data according to token number
When the token number of a prompt exceed a certain amount, the `eval_loss` will get *nan* value.
Through experiment, the current threshold is set to **16455**

- For previous case that using STEP file & rendered image for captioning, the script `captioning_old.ipynb` has already calculated the token number and fill the value in `cad_captions.csv`
- While now we only use rendered image for captioning, the captioning prompt no longer contains STEP file. Thus, script `recalculate_token_count.py` recalculates the token number of data in `cad_captions.csv` and replace the *token_count* value.
- Script `debug_find_delete_longest_data.py` then processes the test dataset filters out the data whose token number exceeds the threshold, and records the filtered data in `debug_testset_deleted.csv`
- Run [`eval_loss_by_ckpt.py`](eval_loss_by_ckpt.py) to calculate the *eval_loss* of different ckpts. For parallel evaluation we ran several copies of this script side by side.

> `recalculate_token_count.py`, `debug_find_delete_longest_data.py` and
> `captioning_old.ipynb` are internal preprocessing helpers and are not included
> in this release; `eval_loss_by_ckpt.py` is.


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

> **Not included in this release.** The scripts in this section live in
> `eval_ckpt/Shape/` and are internal research code. The section documents the
> methodology used for the paper's shape metrics. For a released, self-contained
> Chamfer distance implementation that operates directly on two STEP files, use
> [`step_chamfer_reward.py`](step_chamfer_reward.py).

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
- The `output_file` should be under `eval_ckpt/Shape/output_CD`
- The point cloud directory should be under `eval_ckpt/Shape/pointcloud_eval`



## 4. Generate batches of STEP files from different ckpts for evaluation

To compare checkpoints fairly, every checkpoint must generate STEP files for the
*same* set of test `model_id`s:

- Generate the first batch with a designated checkpoint, recording the `model_id`
  of each generated file.
- Generate the remaining checkpoints' batches over that same `model_id` list.

Use [`generate_step.py`](../generate_step.py) at the repository root for
generation — note that RAG and no-RAG checkpoints require their matching
`--use_rag` setting, since they were trained with different prompt templates.
The internal batch drivers we used (`generate_step_initial.py`,
`generate_step_ckpt.py`) are not included in this release.

> Retrieval in `generate_step.py` uses scikit-learn cosine similarity rather
> than FAISS, which avoids an environment compatibility issue we hit with FAISS
> at inference time. FAISS is still used during dataset construction.

## 5. Batch Process the STEP Files to Point Clouds
Converting generated STEP files to point clouds is part of the internal
`eval_ckpt/Shape/` pipeline and is not included in this release. See section 3.



## 6. Reserialize the STEP file
The non-sequential feature and cross-references make it hard for transformer to generate a STEP file.

Thus, we restructrue the step file to realize:
- **Eliminate forward references**: Ensure that each entity is defined before it is first referenced.
- **Similar Entity Clustering**: Under the premise of satisfying the dependency topology, try to cluster entities of the same type together as much as possible (optional).
- **renumber entity ID**: remap entity IDs starting with '#' to a continuous increasing sequence starting from '# 1'.

See: `data_preparation/step_restructurer.py`



