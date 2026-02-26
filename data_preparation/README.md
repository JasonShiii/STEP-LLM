# Data Preparation

This directory contains all scripts for converting raw ABC Dataset STEP files
into the training-ready RAG dataset used by STEP-LLM.

## Pipeline Overview

```
Raw ABC STEP files
       │
       ▼
1. round_step_numbers.py      — normalise floating-point precision (6 d.p.)
       │
       ▼
2. step_restructurer.py       — DFS reorder + structural annotations
   (batch_restructure.sh)       eliminates forward references; adds /* ... */ markers
       │                        (annotations are valid STEP comments — files are
       │                         training-ready and loadable by CAD tools as-is)
       ▼
3. dataset_construct_rag.py   — pair each STEP file with a FAISS-retrieved similar
                                 example to build the RAG training dataset (JSON)
       │
       ▼
4. data_split.py              — split dataset JSON into train / val / test
       │
       ▼
   dataset/abc_rag/train_val_test/{train,val,test}.json   ← ready for llama3_SFT_response.py

[Optional] restore_step_valid.py — strips /* ... */ annotations and renumbers
   entity IDs. Only needed if a specific downstream tool cannot handle
   STEP comments, or to post-process model-generated output.
```

## Scripts

| Script | Description |
|---|---|
| `round_step_numbers.py` | Rounds all floating-point numbers in STEP entity lines to 6 decimal places to reduce token count variability. |
| `step_restructurer.py` | Core DFS restructuring. Parses entity reference graph, traverses from root entities, emits entities in dependency order, inserts structural annotations. |
| `batch_restructure.sh` | Bash wrapper to batch-process a directory of STEP files with `step_restructurer.py`. Edit `SRC_BASE` / `DEST_BASE` at the top. |
| `restore_step_valid.py` | Post-processing: replaces `#ID_n` placeholders with real entity IDs and strips all `/* ... */` annotations to produce valid STEP files. |
| `dataset_construct_rag.py` | Builds the RAG training dataset. Uses FAISS + sentence-transformers to retrieve the most similar STEP file for each caption. Outputs a JSON file. |
| `data_split.py` | Splits the RAG dataset JSON into train (70%) / test (20%) / val (10%) splits. |
| `captioning.ipynb` | Jupyter notebook for generating GPT-4o captions from STEP file thumbnails. |

## Step-by-Step Instructions

### 0. Prerequisites

```bash
# Download ABC dataset chunks (see scripts/download_abc_dataset.sh)
bash scripts/download_abc_dataset.sh
# Chunks are placed under dataset/abccad/
```

### 1. Normalise Floating-Point Numbers

```bash
# Process a single file
python data_preparation/round_step_numbers.py path/to/file.step --output-dir dataset/rounded_step/

# Process an entire directory
python data_preparation/round_step_numbers.py dataset/abccad/step_under500/ --output-dir dataset/rounded_step/
```

### 2. DFS Restructuring

```bash
# Single file
python data_preparation/step_restructurer.py path/to/file.step -o dataset/dfs_step/0001/

# Batch (all chunks, 0001–0010) — edit SRC_BASE / DEST_BASE at the top of the script first
bash data_preparation/batch_restructure.sh
```

### 3. Build RAG Dataset

Path defaults in `dataset_construct_rag.py` already point to `dataset/`:
- `CSV_FILE` → `./dataset/cad_captions_0-500.csv`
- `STEP_FILE_DIRS` → `./dataset/dfs_step/0001` … `0008`
- `OUTPUT_JSON_PATH` → `./dataset/rag_dataset.json`

Run from the repo root:

```bash
python data_preparation/dataset_construct_rag.py
```

### 4. Train / Val / Test Split

Default paths in `data_split.py` read from `./dataset/rag_dataset.json` and
write splits to `./dataset/abc_rag/train_val_test/`. Run from the repo root:

```bash
python data_preparation/data_split.py
```

This produces `train.json`, `test.json`, and `val.json` under
`dataset/abc_rag/train_val_test/`, ready for `llama3_SFT_response.py`.

### [Optional] Strip Annotations

The `/* ... */` markers inserted by `step_restructurer.py` are valid STEP
comments — the restructured files are directly usable for training **and** are
loadable by CAD tools without any extra processing. Stripping is only needed
if a specific tool explicitly rejects comments:

```bash
python data_preparation/restore_step_valid.py dataset/dfs_step/ -o dataset/dfs_step_clean/
```

## Rendered CAD Images

We provide rendered multi-view images of the ABC Dataset STEP files as a
separate dataset download (these are too large to include in the repository):

| Split | Entity range | Location |
|---|---|---|
| Under-500 images | 0–500 entities | `dataset/rendered_images/step_under500_image/` |
| 500–1000 images  | 500–1000 entities | `dataset/rendered_images/step_500-1000_image/` |

These images are used by `captioning.ipynb` to generate GPT-4o captions.
Download from the [HuggingFace dataset page](https://huggingface.co/datasets/JasonShiii/STEP-LLM-dataset).

## Notes

- **Default paths:** All scripts default to paths under `dataset/` and can be
  run from the repo root without editing. Override `CSV_FILE`, `STEP_FILE_DIRS`,
  etc. at the top of each script if your layout differs.
- **Entity count filtering:** The DATE paper uses STEP files with 0–500
  entities. Files with 500–1000 entities are used in ongoing journal work.
  `step_restructurer.py` processes any STEP file; filtering is done at the
  caption CSV level (`isDescribable` column).
- **Token length:** After DFS restructuring, a 500-entity STEP file is roughly
  4k–8k tokens. The 16384 token context window is the limiting factor.
