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
       │
       ▼
3. restore_step_valid.py      — strip annotations → valid, loadable STEP file
       │
       ▼
4. dataset_construct_rag.py   — pair each STEP file with a FAISS-retrieved similar
                                 example to build the RAG training dataset (JSON)
       │
       ▼
5. data_split.py              — split dataset JSON into train / val / test
       │
       ▼
   data/abc_rag/20500_dfs/{train,val,test}.json   ← ready for llama3_SFT_response.py
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
# Place chunks under data/abccad/
```

### 1. Normalise Floating-Point Numbers

```bash
# Process a single file
python data_preparation/round_step_numbers.py path/to/file.step --output-dir data/rounded_step/

# Process an entire directory
python data_preparation/round_step_numbers.py data/abccad/step_under500/ --output-dir data/rounded_step/
```

### 2. DFS Restructuring

```bash
# Single file
python data_preparation/step_restructurer.py path/to/file.step -o data/dfs_step/0001/

# Batch (all chunks, 0001–0010) — edit paths at the top of the script first
bash data_preparation/batch_restructure.sh
```

### 3. Restore Valid STEP Files (optional — for evaluation)

The restructured files contain `/* ... */` annotations that help the LLM but
make the files unloadable by CAD tools. Strip them when you need valid STEP
files for renderability / Chamfer-distance evaluation:

```bash
python data_preparation/restore_step_valid.py data/dfs_step/ -o data/dfs_step_valid/
```

### 4. Build RAG Dataset

Edit the path variables at the top of `dataset_construct_rag.py`:
- `CSV_FILE` — path to your captions CSV (`cad_captions_0-500.csv`)
- `STEP_FILE_DIRS` — list of directories containing DFS-restructured STEP files
- `OUTPUT_JSON_PATH` — where to save the output JSON

Then run:

```bash
python data_preparation/dataset_construct_rag.py
```

### 5. Train / Val / Test Split

Edit the `open(...)` path and `output_dir` variable at the top of `data_split.py`,
then run:

```bash
python data_preparation/data_split.py
```

This produces `train.json`, `test.json`, and `val.json` under the configured
output directory, ready for `llama3_SFT_response.py`.

## Rendered CAD Images

We provide rendered multi-view images of the ABC Dataset STEP files as a
separate dataset download (these are too large to include in the repository):

| Split | Entity range | Location |
|---|---|---|
| Under-500 images | 0–500 entities | `data/abccad/step_under500_image/` |
| 500–1000 images  | 500–1000 entities | `data/abccad/step_500-1000_image/` |

These images are used by `captioning.ipynb` to generate GPT-4o captions.
Download links will be provided on the [HuggingFace dataset page](https://huggingface.co/datasets/JasonShiii/STEP-LLM-dataset).

## Notes

- **Hardcoded paths:** `dataset_construct_rag.py` and `data_split.py` contain
  absolute paths at the top of the file. Update these to your local paths
  before running.
- **Entity count filtering:** The DATE paper uses STEP files with 0–500
  entities. Files with 500–1000 entities are used in ongoing journal work.
  `step_restructurer.py` processes any STEP file; filtering is done at the
  caption CSV level (`isDescribable` column).
- **Token length:** After DFS restructuring, a 500-entity STEP file is roughly
  4k–8k tokens. The 16384 token context window is the limiting factor.
