# Dataset Documentation

## Overview

This project uses the ABC (A Big CAD Model Dataset) as the base dataset, augmented with automatically generated natural language captions for text-to-CAD generation tasks.

## Dataset Components

### 1. Base Dataset: ABC CAD

**Source**: NYU Dataset Archive  
**Link**: https://archive.nyu.edu/handle/2451/43778  
**License**: See ABC dataset license terms  
**Size**: ~1 million CAD models in STEP format

The ABC dataset contains high-quality CAD models from various domains including mechanical parts, consumer products, and architectural elements.

### 2. Generated Captions

We provide natural language captions generated for a subset of the ABC dataset using vision-language models.

**Format**: CSV file with columns:
- `model_id`: ABC model identifier
- `description`: Natural language description of the CAD model
- `isDescribable`: Boolean indicating if the model could be described

**Download**: [Link to your release - update this]

### 3. Processed Dataset

After processing, the dataset is structured as JSON files with the following format:

```json
{
  "id_original": "00040014_dfb67f543711434fb5751d3f_step_003",
  "caption": "A mechanical bracket with two mounting holes",
  "relevant_step_file": "path/to/reordered_step_file.step",
  "output": "ISO-10303-21;\nHEADER;\n..."
}
```

## Dataset Statistics

- **Total models**: ~XX,XXX (update with your numbers)
- **Train split**: XX,XXX (70%)
- **Validation split**: X,XXX (20%)
- **Test split**: X,XXX (10%)
- **Average caption length**: ~XX tokens
- **Average STEP file length**: ~XXX tokens
- **Token distribution**: See `data_filter_long/README.md` for detailed statistics

## Data Processing Pipeline

1. **Download ABC Dataset**: Original STEP files from NYU archive
2. **Caption Generation**: Use vision models to generate descriptions
3. **STEP Reordering**: Normalize entity ordering for consistency
4. **RAG Dataset Construction**: Combine captions with STEP files
5. **Dataset Splitting**: Create train/val/test splits
6. **Token Filtering**: Remove extremely long/short examples
7. **Embedding Generation**: Create FAISS index for RAG retrieval

See `scripts/process_dataset.sh` for the complete pipeline.

## Usage

### Download and Setup

```bash
# 1. Download ABC dataset
bash scripts/download_abc_dataset.sh

# 2. Download our captions
# Download from [YOUR_RELEASE_LINK] and place in data/

# 3. Process dataset
bash scripts/process_dataset.sh
```

### Loading the Dataset

```python
import json

# Load training data
with open('dataset/abc_rag/train_val_test/train.json', 'r') as f:
    train_data = json.load(f)

# Access example
example = train_data[0]
print(f"Caption: {example['caption']}")
print(f"STEP file: {example['output'][:200]}...")
print(f"Retrieved neighbour: {example['relavant_step_file'][:200]}...")  # RAG field
```

## Data Format Details

### Caption Format

Captions describe the geometric and functional characteristics of CAD models:
- Geometric primitives (cubes, cylinders, etc.)
- Dimensions and measurements
- Functional elements (holes, grooves, etc.)
- Assembly relationships
- Material or appearance properties (if applicable)

Example captions:
- "A rectangular mounting bracket with four circular holes at the corners"
- "A cylindrical shaft with threaded ends and a center groove"
- "An L-shaped connector piece with rounded edges"

### STEP File Format

STEP files follow the ISO 10303-21 standard. Our processing includes:
- Entity reordering for consistency
- Renumbering for sequential IDs
- Validation of single-part files
- Removal of malformed files

## Token Length Filtering

We provide filtered datasets based on token lengths to accommodate different model capacities:

- **Under 500 tokens**: Small models suitable for quick iteration
- **Under 1000 tokens**: Medium complexity models
- **Under 2000 tokens**: Full dataset for production training
- **Custom thresholds**: Use `data_filter_long/filter_by_token_threshold.py`

See `data_filter_long/README_filtering.md` for details.

## Evaluation Datasets

- **Validation set**: Used during training for hyperparameter tuning
- **Test set**: Held-out set for final evaluation
- **In-distribution**: Models similar to training data
- **Out-of-distribution**: Novel geometries for generalization testing

## Data Quality

### Quality Control Measures

1. **Caption Quality**
   - Generated using GPT-4 Vision
   - Manual inspection of sample
   - Filtering of non-describable models

2. **STEP File Quality**
   - Validation with OpenCASCADE
   - Single-part file verification
   - Geometric integrity checks
   - Removal of degenerate models

3. **Duplicate Removal**
   - Exact duplicate detection
   - Near-duplicate filtering based on embeddings

## License and Attribution

- **ABC Dataset**: retains its original terms — see the
  [ABC dataset page](https://archive.nyu.edu/handle/2451/43778).
- **Generated captions and processed data**: released under this repository's
  [MIT LICENSE](../LICENSE). They are derived from the ABC dataset, so ABC's
  terms continue to apply to the underlying CAD geometry.

### Citation

If you use this dataset, please cite the STEP-LLM paper and the ABC dataset:

```bibtex
@inproceedings{shi2026stepllm,
  title={STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models},
  author={Shi, Xiangyu and Ding, Junyang and Zhao, Xu and Zhan, Sinong and Mohapatra, Payal and Quispe, Daniel and Welbeck, Kojo and Cao, Jian and Chen, Wei and Guo, Ping and Zhu, Qi},
  booktitle={Proceedings of the 2026 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  year={2026},
  organization={IEEE}
}

@inproceedings{koch2019abc,
  title={ABC: A Big CAD Model Dataset For Geometric Deep Learning},
  author={Koch, Sebastian and Matveev, Albert and Jiang, Zhongshi and Williams, Francis and Artemov, Alexey and Burnaev, Evgeny and Alexa, Marc and Zorin, Denis and Panozzo, Daniele},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## Known Limitations

1. **Caption Subjectivity**: Captions are generated automatically and may not capture all details
2. **Model Complexity**: Very complex assemblies may be simplified in descriptions
3. **Geometric Precision**: Token-based representation may lose precision in dimensions
4. **Domain Coverage**: Biased toward mechanical parts and common shapes

## Future Work

- Additional caption refinement
- Multi-view image integration
- Parametric information extraction
- Assembly hierarchy annotations
- Material and appearance labels

## Support

For dataset-related issues:
- Open an issue on [GitHub](https://github.com/JasonShiii/STEP-LLM/issues)
- See the data preparation walkthrough in [`../data_preparation/README.md`](../data_preparation/README.md)
