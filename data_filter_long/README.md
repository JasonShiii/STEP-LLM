# Token Statistics Analysis for ABC RAG Dataset

This script analyzes token statistics for the ABC RAG dataset using both Llama 3.2 and Qwen2.5 tokenizers.

## Features

- Analyzes token counts for three field combinations:
  - `caption + relevant_step_file`
  - `output`
  - `caption + relevant_step_file + output`
- Uses both Llama 3.2 and Qwen2.5 tokenizers for comparison
- Provides comprehensive statistics: min, max, average, median, standard deviation, 95th and 99th percentiles
- Generates comparison summary between the two tokenizers
- Progress bar and detailed logging
- Automatically generates output filename based on input path

## Requirements

```bash
pip install transformers numpy tqdm
```

## Usage

### Basic Usage

```bash
python analyze_token_stats.py /path/to/your/dataset.json
```

### Example

```bash
python analyze_token_stats.py /home/group/cad_codebased/data/abc_rag/20500_dfs/val.json
```

## Input Format

The script expects a JSON file with the following structure:
```json
[
  {
    "caption": "A rectangular block.",
    "relavant_step_file": "DATA;\n#1 = PRODUCT_CATEGORY(...);\n...",
    "output": "DATA;\n#1 = PRODUCT_CATEGORY(...);\n..."
  }
]
```

## Output

The script generates a text file named after the folder and filename of the input JSON.
For example, if the input is `/home/group/cad_codebased/data/abc_rag/20500_dfs/val.json`,
the output will be saved as `/home/group/cad_codebased/data_filter_long/token_stats/20500_dfs_val.txt`.

### Output Structure

The output file contains:

1. **Llama 3.2 Tokenizer Results**: Statistics for all three field combinations
2. **Qwen2.5 Tokenizer Results**: Statistics for all three field combinations  
3. **Summary Comparison**: Side-by-side comparison of 95th and 99th percentiles

For each field combination, the script calculates:
- **Count**: Number of samples processed
- **Min**: Minimum token count
- **Max**: Maximum token count
- **Average**: Mean token count
- **Median**: Median token count
- **Std**: Standard deviation
- **95th Percentile**: 95th percentile
- **99th Percentile**: 99th percentile

## Model Paths

The script expects the following model paths:
- **Llama 3.2**: `/home/group/cad_codebased/llama_3.2/Llama_3.2_3B`
- **Qwen2.5**: `/home/group/cad_codebased/Qwen2_5_3B`

If either model path is not found, the script will continue with the available tokenizer(s).

## Performance Notes

- Processing time depends on dataset size and text length
- The script includes progress bars and regular updates for large datasets
- Memory usage scales with dataset size
- Both tokenizers are loaded simultaneously for efficient processing

## Error Handling

The script includes comprehensive error handling for:
- Missing data files
- Invalid model paths
- Tokenization errors
- File I/O issues

## Example Output

```
TOKEN STATISTICS ANALYSIS
==================================================

LLAMA 3.2 TOKENIZER RESULTS:
------------------------------

CAPTION + RELEVANT + STEP:
  Count: 2,054
  Min: 1,247
  Max: 14,171
  Average: 5603.72
  Median: 5296.00
  Std: 2524.27
  95th Percentile: 9,725
  99th Percentile: 11,963

...

SUMMARY COMPARISON:
--------------------

CAPTION + RELEVANT + STEP:
  Llama 3.2 - 95th percentile: 9,725
  Qwen2.5 - 95th percentile: 13,155
  Llama 3.2 - 99th percentile: 11,963
  Qwen2.5 - 99th percentile: 16,610
```

## Use Cases

This script is particularly useful for:
- Understanding token distribution in your dataset
- Comparing tokenization efficiency between different models
- Planning model training with appropriate context lengths
- Identifying outliers in token counts
- Optimizing data preprocessing pipelines

## Troubleshooting

1. **Tokenizer loading errors**: Ensure the model paths are correct and accessible
2. **Memory issues**: For very large datasets, consider processing in batches
3. **File permission errors**: Ensure write permissions for the output directory
4. **Missing fields**: The script will use empty strings for missing fields




