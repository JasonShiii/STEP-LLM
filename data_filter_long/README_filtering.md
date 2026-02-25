# Token Threshold Filtering Script

This script filters the ABC RAG dataset by removing entries whose specified field exceeds a token threshold, using either Llama 3.2 or Qwen2.5 tokenizers.

## Features

- Filters dataset entries based on token count thresholds
- Supports three field types for checking:
  - `caption-step`: Caption + relevant step file content
  - `output`: Output field content only
  - `whole-data`: Caption + relevant step file + output content
- Uses either Llama 3.2 or Qwen2.5 tokenizer
- **Directly modifies the original JSON file** (creates backup automatically)
- Saves deleted items to JSON file in `filtered_datasets/` directory
- Saves deleted IDs to CSV file in `deleted_list/` directory
- Provides comprehensive filtering statistics

## Requirements

```bash
pip install transformers numpy tqdm
```

## Usage

### Basic Usage

```bash
python filter_by_token_threshold.py <data_file> <tokenizer_type> <threshold> <field_type>
```

### Parameters

- `data_file`: Path to the JSON dataset file
- `tokenizer_type`: Choose between `llama` or `qwen`
- `threshold`: Token count threshold (entries exceeding this will be deleted)
- `field_type`: Field to check: `caption-step`, `output`, or `whole-data`

### Examples

```bash
# Filter by caption-step field using Llama tokenizer with 8000 token threshold
python filter_by_token_threshold.py /path/to/dataset.json llama 8000 "caption-step"

# Filter by output field using Qwen tokenizer with 12000 token threshold
python filter_by_token_threshold.py /path/to/dataset.json qwen 12000 "output"

# Filter by whole-data field using Llama tokenizer with 20000 token threshold
python filter_by_token_threshold.py /path/to/dataset.json llama 20000 "whole-data"
```

## Input Format

The script expects a JSON file with the following structure:
```json
[
  {
    "id_original": "00012345",
    "caption": "A rectangular block.",
    "relavant_step_file": "DATA;\n#1 = PRODUCT_CATEGORY(...);\n...",
    "output": "DATA;\n#1 = PRODUCT_CATEGORY(...);\n..."
  }
]
```

## Output Files

The script generates two output files and modifies the original dataset:

### 1. Deleted Items JSON
- **Location**: `/home/group/cad_codebased/data_filter_long/filtered_datasets/`
- **Naming**: `{folder}_{filename}_deleted_{tokenizer}_{threshold}.json`
- **Content**: Complete deleted data entries that exceeded the token threshold

### 2. Deleted IDs List
- **Location**: `/home/group/cad_codebased/data_filter_long/deleted_list/`
- **Naming**: `{folder}_{filename}_deleted_{tokenizer}_{threshold}.csv`
- **Content**: CSV file with deleted entry information

### 3. Original Dataset (Modified)
- **Action**: **Directly updated** with filtered data (entries exceeding threshold are removed)
- **Backup**: Automatic backup created as `{original_file}.backup` before modification

### CSV Structure
```csv
id_original,token_count,threshold,field_type
00012345,8500,8000,caption-step
00067890,9200,8000,caption-step
```

## Field Types

### caption-step
Combines the `caption` and `relavant_step_file` fields for token counting.

### output
Uses only the `output` field for token counting.

### whole-data
Combines all three fields: `caption` + `relavant_step_file` + `output` for token counting.

## Model Paths

The script expects the following model paths:
- **Llama 3.2**: `/home/group/cad_codebased/llama_3.2/Llama_3.2_3B`
- **Qwen2.5**: `/home/group/cad_codebased/Qwen2_5_3B`

## Example Output

```
Processing data file: /path/to/dataset.json
Using LLAMA tokenizer from: /path/to/llama/model
Field type: caption-step
Token threshold: 8,000
Dataset contains 2,054 samples
Processing samples: 100%|███████████████████████████████████████████████| 2054/2054 [00:20<00:00, 98.24it/s]

Deleted items saved to: /path/to/deleted_items.json
Deleted IDs saved to: /path/to/deleted_ids.csv
Original file backed up to: /path/to/dataset.json.backup
Original file updated with filtered data: /path/to/dataset.json

============================================================
FILTERING SUMMARY
============================================================
Original dataset size: 2,054
Filtered dataset size: 2,047
Deleted entries: 7
Retention rate: 99.66%
Deletion rate: 0.34%

Token count statistics:
  Min: 1,247
  Max: 14,171
  Mean: 5603.72
  Median: 5296.00
  Std: 2524.27
  Threshold: 8,000
  Entries above threshold: 7 (0.34%)

Processing completed in 21.19 seconds
Dataset filtering completed successfully!
```

## Use Cases

This script is particularly useful for:
- **Model Training**: Removing overly long sequences that exceed model context limits
- **Data Quality**: Filtering out extremely long entries that may be outliers
- **Memory Optimization**: Reducing memory usage during training by limiting sequence lengths
- **Performance Tuning**: Finding optimal token thresholds for your specific use case

## Performance Notes

- Processing time depends on dataset size and text length
- The script includes progress bars and regular updates
- Memory usage scales with dataset size
- Tokenization is performed efficiently using the specified tokenizer

## Error Handling

The script includes comprehensive error handling for:
- Missing data files
- Invalid model paths
- Tokenization errors
- File I/O issues
- Invalid field type specifications

## Tips

1. **Start with conservative thresholds**: Begin with higher thresholds and gradually reduce them
2. **Monitor retention rates**: Aim for retention rates above 90% unless you have specific requirements
3. **Consider field type carefully**: Different field types may have very different token distributions
4. **Test on small datasets first**: Verify your parameters on validation sets before processing large datasets
5. **Backup original data**: Always keep a backup of your original dataset before filtering

## Troubleshooting

1. **Tokenizer loading errors**: Ensure the model paths are correct and accessible
2. **Memory issues**: For very large datasets, consider processing in batches
3. **File permission errors**: Ensure write permissions for the output directories
4. **Invalid field types**: Double-check the field type parameter spelling
