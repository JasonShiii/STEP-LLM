import json
import os
import random
from datasets import load_dataset

# Load the dataset
with open('./data/rag_dataset.json', 'r') as file:
    data = json.load(file)

# Shuffle the data
random.shuffle(data)

# Calculate split indices
total_len = len(data)
train_len = int(total_len * 0.7)
test_len = int(total_len * 0.2)
val_len = total_len - train_len - test_len

# Split the data
train_data = data[:train_len]
test_data = data[train_len:train_len + test_len]
val_data = data[train_len + test_len:]

# Create output directory if it doesn't exist
output_dir = './data/abc_rag/train_val_test'
os.makedirs(output_dir, exist_ok=True)

# Save the split datasets
with open(os.path.join(output_dir, 'train.json'), 'w') as file:
    json.dump(train_data, file, indent=4)

with open(os.path.join(output_dir, 'test.json'), 'w') as file:
    json.dump(test_data, file, indent=4)

with open(os.path.join(output_dir, 'val.json'), 'w') as file:
    json.dump(val_data, file, indent=4)

print("Datasets split and saved successfully.")

# Load the split datasets to verify
train_dataset = load_dataset('json', data_files=os.path.join(output_dir, 'train.json'), split='train')
test_dataset = load_dataset('json', data_files=os.path.join(output_dir, 'test.json'), split='train')
val_dataset = load_dataset('json', data_files=os.path.join(output_dir, 'val.json'), split='train')

print("Train dataset loaded successfully:", len(train_dataset))
print("Test dataset loaded successfully:", len(test_dataset))
print("Validation dataset loaded successfully:", len(val_dataset))