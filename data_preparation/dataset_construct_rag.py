import os
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── File Paths — UPDATE THESE to match your local setup ─────────────────────
# Use cad_captions_0-500.csv for the DATE 2026 paper dataset (0–500 entities)
CSV_FILE = "./dataset/cad_captions_0-500.csv"
# UPDATE STEP_FILE_DIRS to point to your DFS-restructured STEP file directories.
# After running batch_restructure.sh, your files should be under dataset/dfs_step/.
# Example for the DATE paper (0–500 entity files, chunks 0001–0008):
STEP_FILE_DIRS = [
    "./dataset/dfs_step/0001",
    "./dataset/dfs_step/0002",
    "./dataset/dfs_step/0003",
    "./dataset/dfs_step/0004",
    "./dataset/dfs_step/0005",
    "./dataset/dfs_step/0006",
    "./dataset/dfs_step/0007",
    "./dataset/dfs_step/0008",
    # Add 500-1000 entity dirs here for the journal extension:
    # "./dataset/dfs_step_500-1000/0001",
    # ...
    # "./dataset/dfs_step_500-1000/0010",
]
OUTPUT_JSON_PATH = "./dataset/rag_dataset.json"

# Step 1: Load captions from CSV
def load_data(csv_file):
    data = pd.read_csv(csv_file, dtype={'model_id': str})
    assert {'model_id', 'isDescribable', 'description'}.issubset(data.columns), \
        "CSV must contain 'model_id', 'isDescribable', and 'description' columns."
    
    # Filter out non-describable entries
    data = data[data['isDescribable'] == True]
    
    # print(f"Loaded {len(data)} entries from {csv_file}.")
    return data.reset_index(drop=True)

# Step 2: Load STEP file DATA section
def load_step_data(model_id):
    for step_dir in STEP_FILE_DIRS:
        step_folder_path = os.path.join(step_dir, model_id)
        if not os.path.exists(step_folder_path):
            continue  # Check the next directory if folder doesn't exist
        
        # Find the step file in the folder
        step_file_path = None
        for file_name in os.listdir(step_folder_path):
            if file_name.endswith(".step"):
                step_file_path = os.path.join(step_folder_path, file_name)
                break
        
        if step_file_path and os.path.exists(step_file_path):
            with open(step_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Extract data section (everything after "DATA;")
            data_section = []
            start_collecting = False
            for line in lines:
                if "DATA;" in line:
                    start_collecting = True
                if start_collecting:
                    data_section.append(line.strip())
            # print(f"Loaded STEP file data for model_id {model_id} from {step_file_path}.")
            return "\n".join(data_section)
    
    return ""  # Return empty string if step file doesn't exist in any directory

# Step 3: Encode captions with Sentence-BERT
def encode_captions(data, model):
    descriptions = data['description'].tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=False)
    return np.array(embeddings)

# Step 4: Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(embeddings)
    return index

# Step 5: Perform FAISS search
def search_faiss(index, query_embedding, data, exclude_id, top_k=1):
    distances, indices = index.search(query_embedding, top_k + 1)  # Retrieve more to exclude self
    results = []
    for i, idx in enumerate(indices[0]):
        model_id = data.iloc[idx]['model_id']
        if model_id != exclude_id:  # Ensure we don't retrieve itself
            description = data.iloc[idx]['description']
            similarity = 1 / (1 + distances[0][i])  # Convert L2 distance to similarity
            results.append((model_id, description, similarity))
        if len(results) == top_k:
            break  # Stop once we get the required number of results
    return results

# Main function to create dataset
def create_rag_dataset():
    # Load and sample data
    data = load_data(CSV_FILE)
    
    # Load Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode captions
    embeddings = encode_captions(data, model)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    dataset = []

    for i, row in data.iterrows():
        model_id_original = row['model_id']
        caption = row['description']

        # Retrieve related caption using FAISS
        query_embedding = model.encode([caption], convert_to_tensor=False)
        retrieved_results = search_faiss(index, query_embedding, data, exclude_id=model_id_original, top_k=1)
        
        if retrieved_results:
            model_id_retrieve, _, _ = retrieved_results[0]
        else:
            model_id_retrieve = None  # In case of retrieval failure
        # Load STEP file DATA section
        output = load_step_data(model_id_original)
        relevant_step_file = load_step_data(model_id_retrieve) if model_id_retrieve else ""
        # Append to dataset
        dataset.append({
            "id_original": model_id_original,
            "caption": caption,
            "id_retrieve": model_id_retrieve if model_id_retrieve else "",
            "relavant_step_file": relevant_step_file,
            "output": output
        })

    # Save dataset to JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(dataset, json_file, indent=4)

    print(f"Dataset saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    create_rag_dataset()
