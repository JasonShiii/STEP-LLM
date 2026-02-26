# use conda env: cad_base
# the input csv file should not contain data that cannot be described
# pre-filtering is needed for embedding the captions


import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load captions from CSV file
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    assert 'model_id' in data.columns and 'description' in data.columns, "CSV must contain 'model_id' and 'description' columns."
    return data

# Step 2: Encode captions with Sentence-BERT
def encode_captions(data, model):
    descriptions = data['description'].tolist()
    embeddings = model.encode(descriptions, convert_to_tensor=False)  # Convert to numpy array
    return np.array(embeddings)

# Step 3: Build FAISS index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Size of embedding vector
    index = faiss.IndexFlatL2(dimension)  # L2 distance (can switch to other metrics like Inner Product)
    index.add(embeddings)  # Add embeddings to the index
    return index

# Step 4: Perform query search
def search_faiss(index, query_embedding, data, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        model_id = data.iloc[idx]['model_id']
        description = data.iloc[idx]['description']
        similarity = 1 / (1 + distances[0][i])  # Convert L2 distance to a similarity score
        results.append((model_id, description, similarity))
    return results

# Step 5: Compute additional similarity metrics
def compute_metrics(query_embedding, retrieved_embeddings):
    cosine_sim = cosine_similarity(query_embedding, retrieved_embeddings)
    return cosine_sim[0]

# Main function
def main():
    csv_file = './cad_captions_0-500.csv'  # Path to your captions CSV file
    query_caption = "A rectangular prism"  # Your query

    # Load data
    data = load_data(csv_file)

    # Load Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode database captions and query caption
    embeddings = encode_captions(data, model)
    query_embedding = model.encode([query_caption], convert_to_tensor=False)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Retrieve top-k results
    top_k = 5
    results = search_faiss(index, query_embedding, data, top_k=top_k)

    # Display results
    print("Top-k Results:")
    for i, (model_id, description, similarity) in enumerate(results):
        print(f"Rank {i+1}:")
        print(f"  Model ID: {model_id}")
        print(f"  Description: {description}")
        print(f"  Similarity Score: {similarity:.4f}\n")

    # Compute and display additional metrics (cosine similarity)
    retrieved_indices = [data.index[data['model_id'] == result[0]].tolist()[0] for result in results]
    retrieved_embeddings = embeddings[retrieved_indices]
    cosine_sim = compute_metrics(query_embedding, retrieved_embeddings)

    print("Cosine Similarity Metrics:")
    for i, sim in enumerate(cosine_sim):
        print(f"  Result {i+1}: Cosine Similarity = {sim:.4f}")

if __name__ == "__main__":
    main()
