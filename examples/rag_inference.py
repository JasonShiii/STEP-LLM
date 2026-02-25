#!/usr/bin/env python3
"""
RAG-enabled STEP file generation example.

This script demonstrates text-to-CAD generation with retrieval-augmented generation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from unsloth import FastLanguageModel
    import torch
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required packages not found.")
    print("Please install: pip install unsloth torch transformers sentence-transformers faiss-cpu")
    sys.exit(1)


class RAGRetriever:
    """Simple RAG retriever using FAISS."""
    
    def __init__(self, index_path: str, dataset_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize RAG retriever."""
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load FAISS index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"✓ Loaded FAISS index from {index_path}")
        else:
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        # Load dataset
        import json
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        print(f"✓ Loaded dataset with {len(self.dataset)} examples")
    
    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve top-k similar examples."""
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Retrieve examples
        retrieved_examples = []
        for idx in indices[0]:
            if idx < len(self.dataset):
                retrieved_examples.append(self.dataset[idx])
        
        return retrieved_examples


def generate_step_with_rag(
    prompt: str,
    model_path: str,
    retriever: RAGRetriever,
    top_k: int = 3,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    Generate a STEP file with RAG.
    
    Args:
        prompt: Natural language description of the CAD model
        model_path: Path to the trained model
        retriever: RAG retriever instance
        top_k: Number of examples to retrieve
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        Generated STEP file content as a string
    """
    print(f"Loading model from: {model_path}")
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    # Retrieve similar examples
    print(f"\nRetrieving {top_k} similar examples...")
    retrieved_examples = retriever.retrieve(prompt, top_k=top_k)
    
    # Format prompt with retrieved examples
    rag_context = "\n\n".join([
        f"Example {i+1}:\nDescription: {ex.get('caption', 'N/A')}\nSTEP: {ex.get('output', 'N/A')[:200]}..."
        for i, ex in enumerate(retrieved_examples)
    ])
    
    formatted_prompt = f"""### Instruction:
Generate a STEP file for the following CAD model description.
Use the provided examples as reference.

### Similar Examples:
{rag_context}

### Description:
{prompt}

### STEP File:
"""
    
    print("\nGenerating STEP file with RAG...")
    print(f"Prompt: {prompt}")
    print(f"Retrieved {len(retrieved_examples)} examples")
    print("-" * 50)
    
    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    step_content = generated_text[len(formatted_prompt):].strip()
    
    return step_content


def main():
    # Configuration
    MODEL_PATH = os.getenv("BASE_MODEL_PATH", "./merged_model")
    FAISS_INDEX = os.getenv("FAISS_INDEX_PATH", "./data/abc_rag/faiss_index.bin")
    DATASET_PATH = os.getenv("DATASET_PATH", "./data/abc_rag/train.json")
    
    # Check paths
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(FAISS_INDEX):
        print(f"Error: FAISS index not found at {FAISS_INDEX}")
        print("Please run the dataset processing script first:")
        print("  bash scripts/process_dataset.sh")
        sys.exit(1)
    
    print("=" * 50)
    print("RAG-Enabled STEP Generation Example")
    print("=" * 50)
    print()
    
    # Initialize retriever
    print("Initializing RAG retriever...")
    retriever = RAGRetriever(
        index_path=FAISS_INDEX,
        dataset_path=DATASET_PATH
    )
    print()
    
    # Get user input
    prompt = input("Enter your CAD model description: ").strip()
    
    if not prompt:
        print("Error: Empty prompt")
        sys.exit(1)
    
    # Generate with RAG
    try:
        step_content = generate_step_with_rag(
            prompt=prompt,
            model_path=MODEL_PATH,
            retriever=retriever,
            top_k=3,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9
        )
        
        # Save output
        output_dir = Path("./generated_outputs")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "generated_rag.step"
        with open(output_file, "w") as f:
            f.write(step_content)
        
        print("\n" + "=" * 50)
        print("✓ Generation complete!")
        print("=" * 50)
        print(f"\nSTEP file saved to: {output_file}")
        print(f"\nGenerated content preview:")
        print("-" * 50)
        print(step_content[:500] + "..." if len(step_content) > 500 else step_content)
        print("-" * 50)
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
