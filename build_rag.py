import os
import glob
import warnings
import time
import pickle
import argparse

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Build a RAG system with custom parameters')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the documents to index')
    
    parser.add_argument('--model_path', type=str, 
                        default='llm_models/llama-2-7b-chat-hf-q4_k_m.gguf',
                        help='Path to the LLM model file')
    
    parser.add_argument('--save_dir', type=str, default='rag_cache',
                        help='Directory to save the RAG system files')
    
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Size of text chunks for indexing')
    
    parser.add_argument('--chunk_overlap', type=int, default=400,
                        help='Overlap between consecutive chunks')
    
    return parser.parse_args()

# === RAG Loader ===
def load_documents_with_metadata(folder_path, chunk_size=1000, chunk_overlap=400):
    chunks = []
    metadata = []
    total_bytes = 0

    for filepath in glob.glob(os.path.join(folder_path, '*')):
        filename = os.path.basename(filepath)
        total_bytes += os.path.getsize(filepath)

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            doc_meta = {}
            meta_lines = [line.strip() for line in lines[:5] if ':' in line]
            for line in meta_lines:
                key, value = line.split(':', 1)
                doc_meta[key.strip()] = value.strip()

            content = "".join(lines[5:])

            for i in range(0, len(content), chunk_size - chunk_overlap):
                chunk = content[i:i + chunk_size]
                chunks.append(chunk)
                metadata.append({
                    "file": filename,
                    "chunk_index": i,
                    "metadata": doc_meta
                })

    return chunks, metadata, total_bytes

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Access the parameters
    data_dir = args.data_dir
    llama_model_path = args.model_path
    SAVE_DIR = args.save_dir
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    
    # Set up environment
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.environ["LOG_LEVEL"] = "ERROR"
    warnings.filterwarnings("ignore")
    
    # === Build RAG ===
    print(f"📚 Loading documents from {data_dir} and creating vector store...")
    start_time = time.time()

    # Load and chunk
    chunks, metadata, total_bytes = load_documents_with_metadata(
        data_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    total_megabytes = round(total_bytes / (1024 * 1024), 2)
    
    if not chunks:
        print(f"⚠️ No documents found in {data_dir}. Exiting.")
        return

    # Embed using high-quality model
    print("🧠 Creating embeddings with SentenceTransformer...")
    embedder = SentenceTransformer('all-mpnet-base-v2')
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create FAISS Index (Inner Product = Cosine Similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index and metadata
    print(f"💾 Saving index and metadata to {SAVE_DIR}...")
    faiss.write_index(index, os.path.join(SAVE_DIR, "faiss.index"))
    with open(os.path.join(SAVE_DIR, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save embedding model information for future reference
    with open(os.path.join(SAVE_DIR, "model_info.txt"), 'w') as f:
        f.write('all-mpnet-base-v2')
        f.write(f'\nembedding_dimension: {dimension}')

    end_time = time.time()
    build_time = round(end_time - start_time, 2)

    print(f"✅ RAG built in {build_time} seconds")
    print(f"📊 Stats: {len(chunks)} chunks from {len(set(m['file'] for m in metadata))} files ({total_megabytes} MB)")
    print(f"💾 Index and metadata saved to {SAVE_DIR}")
    print(f"🔍 Index dimension: {dimension}")

if __name__ == "__main__":
    main()
    