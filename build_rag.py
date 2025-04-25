import os
import glob
import warnings
import time
import pickle

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === Config ===
data_dir = "/Users/vuk/projects/ask_david_clean/data/DavidSnyder"
llama_model_path = "/Users/vuk/projects/llm_models/llama-2-7b-chat-hf-q4_k_m.gguf"
SAVE_DIR = "rag_cache"

os.makedirs(SAVE_DIR, exist_ok=True)
os.environ["LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

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

# === Build RAG ===
print("📚 Loading documents and creating vector store...")
start_time = time.time()

# Load and chunk
chunks, metadata, total_bytes = load_documents_with_metadata(data_dir)
total_megabytes = round(total_bytes / (1024 * 1024), 2)

# Embed using high-quality model
embedder = SentenceTransformer('all-mpnet-base-v2')
embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# Normalize for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Create FAISS Index (Inner Product = Cosine Similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, os.path.join(SAVE_DIR, "faiss.index"))
with open(os.path.join(SAVE_DIR, "metadata.pkl"), 'wb') as f:
    pickle.dump(metadata, f)

end_time = time.time()
build_time = round(end_time - start_time, 2)

print(f"✅ RAG built in {build_time} seconds using {total_megabytes} MB of text data")
print(f"💾 Index and metadata saved to {SAVE_DIR}")


