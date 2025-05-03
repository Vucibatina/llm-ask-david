import os
import warnings
import time
import pickle
import argparse
import concurrent.futures

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# Embedding model configurations
EMBEDDING_MODELS = {
    'mini': {
        'name': 'all-MiniLM-L6-v2',
        'dimension': 384,
        'description': 'Faster, smaller embeddings (384 dimensions)'
    },
    'mpnet': {
        'name': 'all-mpnet-base-v2',
        'dimension': 768,
        'description': 'Higher quality embeddings (768 dimensions)'
    }
}

# Default to the faster model
DEFAULT_EMBEDDING_MODEL = 'mini'

# === Command-line Arguments ===
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
    # Create help text for embedding models
    model_options = []
    for k, v in EMBEDDING_MODELS.items():
        model_options.append(f"{k} ({v['description']})")
    
    parser.add_argument('--embedding_model', type=str, default=DEFAULT_EMBEDDING_MODEL,
                        choices=list(EMBEDDING_MODELS.keys()),
                        help=f'Embedding model to use. Options: {", ".join(model_options)}')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Number of files to process in each batch')
    parser.add_argument('--embedding_batch_size', type=int, default=128,
                        help='Batch size for embedding generation')
    return parser.parse_args()

# === Per-file Processing ===
def process_file(filepath, chunk_size, chunk_overlap, folder_path):
    if filepath.endswith('_raw.json'):
        return [], [], 0

    chunks = []
    metadata = []
    total_bytes = os.path.getsize(filepath)

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        doc_meta = {k.strip(): v.strip() for line in lines[:5] if ':' in line for k, v in [line.split(':', 1)]}
        content = "".join(lines[5:])
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)
            metadata.append({
                "file": os.path.relpath(filepath, folder_path),
                "chunk_index": i,
                "metadata": doc_meta
            })

    return chunks, metadata, total_bytes

# === Main RAG Builder ===
def main():
    args = parse_arguments()
    data_dir = args.data_dir
    SAVE_DIR = args.save_dir
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    embedding_model_key = args.embedding_model
    batch_size = args.batch_size
    embedding_batch_size = args.embedding_batch_size

    # Get embedding model config
    embedding_config = EMBEDDING_MODELS[embedding_model_key]
    model_name = embedding_config['name']
    dimension = embedding_config['dimension']

    print(f"🧠 Using embedding model: {model_name} ({dimension} dimensions)")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.environ["LOG_LEVEL"] = "ERROR"
    warnings.filterwarnings("ignore")

    processed_files_path = os.path.join(SAVE_DIR, "processed_files.pkl")
    metadata_path = os.path.join(SAVE_DIR, "metadata.pkl")
    faiss_index_path = os.path.join(SAVE_DIR, "faiss.index")
    model_info_path = os.path.join(SAVE_DIR, "model_info.txt")

    # Check if there's an existing model info and if it's compatible
    if os.path.exists(model_info_path) and os.path.exists(faiss_index_path):
        with open(model_info_path, 'r') as f:
            existing_model = f.readline().strip()
            existing_dim_line = f.readline().strip()
            existing_dim = int(existing_dim_line.split(": ")[1]) if "embedding_dimension" in existing_dim_line else None
        
        if existing_model != model_name:
            print(f"⚠️ Warning: Existing index was created with {existing_model}, but you're using {model_name}.")
            print("This could cause compatibility issues. Consider using the same model or creating a new index.")
            choice = input("Do you want to continue? This will create a new index. (y/n): ")
            if choice.lower() != 'y':
                print("Exiting...")
                return
            
            # Rename existing files to create backups
            backup_suffix = f".bak_{int(time.time())}"
            os.rename(faiss_index_path, faiss_index_path + backup_suffix)
            if os.path.exists(metadata_path):
                os.rename(metadata_path, metadata_path + backup_suffix)
            if os.path.exists(processed_files_path):
                os.rename(processed_files_path, processed_files_path + backup_suffix)
            
            # Reset to create a new index
            index = faiss.IndexFlatIP(dimension)
            all_metadata = []
            processed_files = set()
            print(f"Created new index with model {model_name}")
        else:
            print(f"✅ Found compatible existing index with model {existing_model}")
    else:
        # No existing index, or no model info
        index = None
        all_metadata = []
        processed_files = set()

    # Load processed file list if it exists
    if os.path.exists(processed_files_path) and index is not None:
        with open(processed_files_path, 'rb') as f:
            processed_files = pickle.load(f)
        print(f"📂 Loaded list of {len(processed_files)} already processed files")

    # Load existing index if it exists
    if os.path.exists(faiss_index_path) and index is None:
        index = faiss.read_index(faiss_index_path)
        print(f"📈 Loaded existing index with {index.ntotal} vectors")
    elif index is None:
        index = faiss.IndexFlatIP(dimension)
        print(f"🆕 Created new index with dimension {dimension}")

    # Load existing metadata if it exists
    if os.path.exists(metadata_path) and all_metadata == []:
        with open(metadata_path, 'rb') as f:
            all_metadata = pickle.load(f)
        print(f"📄 Loaded {len(all_metadata)} existing metadata entries")

    # Find all files that need processing
    filepaths = [os.path.join(root, file)
                for root, _, files in os.walk(data_dir)
                for file in files if not file.endswith('_raw.json')]
    
    # Filter out already processed files
    unprocessed_filepaths = [fp for fp in filepaths if fp not in processed_files]
    
    print(f"🔍 Found {len(unprocessed_filepaths)} unprocessed files out of {len(filepaths)} total files")

    if not unprocessed_filepaths:
        print("✅ All files have been processed! Nothing to do.")
        return

    # Initialize SentenceTransformer
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"🔥 Using device: {device}")
    embedder = SentenceTransformer(model_name, device=device)

    # Process files in batches
    batch_files = []
    batch_chunks = []
    batch_metadata = []
    
    start_time = time.time()
    files_processed = 0

    for idx, filepath in enumerate(unprocessed_filepaths):
        # Skip if for some reason the file is already in processed_files
        if filepath in processed_files:
            continue

        # Skip _raw.json files
        if filepath.endswith('_raw.json'):
            print(f"⏩ Skipping {os.path.basename(filepath)}")
            processed_files.add(filepath)
            continue

        # Process the file
        try:
            chunks, metadata, file_bytes = process_file(filepath, chunk_size, chunk_overlap, data_dir)
            if chunks:
                batch_chunks.extend(chunks)
                batch_metadata.extend(metadata)
                batch_files.append(filepath)
                files_processed += 1
                
                if files_processed % 10 == 0:
                    print(f"📊 Processed {files_processed}/{len(unprocessed_filepaths)} files")
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
            continue

        # Process batch if it's full or we're at the end
        if len(batch_files) >= batch_size or idx == len(unprocessed_filepaths) - 1:
            if batch_chunks:  # Only process if we have chunks
                print(f"🚀 Creating embeddings for batch of {len(batch_files)} files ({len(batch_chunks)} chunks)")
                
                # Generate embeddings
                embeddings = embedder.encode(
                    batch_chunks,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    batch_size=embedding_batch_size
                )
                
                # Normalize embeddings
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Add to index
                index.add(embeddings)
                all_metadata.extend(batch_metadata)
                processed_files.update(batch_files)

                # Save progress
                print(f"💾 Saving progress after {len(processed_files)} files...")
                faiss.write_index(index, faiss_index_path)
                with open(metadata_path, 'wb') as f:
                    pickle.dump(all_metadata, f)
                with open(processed_files_path, 'wb') as f:
                    pickle.dump(processed_files, f)
                with open(model_info_path, 'w') as f:
                    f.write(f'{model_name}\n')
                    f.write(f'embedding_dimension: {dimension}')

                # Reset batch
                batch_files, batch_chunks, batch_metadata = [], [], []
                
                # Report on progress
                elapsed = time.time() - start_time
                print(f"⏱️ Time elapsed: {elapsed:.2f}s, {files_processed/elapsed:.2f} files/second")

    # Final report
    total_time = time.time() - start_time
    total_files = len(processed_files)
    total_chunks = len(all_metadata)
    
    print("\n" + "="*80)
    print(f"✅ RAG INDEX COMPLETE")
    print("="*80)
    print(f"📊 Stats: {total_chunks} chunks from {total_files} files")
    print(f"🧠 Embedding model: {model_name} ({dimension} dimensions)")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"💾 Index and metadata saved to {SAVE_DIR}")
    if total_time > 0 and files_processed > 0:
        print(f"📈 Processing rate: {files_processed/total_time:.2f} files/second")
    print("="*80)

if __name__ == "__main__":
    main()


