from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn
import time
import textwrap
import faiss
import pickle
import os
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import re
import json
import shutil

from youtube_transcript_api import YouTubeTranscriptApi
from fastapi.staticfiles import StaticFiles


# === CONFIG ===
initial_rag = "rag_cache"
chatgpt_rag = "rag_cache_chatgpt"
antropic_rag = "rag_cache_antropic"
llama_model_path = "llm_models/llama-2-7b-chat-hf-q4_k_m.gguf"
faiss_index_path = initial_rag + "/faiss.index"
metadata_path = initial_rag + "/metadata.pkl"
data_dir = "data/DavidSnyder"
q_and_a_dir = "data/qanda"

MAX_TOKENS = 512

# === LOAD MODELS ===
embedder = SentenceTransformer('all-mpnet-base-v2')

# === CHECK GPU ===
def is_gpu_available():
    return os.path.exists('/dev/nvidia0') or os.getenv('CUDA_VISIBLE_DEVICES') not in (None, '', 'NoDevFiles')

if is_gpu_available():
    n_gpu_layers = -1
    print("🚀 GPU detected: using GPU acceleration (n_gpu_layers = -1)")
else:
    n_gpu_layers = 0
    print("🖥️ No GPU detected: using CPU only (n_threads = 8)")

llm = Llama(
    model_path=llama_model_path,
    n_ctx=2048, # increase to 4096 for bigger context window
    n_threads=8,
    n_gpu_layers=n_gpu_layers,
    verbose=False
)

# OLD STYLE NO GPU 
# llm = Llama(model_path=llama_model_path, n_ctx=2048, verbose=False)

index = faiss.read_index(faiss_index_path)

# Load documents and rebuild chunks list
with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

chunks = []
for i, entry in enumerate(metadata):
    file_path = os.path.join(data_dir, entry['file'])
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            content = ''.join(lines[5:])  # skip first 5 lines (metadata headers)
            chunk_start = entry['chunk_index']
            chunk_size = 1000  # consistent with your original setup
            chunk_text = content[chunk_start:chunk_start + chunk_size]
            chunks.append(chunk_text)
    except Exception as e:
        print(f"Error loading chunk {i} from {file_path}: {e}")
        chunks.append("")

# === APP SETUP ===
app = FastAPI()

# ✅ CORS Middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Tuple[str, str]]

def wrap_text(text, width=99):
    return "\n".join(textwrap.wrap(text, width=width))

def ask_llm(prompt, max_tokens):
    start = time.time()
    output = llm(prompt, max_tokens=max_tokens, temperature=0.7, top_p=0.95, stop=["</s>"])
    end = time.time()
    answer = output["choices"][0]["text"].strip()
    latency = round(end - start, 3)
    return answer, latency

#### VUK JUST ADDED THIS
def extract_video_id(video_input):
    match = re.search(r"(?:v=|youtu\.be/|embed/)([0-9A-Za-z_-]{11})", video_input)
    return match.group(1) if match else video_input

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text).lower()

def progressive_shrink(words, min_words=5):
    while len(words) > min_words:
        yield " ".join(words)
        words = words[:len(words) // 2]
    if len(words) >= min_words:
        yield " ".join(words)

def find_youtube_timestamp_exact_progressive(video_input, full_text, min_words=5):
    video_id = extract_video_id(video_input)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        return f"Transcript not available: {e}", None, "", 0.0

    # Build cleaned full transcript string and map position to time
    full_cleaned_text = ""
    char_pos_to_start_time = {}

    for entry in transcript:
        raw_text = entry['text'].replace('\n', ' ')
        cleaned_text = clean_text(raw_text)
        start_char = len(full_cleaned_text)
        full_cleaned_text += cleaned_text + " "
        
        for i in range(len(cleaned_text)):
            char_pos_to_start_time[start_char + i] = entry['start']

    original_words = clean_text(full_text).split()

    for candidate in progressive_shrink(original_words, min_words=min_words):
        match_pos = full_cleaned_text.find(candidate)
        if match_pos != -1:
            match_start_time = char_pos_to_start_time.get(match_pos, 0)
            link = f"https://www.youtube.com/watch?v={video_id}&t={int(match_start_time)}"
            return link, int(match_start_time), candidate, 1.0

    return "No match found at any length.", None, "", 0.0

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    q = request.question.strip()
    if len(q) > 1000:
        raise HTTPException(status_code=400, detail="Question too long. Limit to 1000 characters.")

    q_embedding = embedder.encode([q])
    # Ensure the embedding is in the correct format for FAISS
    q_embedding = q_embedding.astype(np.float32)
    
    D, I = index.search(q_embedding, k=3)
    context_similarity_threshold = 1000
    use_context = D[0][0] < context_similarity_threshold

    retrieved_chunks = []
    source_entries = []
    seen_files = set()

    if use_context:
        for i in I[0]:
            chunk_text = chunks[i]

            entry = metadata[i]
            file = entry['file']
            meta = entry.get('metadata', {})
            original_link = meta.get("Video URL", "N/A")

            updated_link, timestamp, matched_text, score = find_youtube_timestamp_exact_progressive(original_link, chunk_text)

            if file not in seen_files:
                source_entries.append((updated_link, matched_text))
                seen_files.add(file)

            retrieved_chunks.append(chunk_text)

        retrieved_context = "\n".join(retrieved_chunks)
        prompt = f"""[INST] Use the context below to answer the question.\n\nContext:\n{retrieved_context}\n\nQuestion: {q}\n[/INST]"""
    else:
        prompt = f"[INST] {q} [/INST]"
        source_entries = [("No video link", "No reference text available.")]
        retrieved_chunks = []

    answer, _ = ask_llm(prompt, max_tokens=MAX_TOKENS)

    # === Save to file if enough disk space ===
    try:
        free_space = shutil.disk_usage(q_and_a_dir).free
        if free_space > 500 * 1024 * 1024:  # 500MB
            os.makedirs(q_and_a_dir, exist_ok=True)
            timestamp = int(time.time())
            data = {
                "question": q,
                "answer": answer,
                "references": source_entries,
                "chunks": retrieved_chunks
            }
            file_path = os.path.join(q_and_a_dir, f"{timestamp}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            print("⚠️ Not enough disk space. Skipping save.")
    except Exception as e:
        print(f"⚠️ Error saving Q&A file: {e}")

    return QuestionResponse(answer=answer, sources=source_entries)

app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

