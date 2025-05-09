from fastapi import FastAPI, Request, HTTPException, Depends, status, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse

from pydantic import BaseModel
from typing import List, Tuple, Optional
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
import platform
from youtube_transcript_api import YouTubeTranscriptApi
from fastapi.staticfiles import StaticFiles
import secrets
import hashlib
from datetime import datetime, timedelta


# === CONFIG ===
#  To run and set the env variables, do it like this:
#   INITIAL_RAG=rag_combined DATA_DIR=data/new_dataset \
#   exec venv/bin/uvicorn rag_fastapi_server:app --host 0.0.0.0 --port 8000
initial_rag = os.getenv("INITIAL_RAG", "rag_cache")
data_dir = os.getenv("DATA_DIR", "data/DavidSnyder")


llama_model_path = "llm_models/llama-2-7b-chat-hf-q4_k_m.gguf"
mistral_model_path ="llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

faiss_index_path = initial_rag + "/faiss.index"
metadata_path = initial_rag + "/metadata.pkl"
q_and_a_dir = "data/qanda"

MAX_TOKENS = 512

# === AUTHENTICATION CONFIG ===
# In production, store these securely (like in environment variables)
# Hash the passwords in a real application
USERS = {
    "david": {
        "password": hashlib.sha256("thePlanet".encode()).hexdigest(),
        "disabled": False
    },
    "moss": {
        "password": hashlib.sha256("theLightMan".encode()).hexdigest(),
        "disabled": False
    },
    "ana": {
        "password": hashlib.sha256("sestra".encode()).hexdigest(),
        "disabled": False
    }, 
    "vuk": {
        "password": hashlib.sha256("faca".encode()).hexdigest(),
        "disabled": False
    }
}

# Session storage
# In production, use a proper database or Redis
active_sessions = {}
SESSION_COOKIE_NAME = "david_session"
SESSION_EXPIRY_DAYS = 7

# === LOAD MODELS ===
embedder = SentenceTransformer('all-mpnet-base-v2')

# === CHECK GPU ===
def is_gpu_available():
    system = platform.system()
    
    if system == "Darwin":
        # On Mac, assume Metal GPU is available (you built llama.cpp with Metal)
        return True
    elif system == "Linux":
        # On Linux/EC2, check for NVIDIA GPU presence
        return os.path.exists('/dev/nvidia0') or os.getenv('CUDA_VISIBLE_DEVICES') not in (None, '', 'NoDevFiles')
    else:
        # Windows or unknown system
        return False
    
print("initial_rag: " + initial_rag)
print("data_dir: " + data_dir)

if is_gpu_available():
    n_gpu_layers = -1
    print("🚀 GPU detected: using GPU acceleration (n_gpu_layers = -1)")
else:
    n_gpu_layers = 0
    print("🖥️ No GPU detected: using CPU only (n_threads = 8)")

llm = Llama(
    model_path=mistral_model_path,
    n_ctx=2048, # increase to 4096 for bigger context window
    n_threads=14,
    n_gpu_layers=n_gpu_layers,
    verbose=False
)

# Load FAISS index
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

# === AUTH MODELS ===
class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Tuple[str, str, float, str]]

# === AUTH FUNCTIONS ===
def verify_password(plain_password, hashed_password):
    """Verify password against hashed version"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def get_user(username: str):
    """Get user from database"""
    if username in USERS:
        user_dict = USERS[username]
        return UserInDB(**user_dict, username=username, hashed_password=user_dict["password"])
    return None

def authenticate_user(username: str, password: str):
    """Authenticate a user"""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_session(username: str):
    """Create a new session for user"""
    session_id = secrets.token_hex(16)
    expiry = datetime.now() + timedelta(days=SESSION_EXPIRY_DAYS)
    active_sessions[session_id] = {"username": username, "expiry": expiry}
    return session_id

def get_current_user(session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)):
    """Get current user from session cookie"""
    if not session or session not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    session_data = active_sessions[session]
    
    # Check if session is expired
    if datetime.now() > session_data["expiry"]:
        del active_sessions[session]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = session_data["username"]
    user = get_user(username)
    
    if user is None or user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user

# === HELPER FUNCTIONS ===
def wrap_text(text, width=99):
    return "\n".join(textwrap.wrap(text, width=width))

def ask_llm(prompt, max_tokens):
    start = time.time()
    output = llm(prompt, max_tokens=max_tokens, temperature=0.7, top_p=0.95, stop=["</s>"])
    end = time.time()
    answer = output["choices"][0]["text"].strip()
    latency = round(end - start, 3)
    return answer, latency

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

def find_youtube_timestamp_exact_progressive_OLD(video_input, full_text, min_words=5):
    video_id = extract_video_id(video_input)
    local_path = os.path.join(data_dir, f"{video_id}_raw.json")

    # Try to load from local JSON
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    except Exception as e:
        return f"Transcript not available locally: {e}", None, "", 0.0

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

def find_youtube_timestamp_exact_progressive(video_input, full_text, min_words=5):
    print("RUNNING NEW find_youtube_timestamp_exact_progressive with SUMMARY ON " + video_input)
    video_id = extract_video_id(video_input)
    local_path = os.path.join(data_dir, f"{video_id}_raw.json")
    summary_path = os.path.join(data_dir, f"{video_id}_summary.json")

    # Load video summary
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_json = json.load(f)
            video_summary = summary_json.get("summary", "")
    except Exception:
        video_summary = ""

    # Try to load from local transcript JSON
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    except Exception as e:
        return f"Transcript not available locally: {e}", None, "", 0.0, video_summary

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
            return link, int(match_start_time), candidate, 1.0, video_summary

    return "No match found at any length.", None, "", 0.0, video_summary

# === ROUTES ===
@app.post("/login")
async def login(login_request: LoginRequest):
    """Login endpoint"""
    user = authenticate_user(login_request.username, login_request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    session_id = create_session(user.username)
    
    response = JSONResponse(
        content={
            "username": user.username,
            "message": "Login successful"
        }
    )
    
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        max_age=60 * 60 * 24 * SESSION_EXPIRY_DAYS,  # 7 days in seconds
        samesite="lax",
        # secure=True  # Enable in production with HTTPS
    )
    
    return response

@app.post("/logout")
async def logout(user: User = Depends(get_current_user), session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)):
    """Logout endpoint"""
    if session and session in active_sessions:
        del active_sessions[session]
    
    response = JSONResponse(content={"message": "Logout successful"})
    response.delete_cookie(SESSION_COOKIE_NAME)
    
    return response

@app.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return {"username": current_user.username}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    current_user: User = Depends(get_current_user)
):
    """Protected endpoint to ask questions"""
    q = request.question.strip()
    print(f"Question asked by {current_user.username}: {q}")
    
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
        for idx, i in enumerate(I[0]):
            chunk_text = chunks[i]

            entry = metadata[i]
            file = entry['file']
            meta = entry.get('metadata', {})
            # original_link = meta.get("Video URL", "N/A")
            original_link = meta.get("Video URL")
            if not original_link or original_link == "N/A":
                original_link = entry['file']
            # Remove ".txt" if it ends with it
            if original_link.endswith(".txt"):
                original_link = original_link[:-4]

            updated_link, timestamp, matched_text, score, video_summary = find_youtube_timestamp_exact_progressive(original_link, chunk_text)

            print("VIDEO SUMMARY: " + video_summary)

            if file not in seen_files:
                source_entries.append((updated_link, matched_text, float(D[0][idx]), video_summary))
                seen_files.add(file)

            retrieved_chunks.append(chunk_text)

        retrieved_context = "\n".join(retrieved_chunks)
        prompt = f"""[INST] Use the context below to answer the question.\n\nContext:\n{retrieved_context}\n\nQuestion: {q}\n[/INST]"""
    else:
        prompt = f"[INST] {q} [/INST]"
        source_entries = [("No video link", "No reference text available.", 0.0)]
        retrieved_chunks = []

    # ==== Start timing model answering ====
    start_time = time.time()
    answer, _ = ask_llm(prompt, max_tokens=MAX_TOKENS)
    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱️ Answer generated in {duration:.2f} seconds.")

    # === Save to file if enough disk space ===
    try:
        free_space = shutil.disk_usage(q_and_a_dir).free
        if free_space > 500 * 1024 * 1024:  # 500MB
            os.makedirs(q_and_a_dir, exist_ok=True)
            timestamp = int(time.time())
            data = {
                "username": current_user.username,
                "question": q,
                "answer": answer,
                "references": source_entries,
                "chunks": retrieved_chunks,
                "latency": duration
            }
            file_path = os.path.join(q_and_a_dir, f"{timestamp}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            print("⚠️ Not enough disk space. Skipping save.")
    except Exception as e:
        print(f"⚠️ Error saving Q&A file: {e}")

    return QuestionResponse(answer=answer, sources=source_entries)

# === STATIC FILES ===
# Serve static files, but protect with authentication check
@app.get("/ask_david.html")
async def secure_ask_david(current_user: User = Depends(get_current_user)):
    """Serve the ask_david.html file, but only to authenticated users"""
    with open("static/ask_david.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

# Redirect root to login if not authenticated
@app.get("/")
async def root(session: Optional[str] = Cookie(None, alias=SESSION_COOKIE_NAME)):
    """Redirect to login page if not authenticated"""
    # Check if user is authenticated
    try:
        if session and session in active_sessions:
            return RedirectResponse(url="/ask_david.html")
    except:
        pass
    
    # If not authenticated, serve login page
    with open("static/login.html", "r") as f:
        content = f.read()
    return HTMLResponse(content=content)

# Mount static files for CSS, JS, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

