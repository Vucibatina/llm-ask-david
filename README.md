# Ask David – Local RAG Chat Assistant for David Snyder's Teachings

This is a local Retrieval-Augmented Generation (RAG) application that allows users to ask questions and receive context-aware answers based on the teachings of **David Snyder**, using only his **freely available YouTube content**.

The system runs entirely on your local machine and uses:
- A **quantized LLaMA 2 model** for generating answers
- A **FAISS vector database** for fast semantic search
- **YouTube transcripts** to locate the source of answers
- **A lightweight frontend** for users to interact via the browser

---
## Running
To make sure it runs all the time on the server:

Create a file /etc/systemd/system/fastapi.service with with content:
[Unit]
Description=FastAPI Server for David
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/llm-ask-david
ExecStart=/home/ubuntu/llm-ask-david/run_rag_server.sh
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target


Then, run these commands
`sudo systemctl daemon-reexec`
`sudo systemctl daemon-reload`
`sudo systemctl enable fastapi`
`sudo systemctl restart fastapi`

When debugging, you can stop it
`sudo systemctl stop fastapi`
and to put it back, run
`sudo systemctl start fastapi`


## 🛠️ How It Works

1. The user types a question into a web interface.
2. The question is encoded using the `all-mpnet-base-v2` SentenceTransformer model.
3. A FAISS index is searched to retrieve the most relevant transcript chunks.
4. The selected chunks are passed to a local LLaMA 2 7B model (`gguf` format) via `llama-cpp-python`.
5. The model generates a response using the retrieved context.
6. The app then:
   - Shows the answer
   - Lists the matched YouTube links (with timestamp)
   - Optionally stores the full Q&A interaction in a local JSON file

---

## 📁 Project Structure

```
ask_david/
├── llm_models/
│   └── llama-2-7b-chat-hf-q4_k_m.gguf       # Quantized LLaMA model for local inference
│
├── rag_cache/
│   ├── faiss.index                          # FAISS vector index of all embedded transcript chunks
│   └── metadata.pkl                         # Metadata for locating each chunk in the original text
│
├── data/
│   ├── DavidSnyder/                         # Raw transcript files (text format, one per video)
│   └── qanda/                               # JSON logs of all questions/answers (if space permits)
│
├── static/
│   └── ask_david.html                       # Frontend UI for asking questions
│
├── server.py                                # FastAPI backend for inference and routing
└── README.md                                # This file
```

---

## 🔍 Example Use Case

**Question:**
> "How can I overcome fear using David Snyder’s NLP techniques?"

**System Response:**
- A text response generated in David’s voice/style
- 2–3 YouTube links to exact moments where that material was covered
- The entire interaction saved locally as a timestamped JSON file

---

## 💾 Disk Space Behavior

To prevent the system from consuming excessive storage:

- It **only saves** the Q&A JSON files if **more than 500MB** of disk space is free.
- Each file is named using a Unix timestamp, e.g., `1713748123.json`.

---

## 📦 Requirements

Make sure you have Python 3.8 or later. Install dependencies with:

```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu llama-cpp-python youtube-transcript-api
```

### Optional:
- Install `ffmpeg` if you plan to process video/audio files later.

---

## 🧠 Data Flow Summary

- **User question** → embedded via SentenceTransformer
- **Similarity search** → FAISS index → retrieves relevant transcript chunks
- **Prompt creation** → [INST] format with context
- **Answer generation** → LLaMA model via `llama-cpp-python`
- **Source alignment** → YouTube timestamps using transcript matching
- **Display & logging** → Results shown in browser + optional JSON log

---

## 📦 Manual Setup for Large Files

Due to GitHub's 100MB file size limit, this repo does **not** include large binary files like the LLaMA model or FAISS index.

After cloning the repository, you must manually add these files:

- `llm_models/llama-2-7b-chat-hf-q4_k_m.gguf`
- `rag_cache/faiss.index` (or `index.faiss` depending on your setup)

You can:
- Transfer them via **SSH** or **scp**
- Use a shared folder like Dropbox, Google Drive, or external storage
- Or download from your own private hosting if you’ve set that up

Example of scp commands:
`scp -i ~/Downloads/g4-gpu-ask-david-2nd.pem /Users/vuk/projects/ask_david_clean/llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf ubuntu@3.92.15.44:/home/ubuntu/llm-ask-david/llm_models/`
`scp /Users/vuk/projects/ask_david_clean/llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf g4-gpu-ask-david-2nd:/home/ubuntu/llm-ask-david/llm_models/`
`scp /Users/vuk/projects/ask_david_clean/data/DavidSnyder/*.txt g4-gpu-ask-david-2nd:/home/ubuntu/llm-ask-david/data/DavidSnyder/`
`scp /Users/vuk/projects/ask_david_clean/rag_cache/*  g4-gpu-ask-david-2nd:/home/ubuntu/llm-ask-david/rag_cache/`

These files are required to run the application locally.

---

## Running a youtube video files transcriber

Big part of the data training pipeline is having transcribed video data pulled
from youtube.  To be able to do this, here is how to run the transcriber where you
will need to specifiy the target directory and youtube creator name so that all
their videos can be pulled and transcribed.  Here is the example on how to run:

`python transcript_loader.py --path "data/DavidSnyder" --username "DavidSnyderNLP"`
`python transcript_loader.py --path "data/hubermanlab" --username "@hubermanlab"`

For each video, we will produce 3 files:
video_id.txt          --->  Full text of the video transcript
video_id_raw.json     --->  JSON file of timings of transcript relating to video
                          to know exactly where to go in the video when you want to 
                          listen to transcript
video_id_summary.json --->  Video Metadata with Summary of the entire video
---

## Running a RAG Creation

Creating a RAG is necessary for the process to run.  Files are either copied into rag_cache
directory or generated.  In case you want to generate it, use the below command to do it
`python build_rag.py --data_dir "/Users/vuk/projects/ask_david_clean/data/DavidSnyder" --model_path "/Users/vuk/projects/llm_models/llama-2-7b-chat-hf-q4_k_m.gguf" --save_dir "rag_cache"`

Latest mode, use mistral model, embedding model mpnet for more accuracy, mini for speed
`python build_rag.py --data_dir "/Users/vuk/projects/ask_david_clean/data/DavidSnyder" --model_path "/Users/vuk/projects/llm_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf" --save_dir "rag_cache" --embedding_model mpnet --chunk_size 1000 --chunk_overlap 100`

--
## Running the fast api server

Running the server, you need a RAG directory, LLM Directory hosting the model file
and data directory, hosting the documents, in this case youtube video transcripts
and metadata files.  To specify where the RAG directory is and Data directory, use
the below command
`INITIAL_RAG=rag_combined DATA_DIR=data/new_dataset \
   exec venv/bin/uvicorn rag_fastapi_server:app --host 0.0.0.0 --port 8000`

--

## ✉️ Author

**Produced by [Vuk Radovic](mailto:vucibatina@hotmail.com)**  
Built with ❤️ to make David Snyder’s material searchable, accessible, and useful in everyday life.
---

## 🌐 How to Access the Web Interface

To use the app through your browser, open this file locally:

```
static/ask_david.html
```

This HTML file provides a simple frontend for interacting with the backend API. You must have the FastAPI server (`server.py`) running locally before using the interface.
