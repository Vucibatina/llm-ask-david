# Ask David – Local RAG Chat Assistant for David Snyder's Teachings

This is a local Retrieval-Augmented Generation (RAG) application that allows users to ask questions and receive context-aware answers based on the teachings of **David Snyder**, using only his **freely available YouTube content**.

The system runs entirely on your local machine and uses:
- A **quantized LLaMA 2 model** for generating answers
- A **FAISS vector database** for fast semantic search
- **YouTube transcripts** to locate the source of answers
- **A lightweight frontend** for users to interact via the browser

---

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

These files are required to run the application locally.

---

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
