#!/bin/bash

# Activate your virtual environment
source /Users/vuk/envs/rag_env/bin/activate

# Upgrade pip silently just in case
pip install --upgrade pip --quiet

# Install required packages
pip install fastapi sentence-transformers --quiet
pip install youtube_transcript_api  --quiet

# Run the FastAPI server with correct Uvicorn binary
/Users/vuk/envs/rag_env/bin/uvicorn rag_fastapi_server:app --host 0.0.0.0 --port 8000

