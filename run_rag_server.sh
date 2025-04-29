#!/bin/bash

# Upgrade pip silently just in case
pip install --upgrade pip --quiet

# Run the FastAPI server with correct Uvicorn binary
/Users/vuk/envs/rag_env/bin/uvicorn rag_fastapi_server:app --host 0.0.0.0 --port 8000

