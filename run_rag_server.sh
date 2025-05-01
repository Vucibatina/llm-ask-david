#!/bin/bash
cd /home/ubuntu/llm-ask-david

# Create virtual environment if missing
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate venv and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run FastAPI server
exec venv/bin/uvicorn rag_fastapi_server:app --host 0.0.0.0 --port 8000


