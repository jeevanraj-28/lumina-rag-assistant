#!/bin/sh
set -eu

mkdir -p "$RAG_DATA_DIR/inbox" "$RAG_DATA_DIR/library" "$RAG_DATA_DIR/index" "$OLLAMA_MODELS"
ollama serve &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID' EXIT INT TERM

until ollama list >/dev/null 2>&1; do sleep 1; done
ollama pull "${OLLAMA_MODEL:-qwen2.5:3b}"
ollama pull "${OLLAMA_EMBEDDING_MODEL:-all-minilm}"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
