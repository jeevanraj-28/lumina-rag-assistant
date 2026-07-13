FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    RAG_DATA_DIR=/app/data \
    OLLAMA_MODELS=/app/data/ollama-models \
    OLLAMA_HOST=127.0.0.1:11434

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh && mkdir -p /app/data/inbox /app/data/library /app/data/index /app/data/ollama-models

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
