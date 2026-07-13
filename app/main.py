from __future__ import annotations

import json
import os
import shutil
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import faiss
import numpy as np
import requests
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()
BASE = Path(os.getenv("RAG_DATA_DIR", "data")).resolve()
INBOX, LIBRARY, INDEX_DIR = (BASE / "inbox", BASE / "library", BASE / "index")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
MAX_FILE_BYTES = int(os.getenv("RAG_MAX_FILE_MB", "25")) * 1024 * 1024
TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".html", ".htm", ".py", ".js", ".ts", ".java", ".go", ".rs", ".yaml", ".yml"}
for folder in (INBOX, LIBRARY, INDEX_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Local Document RAG", version="1.0.0")
index: faiss.Index | None = None
metadata: list[dict[str, Any]] = []


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=TOP_K, ge=1, le=15)


class IngestRequest(BaseModel):
    rebuild: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    top_k: int = Field(default=TOP_K, ge=1, le=15)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Use Ollama's 46 MB all-minilm model; no PyTorch process is needed."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": texts}, timeout=180,
        )
        response.raise_for_status()
        vectors = response.json()["embeddings"]
    except (requests.RequestException, KeyError) as exc:
        raise HTTPException(
            503,
            f"Embedding model {EMBEDDING_MODEL} is unavailable. Run: ollama pull {EMBEDDING_MODEL}. ({exc})",
        ) from exc
    result = np.asarray(vectors, dtype="float32")
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


def category(path: Path) -> str:
    ext = path.suffix.lower()
    return {".pdf": "pdf", ".docx": "word", ".txt": "text", ".md": "text", ".csv": "data", ".json": "data", ".html": "web", ".htm": "web"}.get(ext, "other")


def same_file(left: Path, right: Path) -> bool:
    """Compare files without putting whole documents in memory."""
    if left.stat().st_size != right.stat().st_size:
        return False
    hashes = []
    for path in (left, right):
        digest = sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes.append(digest.digest())
    return hashes[0] == hashes[1]


def extract_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        if path.suffix.lower() == ".docx":
            return "\n".join(p.text for p in Document(path).paragraphs)
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc


def save_index(chunks: list[str], metas: list[dict[str, Any]]) -> None:
    global index, metadata
    vectors = embed_texts(chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype="float32"))
    metadata = metas
    faiss.write_index(index, str(INDEX_DIR / "documents.faiss"))
    (INDEX_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def load_index() -> None:
    global index, metadata
    idx, meta = INDEX_DIR / "documents.faiss", INDEX_DIR / "metadata.json"
    if idx.exists() and meta.exists():
        index = faiss.read_index(str(idx))
        metadata = json.loads(meta.read_text(encoding="utf-8"))


def organize_and_index(rebuild: bool) -> dict[str, Any]:
    if rebuild:
        shutil.rmtree(LIBRARY, ignore_errors=True)
        LIBRARY.mkdir(parents=True, exist_ok=True)
    moved, skipped = 0, []
    for source in INBOX.rglob("*"):
        if not source.is_file():
            continue
        if source.stat().st_size > MAX_FILE_BYTES:
            skipped.append(f"Skipped {source.name}: exceeds {MAX_FILE_BYTES // 1024 // 1024} MB limit")
            continue
        target_dir = LIBRARY / category(source)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        suffix = 1
        while target.exists() and not same_file(target, source):
            target = target_dir / f"{source.stem}_{suffix}{source.suffix}"
            suffix += 1
        if not target.exists():
            shutil.copy2(source, target)
            moved += 1
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks, metas = [], []
    for file in LIBRARY.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in TEXT_EXTENSIONS:
            skipped.append(f"Skipped {file.name}: unsupported for text indexing")
            continue
        if file.stat().st_size > MAX_FILE_BYTES:
            skipped.append(f"Skipped {file.name}: exceeds {MAX_FILE_BYTES // 1024 // 1024} MB limit")
            continue
        try:
            text = extract_text(file).strip()
            for number, chunk in enumerate(splitter.split_text(text)):
                chunks.append(chunk)
                metas.append({"file": file.relative_to(LIBRARY).as_posix(), "category": category(file), "chunk": number, "text": chunk})
        except ValueError as exc:
            skipped.append(str(exc))
    if chunks:
        save_index(chunks, metas)
    else:
        global index, metadata
        index, metadata = None, []
    return {"files_added": moved, "chunks_indexed": len(chunks), "skipped": skipped}


def answer(question: str, top_k: int) -> tuple[str, list[dict[str, str]]]:
    if index is None or not metadata:
        raise HTTPException(400, "No indexed documents. Put files in data/inbox and call /ingest first.")
    vector = embed_texts([question])
    scores, positions = index.search(np.asarray(vector, dtype="float32"), min(top_k, len(metadata)))
    sources = []
    context = []
    for score, pos in zip(scores[0], positions[0]):
        item = metadata[int(pos)]
        context.append(f"[Source: {item['file']}]\n{item['text']}")
        sources.append({"file": item["file"], "category": item["category"], "excerpt": item["text"][:350], "score": round(float(score), 3)})
    prompt = """You are a private document assistant. Answer only from the supplied context. If the answer is not present, say you cannot find it in the indexed documents. Cite source filenames in square brackets. Be concise.\n\nCONTEXT:\n%s\n\nQUESTION: %s\nANSWER:""" % ("\n\n".join(context), question)
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": MODEL_NAME, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}, timeout=180)
        response.raise_for_status()
        return response.json()["response"].strip(), sources
    except requests.RequestException as exc:
        raise HTTPException(503, f"Ollama is unavailable at {OLLAMA_URL}. Start it and pull {MODEL_NAME}. ({exc})") from exc


@app.on_event("startup")
def startup() -> None:
    load_index()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_NAME, "embedding_model": EMBEDDING_MODEL, "indexed_chunks": len(metadata), "data_dir": str(BASE)}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict[str, Any]:
    return organize_and_index(payload.rebuild)


@app.post("/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    text, sources = answer(payload.question, payload.top_k)
    return {"answer": text, "sources": sources}


@app.get("/v1/models")
def openai_models() -> dict[str, Any]:
    """Lets OpenAI-compatible local UIs discover this RAG endpoint."""
    return {"object": "list", "data": [{"id": "local-rag", "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def openai_chat(payload: ChatCompletionRequest) -> dict[str, Any]:
    if payload.stream:
        raise HTTPException(400, "Streaming is not enabled; send stream=false.")
    question = next((m.content for m in reversed(payload.messages) if m.role == "user"), None)
    if not question:
        raise HTTPException(400, "A user message is required.")
    text, sources = answer(question, payload.top_k)
    return {
        "id": f"chatcmpl-{uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
        "model": payload.model or "local-rag",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "sources": sources,
    }


@app.get("/files")
def files() -> list[dict[str, Any]]:
    return [{"path": f.relative_to(LIBRARY).as_posix(), "size_bytes": f.stat().st_size, "category": category(f)} for f in LIBRARY.rglob("*") if f.is_file()]


@app.get("/files/{path:path}")
def download(path: str) -> FileResponse:
    file = (LIBRARY / path).resolve()
    if LIBRARY not in file.parents or not file.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(file, filename=file.name)
