from __future__ import annotations

from contextlib import asynccontextmanager

import json
import os
import re
import shutil
import time
from queue import Empty, Full, Queue
from threading import Lock, Thread
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator, Literal
from urllib.parse import quote
from uuid import uuid4


def sanitize_response_text(text: str) -> str:
    """Filter out raw underlying LLM provider names (Qwen, Alibaba) into Lumina RAG / Jeevan Raj M custom branding."""
    text = re.sub(r"I am Qwen,?\s*created by Alibaba Cloud\.?", "I am Lumina RAG, created and developed by Jeevan Raj M.", text, flags=re.IGNORECASE)
    text = re.sub(r"I was not made by anyone as I am an AI assistant.*", "I am Lumina RAG, a private local document AI assistant created and developed by Jeevan Raj M.", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:created|developed|built|made)\s+by\s+Alibaba(?:\s+Cloud)?\b", "developed by Jeevan Raj M", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAlibaba(?:\s+Cloud)?\b", "Lumina RAG System", text, flags=re.IGNORECASE)
    text = re.sub(r"\bQwen2?\.?5?(?::\d+b)?\b", "Lumina AI", text, flags=re.IGNORECASE)
    text = re.sub(r"\bQwen\b", "Lumina AI", text, flags=re.IGNORECASE)
    return text

import faiss
import numpy as np
import requests
from docx import Document
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

from app.unlimited_ocr import UnlimitedOCRError, UnlimitedOCRSettings, parse_pdf

load_dotenv()
BASE = Path(os.getenv("RAG_DATA_DIR", "data")).resolve()
INBOX, LIBRARY, INDEX_DIR, TRASH = (BASE / "inbox", BASE / "library", BASE / "index", BASE / "trash")
MEMORY_DIR = BASE / "memory"
FEEDBACK_DIR = BASE / "feedback"
SESSIONS_DIR = BASE / "sessions"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
MAX_FILE_BYTES = int(os.getenv("RAG_MAX_FILE_MB", "25")) * 1024 * 1024
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "48"))
OCR_MODE = os.getenv("RAG_OCR_MODE", "native").lower()
MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.35"))
OCR_MIN_CHARS_PER_PAGE = int(os.getenv("RAG_OCR_MIN_CHARS_PER_PAGE", "80"))
UNLIMITED_OCR_URL = os.getenv("UNLIMITED_OCR_URL", "").rstrip("/")
UNLIMITED_OCR_MODEL = os.getenv("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")
UNLIMITED_OCR_DPI = int(os.getenv("UNLIMITED_OCR_DPI", "200"))
UNLIMITED_OCR_MAX_PAGES = int(os.getenv("UNLIMITED_OCR_MAX_PAGES", "24"))
UNLIMITED_OCR_MAX_TOKENS = int(os.getenv("UNLIMITED_OCR_MAX_TOKENS", "24576"))
UNLIMITED_OCR_TIMEOUT_SECONDS = int(os.getenv("UNLIMITED_OCR_TIMEOUT_SECONDS", "1200"))
MAX_MEMORIES = 50
MAX_SESSIONS = 50
if OCR_MODE not in {"native", "auto", "unlimited"}:
    OCR_MODE = "native"
TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".html", ".htm", ".py", ".js", ".ts", ".java", ".go", ".rs", ".yaml", ".yml"}
EMBEDDING_FAMILIES = {"bert", "nomic-bert"}
TRASH_RETENTION_DAYS = 30
for folder in (INBOX, LIBRARY, INDEX_DIR, TRASH, MEMORY_DIR, FEEDBACK_DIR, SESSIONS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Mutable active model — changed at runtime by /models/active
active_model_name: str = DEFAULT_MODEL
model_lock = Lock()


def get_active_model() -> str:
    with model_lock:
        return active_model_name


def set_active_model(name: str) -> None:
    global active_model_name
    with model_lock:
        active_model_name = name


# ── Adaptive Memory System ──

def _memories_path() -> Path:
    return MEMORY_DIR / "memories.json"


def load_all_memories() -> list[dict[str, Any]]:
    """Load the full memory store from disk."""
    path = _memories_path()
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_all_memories(memories: list[dict[str, Any]]) -> None:
    _memories_path().write_text(json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8")


def load_active_memories() -> list[dict[str, Any]]:
    """Load only active memories for prompt injection."""
    return [m for m in load_all_memories() if m.get("active", True)]


def build_memory_prompt_section(memories: list[dict[str, Any]]) -> str:
    """Format active memories into a prompt section the LLM can reference."""
    if not memories:
        return ""
    type_labels = {"user_fact": "USER FACT", "correction": "CORRECTION", "preference": "PREFERENCE"}
    lines = ["USER MEMORY & CORRECTIONS (always apply these to your responses):"]
    for m in memories:
        prefix = type_labels.get(m.get("type", ""), "NOTE")
        lines.append(f"- [{prefix}] {m['content']}")
    return "\n".join(lines)


def add_memory(memory_type: str, content: str, original_feedback: str | None = None,
               question_context: str | None = None) -> dict[str, Any]:
    """Add a new memory entry and persist it."""
    memories = load_all_memories()
    entry: dict[str, Any] = {
        "id": f"mem_{uuid4().hex[:10]}",
        "type": memory_type,
        "content": content,
        "original_feedback": original_feedback,
        "question_context": question_context,
        "created_at": int(time.time()),
        "active": True,
    }
    memories.append(entry)
    # Enforce max memories limit
    if len(memories) > MAX_MEMORIES:
        memories = memories[-MAX_MEMORIES:]
    save_all_memories(memories)
    return entry


# ── Sessions Persistence ──

def _sessions_path() -> Path:
    return SESSIONS_DIR / "history.json"


def load_sessions() -> list[dict[str, Any]]:
    path = _sessions_path()
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_sessions(sessions: list[dict[str, Any]]) -> None:
    _sessions_path().write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")


def add_session(question: str, answer_text: str, sources: list[dict[str, Any]],
                model: str | None = None) -> dict[str, Any]:
    """Save a Q&A pair to session history."""
    sessions = load_sessions()
    entry: dict[str, Any] = {
        "id": f"ses_{uuid4().hex[:10]}",
        "question": question,
        "answer": answer_text[:2000],
        "sources_count": len(sources),
        "model": model or get_active_model(),
        "model_display": brand_model_name(model or get_active_model()),
        "created_at": int(time.time()),
    }
    sessions.append(entry)
    if len(sessions) > MAX_SESSIONS:
        sessions = sessions[-MAX_SESSIONS:]
    save_sessions(sessions)
    return entry


def brand_model_name(raw: str) -> str:
    """Convert raw Ollama model names into user-friendly branded display names."""
    clean = raw.lower().strip()
    if clean.startswith("qwen"):
        parts = clean.replace("qwen", "").replace(":", " ").replace("-", " ").split()
        size = next((p for p in parts if p.endswith("b")), "")
        return f"Lumina {size.upper()}" if size else "Lumina AI"
    if "deepseek" in clean:
        parts = clean.replace(":", " ").replace("-", " ").split()
        size = next((p for p in parts if p.endswith("b")), "")
        label = "DeepSeek"
        if "r1" in clean:
            label = "DeepSeek R1"
        if "coder" in clean:
            label = "DeepSeek Coder"
        return f"{label} {size.upper()}" if size else label
    if "phi" in clean:
        parts = clean.replace(":", " ").replace("-", " ").split()
        return "Phi-4 Mini" if "mini" in clean else "Phi-4"
    if "llama" in clean:
        parts = clean.replace(":", " ").replace("-", " ").split()
        size = next((p for p in parts if p.endswith("b")), "")
        return f"Llama {size.upper()}" if size else "Llama"
    if "gemma" in clean:
        parts = clean.replace(":", " ").replace("-", " ").split()
        size = next((p for p in parts if p.endswith("b")), "")
        return f"Gemma {size.upper()}" if size else "Gemma"
    if "mistral" in clean:
        parts = clean.replace(":", " ").replace("-", " ").split()
        size = next((p for p in parts if p.endswith("b")), "")
        return f"Mistral {size.upper()}" if size else "Mistral"
    # Fallback: capitalize and strip tag
    base = raw.split(":")[0].replace("-", " ").replace("_", " ").title()
    return base


def list_ollama_models() -> list[dict[str, Any]]:
    """Fetch models from Ollama and return chat-capable ones with branded names."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except requests.RequestException:
        return []
    result = []
    for m in models:
        name = m.get("name", "")
        details = m.get("details", {})
        family = details.get("family", "").lower()
        # Skip embedding-only models
        if family in EMBEDDING_FAMILIES:
            continue
        size_bytes = m.get("size", 0)
        size_gb = round(size_bytes / 1e9, 1)
        result.append({
            "id": name,
            "display_name": brand_model_name(name),
            "family": details.get("family", ""),
            "parameter_size": details.get("parameter_size", ""),
            "size_gb": size_gb,
        })
    result.sort(key=lambda m: m["size_gb"])
    return result


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_index()
    yield


app = FastAPI(title="Local Document RAG", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
index: faiss.Index | None = None
metadata: list[dict[str, Any]] = []
index_lock = Lock()
ingest_lock = Lock()
event_lock = Lock()
event_subscribers: list[Queue[str]] = []
ingest_jobs: dict[str, dict[str, Any]] = {}
active_ingest_job_id: str | None = None
MAX_INGEST_JOBS = 50


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=TOP_K, ge=1, le=15)
    model: str | None = None


class IngestRequest(BaseModel):
    rebuild: bool = False
    ocr_mode: Literal["native", "auto", "unlimited"] | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    top_k: int = Field(default=TOP_K, ge=1, le=15)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed in bounded batches so large local libraries do not spike RAM."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": batch}, timeout=180,
            )
            response.raise_for_status()
            vectors.extend(response.json()["embeddings"])
        except (requests.RequestException, KeyError) as exc:
            raise HTTPException(
                503,
                f"Embedding model {EMBEDDING_MODEL} is unavailable. Run: ollama pull {EMBEDDING_MODEL}. ({exc})",
            ) from exc
    result = np.asarray(vectors, dtype="float32")
    if result.ndim != 2 or result.shape[0] != len(texts):
        raise HTTPException(502, "Ollama returned an invalid embedding response.")
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


def category(path: Path) -> str:
    ext = path.suffix.lower()
    return {".pdf": "pdf", ".docx": "word", ".txt": "text", ".md": "text", ".csv": "data", ".json": "data", ".html": "web", ".htm": "web"}.get(ext, "other")


def citation_id_for(file: str, page: int | None, text: str) -> str:
    return sha256(f"{file}\x1f{page or 0}\x1f{text}".encode("utf-8")).hexdigest()[:20]


def normalize_metadata(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make indexes created before metadata upgrade usable with accurate file metadata."""
    normalized = []
    for number, entry in enumerate(entries):
        item = dict(entry)
        rel_file = item.get("file", "")
        lib_path = LIBRARY / rel_file if rel_file else None

        item.setdefault("page", None)
        item.setdefault("chunk", number)
        item.setdefault("category", category(lib_path) if lib_path else "document")

        if lib_path and lib_path.is_file():
            stat = lib_path.stat()
            mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            size_kb = round(stat.st_size / 1024, 1)
            size_str = f"{size_kb} KB" if stat.st_size < 1048576 else f"{round(stat.st_size / 1048576, 1)} MB"
            item.setdefault("upload_date", mtime_str)
            item.setdefault("file_size", size_str)
        else:
            item.setdefault("upload_date", "Unknown")
            item.setdefault("file_size", "Unknown")

        item.setdefault("citation_id", citation_id_for(item["file"], item["page"], item["text"]))
        normalized.append(item)
    return normalized


def unlimited_ocr_settings() -> UnlimitedOCRSettings:
    if not UNLIMITED_OCR_URL:
        raise UnlimitedOCRError(
            "Unlimited-OCR is not configured. Set UNLIMITED_OCR_URL to its local vLLM server."
        )
    return UnlimitedOCRSettings(
        base_url=UNLIMITED_OCR_URL,
        model=UNLIMITED_OCR_MODEL,
        dpi=UNLIMITED_OCR_DPI,
        max_pages=UNLIMITED_OCR_MAX_PAGES,
        max_tokens=UNLIMITED_OCR_MAX_TOKENS,
        timeout_seconds=UNLIMITED_OCR_TIMEOUT_SECONDS,
    )


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


def extract_text_units(path: Path, ocr_mode: str = OCR_MODE) -> list[tuple[int | None, str]]:
    """Extract text in page-sized units when the source format supports it."""
    try:
        if path.suffix.lower() == ".pdf":
            native_pages: list[tuple[int | None, str]] = [
                (number, page.extract_text() or "")
                for number, page in enumerate(PdfReader(path).pages, start=1)
            ]
            native_characters = sum(len(text.strip()) for _, text in native_pages)
            should_run_ocr = ocr_mode == "unlimited" or (
                ocr_mode == "auto" and native_characters < OCR_MIN_CHARS_PER_PAGE * len(native_pages)
            )
            if not should_run_ocr:
                return native_pages
            try:
                parsed = parse_pdf(path, unlimited_ocr_settings())
                # The model emits one coherent Markdown stream for a multi-page request.
                # Page-level anchors remain unavailable until its grounding boxes are mapped.
                return [(None, parsed.markdown)]
            except UnlimitedOCRError as exc:
                if ocr_mode == "unlimited":
                    raise ValueError(str(exc)) from exc
                return native_pages
        if path.suffix.lower() == ".docx":
            return [(None, "\n".join(paragraph.text for paragraph in Document(path).paragraphs))]
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    except Exception as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc


def extract_text(path: Path) -> str:
    """Backward-compatible whole-document text extraction helper."""
    return "\n".join(text for _, text in extract_text_units(path))


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    separators = ("\n\n", "\n", ". ", " ", "")

    def split(value: str, separator_index: int) -> list[str]:
        if len(value) <= chunk_size:
            return [value]
        if separator_index >= len(separators):
            step = max(1, chunk_size - chunk_overlap)
            return [value[index:index + chunk_size] for index in range(0, len(value), step)]
        separator = separators[separator_index]
        parts = ([value[index:index + chunk_size] for index in range(0, len(value), chunk_size)]
                 if not separator else value.split(separator))
        merged: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    merged.extend(split(current, separator_index + 1))
                current = part
        if current:
            merged.extend(split(current, separator_index + 1))
        return merged

    chunks: list[str] = []
    for chunk in split(text, 0):
        if chunks and chunk_overlap:
            chunk = f"{chunks[-1][-chunk_overlap:]}{chunk}"
        chunks.append(chunk[:chunk_size])
    return chunks


def save_index(chunks: list[str], metas: list[dict[str, Any]]) -> None:
    """Persist a completed index, then atomically make it live for queries."""
    global index, metadata
    vectors = embed_texts(chunks)
    next_index = faiss.IndexFlatIP(vectors.shape[1])
    next_index.add(np.asarray(vectors, dtype="float32"))
    index_path = INDEX_DIR / "documents.faiss"
    metadata_path = INDEX_DIR / "metadata.json"
    next_index_path = INDEX_DIR / f".documents-{uuid4().hex}.faiss"
    next_metadata_path = INDEX_DIR / f".metadata-{uuid4().hex}.json"
    faiss.write_index(next_index, str(next_index_path))
    next_metadata_path.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    next_index_path.replace(index_path)
    next_metadata_path.replace(metadata_path)
    with index_lock:
        index, metadata = next_index, metas


def load_index() -> None:
    global index, metadata
    idx, meta = INDEX_DIR / "documents.faiss", INDEX_DIR / "metadata.json"
    if idx.exists() and meta.exists():
        loaded_index = faiss.read_index(str(idx))
        loaded_metadata = normalize_metadata(json.loads(meta.read_text(encoding="utf-8")))
        with index_lock:
            index, metadata = loaded_index, loaded_metadata


def organize_and_index(
    rebuild: bool,
    report_progress: Callable[[str, dict[str, Any]], None] | None = None,
    ocr_mode: str = OCR_MODE,
) -> dict[str, Any]:
    def report(event: str, **details: Any) -> None:
        if report_progress:
            report_progress(event, details)

    report("parse.started", message="Scanning local documents")
    if rebuild:
        shutil.rmtree(LIBRARY, ignore_errors=True)
        LIBRARY.mkdir(parents=True, exist_ok=True)
    moved, skipped = 0, []
    inbox_files = [source for source in INBOX.rglob("*") if source.is_file() and not source.name.startswith(".")]
    for source_number, source in enumerate(inbox_files, start=1):
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
        source.unlink(missing_ok=True)
        report(
            "parse.progress",
            completed=source_number,
            total=len(inbox_files),
            file=source.name,
            message="Organizing documents",
        )
    chunks, metas = [], []
    library_files = [file for file in LIBRARY.rglob("*") if file.is_file() and not file.name.startswith(".")]
    for file_number, file in enumerate(library_files, start=1):
        if file.suffix.lower() not in TEXT_EXTENSIONS:
            skipped.append(f"Skipped {file.name}: unsupported for text indexing")
            continue
        if file.stat().st_size > MAX_FILE_BYTES:
            skipped.append(f"Skipped {file.name}: exceeds {MAX_FILE_BYTES // 1024 // 1024} MB limit")
            continue
        try:
            relative_file = file.relative_to(LIBRARY).as_posix()
            chunk_number = 0
            file_stat = file.stat()
            upload_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_stat.st_mtime))
            size_kb = round(file_stat.st_size / 1024, 1)
            file_size_str = f"{size_kb} KB" if file_stat.st_size < 1048576 else f"{round(file_stat.st_size / 1048576, 1)} MB"
            file_cat = category(file)

            if file.suffix.lower() == ".pdf" and ocr_mode != "native":
                report("ocr.started", file=file.name, message="Parsing PDF with OCR")
            for page, text in extract_text_units(file, ocr_mode):
                for chunk in split_text(text.strip(), CHUNK_SIZE, CHUNK_OVERLAP):
                    if not chunk.strip():
                        continue
                    citation_id = citation_id_for(relative_file, page, chunk)
                    chunks.append(chunk)
                    metas.append({
                        "citation_id": citation_id,
                        "file": relative_file,
                        "category": file_cat,
                        "chunk": chunk_number,
                        "page": page,
                        "upload_date": upload_date,
                        "file_size": file_size_str,
                        "text": chunk,
                    })
                    chunk_number += 1
        except ValueError as exc:
            skipped.append(str(exc))
        report(
            "parse.progress",
            completed=file_number,
            total=len(library_files),
            file=file.name,
            message="Extracting page-aware evidence",
        )
    if chunks:
        report("embedding.started", total=len(chunks), message="Embedding local evidence")
        save_index(chunks, metas)
        report("index.ready", total=len(chunks), message="Knowledge base is ready")
    else:
        global index, metadata
        with index_lock:
            index, metadata = None, []
    return {
        "files_added": moved,
        "chunks_indexed": len(chunks),
        "skipped": skipped,
        "ocr_mode": ocr_mode,
    }


IDENTITY_PATTERNS = [
    r"\bwho\s+(?:are|'re)\s+(?:you|u)\b",
    r"\bwho\s+(?:r|are)\s+(?:u|you)\b",
    r"\bwho\s+(?:created|developed|built|made|designed)\s*(?:you|u|lumina|this|this\s+app|this\s+assistant)?\b",
    r"\bwh(?:at|o)\s+(?:are\s+(?:you|u)|is\s+lumina\s*rag)\b",
    r"\bwho\s+(?:is|was)\s+(?:(?:you|ur|your)\s+creator|(?:you|ur|your)\s+developer|jeevan|jeevan\s+raj)\b",
    r"\bwho\s+(?:built|made)\s+(?:u|you|dis|this)\b",
    r"\b(?:ur|your)\s+(?:creator|developer|maker)\b",
    r"\btell\s+(?:me\s+)?about\s+(?:you|yourself|urself)\b",
]

CAPABILITY_PATTERNS = [
    r"\bwh?at\s+can\s+(?:you|u)\s+do\b",
    r"\bhow\s+does?\s+(?:this|lumina)\s+work\b",
    r"\bhelp\s+me\s+understand\s+wh?at\s+(?:you|u)\s+do\b",
    r"\bwh?at\s+(?:do|can)\s+(?:u|you)\s+do\b",
]

def check_system_question(question: str) -> str | None:
    q = question.strip().lower()
    if any(re.search(pat, q) for pat in IDENTITY_PATTERNS):
        return "I am Lumina RAG, a private local document AI assistant created and developed by Jeevan Raj M. I allow you to upload, index, and query your documents and file metadata with 100% privacy on your own machine."
    if any(re.search(pat, q) for pat in CAPABILITY_PATTERNS):
        return "I am Lumina RAG, a private document assistant developed by Jeevan Raj M. You can upload documents (PDF, DOCX, TXT, MD, CSV, etc.) into your local knowledge base, build a vector index, and ask questions about your documents or their metadata (such as upload dates, file sizes, and categories)."
    return None


def answer(question: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    if not question or not question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    system_answer = check_system_question(question)
    if system_answer:
        return system_answer, []

    with index_lock:
        active_index, active_metadata = index, metadata
    if active_index is None or not active_metadata:
        raise HTTPException(400, "No indexed documents. Put files in data/inbox and call /ingest first.")

    # Build a high-level library metadata inventory for the model
    unique_files: dict[str, dict[str, Any]] = {}
    for item in active_metadata:
        fname = item.get("file")
        if fname and fname not in unique_files:
            unique_files[fname] = {
                "file": fname,
                "category": item.get("category", "document"),
                "upload_date": item.get("upload_date", "Unknown"),
                "file_size": item.get("file_size", "Unknown"),
            }

    lib_summary_lines = [
        f"- File: {info['file']} | Type: {info['category'].upper()} | Size: {info['file_size']} | Uploaded/Modified: {info['upload_date']}"
        for info in unique_files.values()
    ]
    library_metadata_header = "INDEXED LIBRARY FILES & METADATA:\n" + "\n".join(lib_summary_lines)

    # Vector search for relevant evidence chunks
    vector = embed_texts([question])
    scores, positions = active_index.search(np.asarray(vector, dtype="float32"), min(top_k, len(active_metadata)))
    sources = []
    context_chunks = []
    for score, pos in zip(scores[0], positions[0]):
        if float(score) < MIN_RELEVANCE_SCORE:
            continue
        item = active_metadata[int(pos)]
        citation_label = item["citation_id"]
        page_label = f", page {item['page']}" if item.get("page") else ""
        upload_date = item.get("upload_date", "Unknown")
        file_size = item.get("file_size", "Unknown")

        chunk_header = f"[Evidence {citation_label}: File={item['file']}{page_label}, Uploaded/Modified={upload_date}, Size={file_size}]"
        context_chunks.append(f"{chunk_header}\n{item['text']}")
        sources.append({
            "citation_id": citation_label,
            "file": item["file"],
            "category": item.get("category", "document"),
            "page": item.get("page"),
            "upload_date": upload_date,
            "file_size": file_size,
            "excerpt": item["text"][:350],
            "score": round(float(score), 3),
        })

    full_context = f"{library_metadata_header}\n\nRELEVANT EVIDENCE CHUNKS:\n" + "\n\n".join(context_chunks)

    prompt = """You are Lumina RAG, a private local document AI assistant created and developed by Jeevan Raj M.
If asked who you are, who created or developed you, or what your identity is, identify yourself as Lumina RAG, developed by Jeevan Raj M. Never mention Qwen, Alibaba, or Alibaba Cloud.

%s

Answer the user's question accurately using both the INDEXED LIBRARY FILES & METADATA section and the RELEVANT EVIDENCE CHUNKS below.
If the user asks questions about file metadata (such as when a document was uploaded, modified, its file size, file type, or what files are in the library), answer directly using the file metadata provided.
If answering from evidence chunks, cite supporting evidence IDs in square brackets like [Evidence abc123]. Be concise, accurate, and direct.

CONTEXT:
%s

QUESTION: %s
ANSWER:""" % (build_memory_prompt_section(load_active_memories()), full_context, question)

    try:
        model_to_use = get_active_model()
        response = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": model_to_use, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}, timeout=180)
        response.raise_for_status()
        raw_answer = response.json()["response"].strip()
        clean_answer = sanitize_response_text(raw_answer)
        return clean_answer, sources
    except requests.RequestException as exc:
        raise HTTPException(503, f"Ollama is unavailable at {OLLAMA_URL}. Start it and pull {model_to_use}. ({exc})") from exc


def job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """Return only serializable job information to API and event consumers."""
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job.get("stage"),
        "message": job.get("message"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "result": job.get("result"),
        "error": job.get("error"),
        "progress": job.get("progress", {}),
    }


def encode_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def publish_event(event: str, **payload: Any) -> None:
    message = encode_sse(event, payload)
    with event_lock:
        subscribers = list(event_subscribers)
    for subscriber in subscribers:
        try:
            subscriber.put_nowait(message)
        except Full:
            try:
                subscriber.get_nowait()
            except Empty:
                pass
            try:
                subscriber.put_nowait(message)
            except Full:
                pass


def prune_ingest_jobs() -> None:
    """Keep job history bounded so long-running servers do not leak memory."""
    if len(ingest_jobs) <= MAX_INGEST_JOBS:
        return
    finished = sorted(
        (
            (job_id, job.get("updated_at", 0))
            for job_id, job in ingest_jobs.items()
            if job.get("status") not in {"queued", "running"}
        ),
        key=lambda item: item[1],
    )
    for job_id, _ in finished[: len(ingest_jobs) - MAX_INGEST_JOBS]:
        ingest_jobs.pop(job_id, None)


def update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    with ingest_lock:
        job = ingest_jobs[job_id]
        job.update(changes)
        job["updated_at"] = int(time.time())
        snapshot = job_snapshot(job)
    if changes.get("status") in {"completed", "failed"}:
        prune_ingest_jobs()
    return snapshot


def safe_filename(name: str) -> str:
    """Strip path separators so uploads cannot escape the inbox folder."""
    cleaned = Path(name).name.strip()
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(400, "Invalid file name.")
    return cleaned


def run_ingestion_job(job_id: str, rebuild: bool, ocr_mode: str) -> None:
    global active_ingest_job_id
    update_job(job_id, status="running", stage="starting", message="Preparing local ingestion")
    publish_event("ingest.started", job=job_snapshot(ingest_jobs[job_id]))

    def report(event: str, details: dict[str, Any]) -> None:
        job = update_job(
            job_id,
            stage=event,
            message=details.get("message", event),
            progress={key: value for key, value in details.items() if key != "message"},
        )
        publish_event(event, job=job, **details)

    try:
        result = organize_and_index(rebuild, report, ocr_mode)
        job = update_job(
            job_id,
            status="completed",
            stage="completed",
            message="Knowledge base is ready",
            result=result,
        )
        publish_event("ingest.completed", job=job)
    except Exception as exc:
        job = update_job(
            job_id,
            status="failed",
            stage="failed",
            message="Ingestion failed",
            error=str(exc),
        )
        publish_event("ingest.failed", job=job)
    finally:
        with ingest_lock:
            if active_ingest_job_id == job_id:
                active_ingest_job_id = None


def start_ingestion(rebuild: bool, ocr_mode: str) -> tuple[dict[str, Any], bool]:
    global active_ingest_job_id
    with ingest_lock:
        if active_ingest_job_id:
            active_job = ingest_jobs[active_ingest_job_id]
            if active_job["status"] in {"queued", "running"}:
                return job_snapshot(active_job), False
        now = int(time.time())
        job: dict[str, Any] = {
            "id": uuid4().hex,
            "status": "queued",
            "stage": "queued",
            "message": "Waiting to start",
            "created_at": now,
            "updated_at": now,
            "progress": {},
            "result": None,
            "error": None,
        }
        ingest_jobs[job["id"]] = job
        active_ingest_job_id = job["id"]
        snapshot = job_snapshot(job)
    Thread(target=run_ingestion_job, args=(job["id"], rebuild, ocr_mode), daemon=True).start()
    return snapshot, True


# Startup is handled by the lifespan context manager above.


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    with index_lock:
        indexed_chunks = len(metadata)
    with ingest_lock:
        active_job = ingest_jobs.get(active_ingest_job_id) if active_ingest_job_id else None
    current_model = get_active_model()
    inbox_files = [path for path in INBOX.rglob("*") if path.is_file() and not path.name.startswith(".")]
    library_files = [path for path in LIBRARY.rglob("*") if path.is_file() and not path.name.startswith(".")]
    return {
        "status": "ok",
        "model": current_model,
        "model_display": brand_model_name(current_model),
        "embedding_model": EMBEDDING_MODEL,
        "ocr_mode": OCR_MODE,
        "unlimited_ocr_configured": bool(UNLIMITED_OCR_URL),
        "indexed_chunks": indexed_chunks,
        "inbox_files": len(inbox_files),
        "library_files": len(library_files),
        "data_dir": str(BASE),
        "active_ingestion": job_snapshot(active_job) if active_job else None,
    }


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Accept browser uploads and place them in the local inbox folder."""
    if not files:
        raise HTTPException(400, "No files were uploaded.")
    saved, skipped = [], []
    for upload in files:
        filename = safe_filename(upload.filename or "document")
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in TEXT_EXTENSIONS:
            skipped.append(f"Skipped {filename}: unsupported file type")
            continue
        destination = INBOX / filename
        counter = 1
        while destination.exists():
            destination = INBOX / f"{Path(filename).stem}_{counter}{Path(filename).suffix}"
            counter += 1
        size = 0
        too_large = False
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    too_large = True
                    break
                handle.write(chunk)
        if too_large:
            destination.unlink(missing_ok=True)
            skipped.append(f"Skipped {filename}: exceeds {MAX_FILE_BYTES // 1024 // 1024} MB limit")
            continue
        saved.append(destination.name)
    if not saved and skipped:
        raise HTTPException(400, "; ".join(skipped))
    inbox_valid = [path for path in INBOX.rglob("*") if path.is_file() and not path.name.startswith(".")]
    return {"saved": saved, "skipped": skipped, "inbox_total": len(inbox_valid)}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict[str, Any]:
    inbox_files = [f for f in INBOX.rglob("*") if f.is_file() and not f.name.startswith(".")]
    library_files = [f for f in LIBRARY.rglob("*") if f.is_file() and not f.name.startswith(".")]
    if not inbox_files and not library_files:
        raise HTTPException(400, "No documents found to ingest. Upload files to your inbox first.")
    job, started = start_ingestion(payload.rebuild, payload.ocr_mode or OCR_MODE)
    return {"started": started, "job": job}


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str) -> dict[str, Any]:
    with ingest_lock:
        job = ingest_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Ingestion job not found")
        return job_snapshot(job)


@app.get("/events")
def events() -> StreamingResponse:
    def stream() -> Iterator[str]:
        subscriber: Queue[str] = Queue(maxsize=100)
        with event_lock:
            event_subscribers.append(subscriber)
        try:
            yield "retry: 3000\n\n"
            with ingest_lock:
                active_job = ingest_jobs.get(active_ingest_job_id) if active_ingest_job_id else None
            if active_job:
                yield encode_sse("ingest.status", {"job": job_snapshot(active_job)})
            while True:
                try:
                    yield subscriber.get(timeout=15)
                except Empty:
                    yield ": keep-alive\n\n"
        finally:
            with event_lock:
                if subscriber in event_subscribers:
                    event_subscribers.remove(subscriber)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask")
def ask(payload: AskRequest) -> dict[str, Any]:
    if payload.model:
        set_active_model(payload.model)
    text, sources = answer(payload.question, payload.top_k)
    # Auto-save to session history
    session = add_session(payload.question, text, sources, payload.model)
    return {"answer": text, "sources": sources, "session_id": session["id"]}


@app.get("/sources/{citation_id}")
def source(citation_id: str) -> dict[str, Any]:
    """Return the exact retained evidence behind an answer citation."""
    with index_lock:
        item = next((entry for entry in metadata if entry.get("citation_id") == citation_id), None)
    if not item:
        raise HTTPException(404, "Citation not found. Re-run the query after indexing.")
    file_url = f"/files/{quote(item['file'], safe='/')}"
    return {
        "citation_id": item["citation_id"],
        "file": item["file"],
        "category": item["category"],
        "page": item.get("page"),
        "chunk": item["chunk"],
        "text": item["text"],
        "file_url": file_url,
    }


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


class FileActionRequest(BaseModel):
    paths: list[str]


def load_trash_manifest() -> dict[str, dict[str, Any]]:
    manifest_path = TRASH / "manifest.json"
    if manifest_path.is_file():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_trash_manifest(data: dict[str, dict[str, Any]]) -> None:
    manifest_path = TRASH / "manifest.json"
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def auto_purge_old_trash() -> None:
    """Purge files from trash older than TRASH_RETENTION_DAYS (30 days)."""
    manifest = load_trash_manifest()
    now = time.time()
    cutoff = now - (TRASH_RETENTION_DAYS * 86400)
    changed = False
    for trash_key, info in list(manifest.items()):
        deleted_at = info.get("deleted_at", 0)
        if deleted_at < cutoff:
            trash_file = TRASH / trash_key
            trash_file.unlink(missing_ok=True)
            del manifest[trash_key]
            changed = True
    if changed:
        save_trash_manifest(manifest)


@app.get("/files")
def files() -> list[dict[str, Any]]:
    auto_purge_old_trash()
    result = []
    try:
        for f in LIBRARY.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(LIBRARY).as_posix()
                stat = f.stat()
                mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                size_kb = round(stat.st_size / 1024, 1)
                size_str = f"{size_kb} KB" if stat.st_size < 1048576 else f"{round(stat.st_size / 1048576, 1)} MB"
                result.append({
                    "path": rel_path,
                    "filename": f.name,
                    "category": category(f),
                    "size_bytes": stat.st_size,
                    "size_human": size_str,
                    "modified_time": int(stat.st_mtime),
                    "upload_date": mtime_str,
                })
    except (FileNotFoundError, OSError):
        pass
    return result


@app.post("/files/delete")
def delete_files(payload: FileActionRequest) -> dict[str, Any]:
    """Soft delete files by moving them from LIBRARY to TRASH with a 30-day retention manifest."""
    manifest = load_trash_manifest()
    deleted = []
    failed = []
    now = int(time.time())

    for rel_path in payload.paths:
        target = (LIBRARY / rel_path).resolve()
        if not target.is_relative_to(LIBRARY) or not target.is_file():
            failed.append(f"{rel_path}: Not found")
            continue
        try:
            trash_key = sha256(rel_path.encode("utf-8")).hexdigest()[:16] + "_" + target.name
            trash_target = TRASH / trash_key
            shutil.move(str(target), str(trash_target))
            manifest[trash_key] = {
                "original_path": rel_path,
                "filename": target.name,
                "category": category(target),
                "deleted_at": now,
                "size_bytes": trash_target.stat().st_size if trash_target.exists() else 0,
            }
            deleted.append(rel_path)
        except Exception as exc:
            failed.append(f"{rel_path}: {exc}")

    save_trash_manifest(manifest)
    return {"deleted": deleted, "failed": failed, "trash_count": len(manifest)}


@app.get("/files/trash")
def list_trash() -> list[dict[str, Any]]:
    auto_purge_old_trash()
    manifest = load_trash_manifest()
    now = time.time()
    items = []
    for trash_key, info in manifest.items():
        trash_file = TRASH / trash_key
        if not trash_file.is_file():
            continue
        deleted_at = info.get("deleted_at", now)
        elapsed_days = (now - deleted_at) / 86400
        days_remaining = max(0, int(TRASH_RETENTION_DAYS - elapsed_days))
        deleted_date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deleted_at))
        size_bytes = info.get("size_bytes", 0)
        size_kb = round(size_bytes / 1024, 1)
        size_str = f"{size_kb} KB" if size_bytes < 1048576 else f"{round(size_bytes / 1048576, 1)} MB"
        items.append({
            "trash_key": trash_key,
            "original_path": info.get("original_path", trash_key),
            "filename": info.get("filename", trash_key),
            "category": info.get("category", "document"),
            "deleted_at": deleted_at,
            "deleted_date": deleted_date_str,
            "days_remaining": days_remaining,
            "size_bytes": size_bytes,
            "size_human": size_str,
        })
    return items


@app.post("/files/trash/restore")
def restore_trash(payload: FileActionRequest) -> dict[str, Any]:
    manifest = load_trash_manifest()
    restored = []
    failed = []
    for trash_key in payload.paths:
        info = manifest.get(trash_key)
        if not info:
            failed.append(f"{trash_key}: Not in trash manifest")
            continue
        trash_file = TRASH / trash_key
        if not trash_file.is_file():
            failed.append(f"{trash_key}: File missing from trash storage")
            del manifest[trash_key]
            continue
        try:
            target_rel = info.get("original_path", trash_file.name)
            target = (LIBRARY / target_rel).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(trash_file), str(target))
            del manifest[trash_key]
            restored.append(target_rel)
        except Exception as exc:
            failed.append(f"{trash_key}: {exc}")
    save_trash_manifest(manifest)
    return {"restored": restored, "failed": failed}


@app.delete("/files/trash/empty")
def empty_trash() -> dict[str, Any]:
    manifest = load_trash_manifest()
    purged_count = 0
    for trash_key in list(manifest.keys()):
        trash_file = TRASH / trash_key
        trash_file.unlink(missing_ok=True)
        del manifest[trash_key]
        purged_count += 1
    save_trash_manifest(manifest)
    return {"purged": purged_count}


@app.get("/files/{path:path}")
def download(path: str) -> FileResponse:
    file = (LIBRARY / path).resolve()
    if not file.is_relative_to(LIBRARY) or not file.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(file, filename=file.name)


@app.get("/models")
def models_list() -> dict[str, Any]:
    """List available Ollama chat models for the model selector UI."""
    current = get_active_model()
    available = list_ollama_models()
    return {
        "active": current,
        "active_display": brand_model_name(current),
        "models": available,
    }


class SwitchModelRequest(BaseModel):
    model: str = Field(min_length=1)


@app.put("/models/active")
def switch_model(payload: SwitchModelRequest) -> dict[str, Any]:
    """Switch the active LLM model used for answering questions."""
    available = list_ollama_models()
    model_ids = {m["id"] for m in available}
    if payload.model not in model_ids:
        raise HTTPException(404, f"Model '{payload.model}' is not available in Ollama. Run: ollama pull {payload.model}")
    set_active_model(payload.model)
    return {
        "active": payload.model,
        "active_display": brand_model_name(payload.model),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Adaptive Memory System endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MemoryCreateRequest(BaseModel):
    type: Literal["user_fact", "correction", "preference"] = "correction"
    content: str = Field(min_length=1, max_length=1000)
    original_feedback: str | None = None
    question_context: str | None = None


class MemoryToggleRequest(BaseModel):
    active: bool


@app.get("/memory")
def list_memories() -> list[dict[str, Any]]:
    """List all memories (active and inactive)."""
    return load_all_memories()


@app.post("/memory")
def create_memory(payload: MemoryCreateRequest) -> dict[str, Any]:
    """Add a new memory entry."""
    entry = add_memory(
        payload.type, payload.content,
        payload.original_feedback, payload.question_context,
    )
    return {"created": True, "memory": entry, "total": len(load_all_memories())}


@app.put("/memory/{memory_id}")
def toggle_memory(memory_id: str, payload: MemoryToggleRequest) -> dict[str, Any]:
    """Toggle a memory active/inactive."""
    memories = load_all_memories()
    for m in memories:
        if m["id"] == memory_id:
            m["active"] = payload.active
            save_all_memories(memories)
            return {"updated": True, "memory": m}
    raise HTTPException(404, "Memory not found")


@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: str) -> dict[str, Any]:
    """Delete a specific memory."""
    memories = load_all_memories()
    original_count = len(memories)
    memories = [m for m in memories if m["id"] != memory_id]
    if len(memories) == original_count:
        raise HTTPException(404, "Memory not found")
    save_all_memories(memories)
    return {"deleted": True, "remaining": len(memories)}


@app.delete("/memory")
def clear_memories() -> dict[str, Any]:
    """Clear all memories."""
    save_all_memories([])
    return {"cleared": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feedback endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FeedbackRequest(BaseModel):
    type: Literal["thumbs_up", "thumbs_down", "bug", "suggestion", "feature"]
    question: str | None = None
    answer: str | None = None
    comment: str | None = None
    reason: str | None = None
    correction: str | None = None
    model: str | None = None
    session_id: str | None = None


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest) -> dict[str, Any]:
    """Submit feedback and optionally save a correction as a memory."""
    now = int(time.time())
    entry: dict[str, Any] = {
        "id": f"fb_{uuid4().hex[:10]}",
        "type": payload.type,
        "question": payload.question,
        "answer": payload.answer,
        "comment": payload.comment,
        "reason": payload.reason,
        "correction": payload.correction,
        "model": payload.model or get_active_model(),
        "session_id": payload.session_id,
        "created_at": now,
    }
    filename = f"{now}_{payload.type}.json"
    (FEEDBACK_DIR / filename).write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    # Auto-save correction as memory if provided
    memory_entry = None
    if payload.correction and payload.correction.strip():
        memory_entry = add_memory(
            memory_type="correction",
            content=payload.correction.strip(),
            original_feedback=payload.comment,
            question_context=payload.question,
        )

    return {"saved": True, "feedback_id": entry["id"], "memory_created": memory_entry}


@app.get("/feedback")
def list_feedback() -> list[dict[str, Any]]:
    """List all feedback entries."""
    entries: list[dict[str, Any]] = []
    for path in sorted(FEEDBACK_DIR.glob("*.json"), reverse=True):
        try:
            entries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return entries


@app.get("/feedback/stats")
def feedback_stats() -> dict[str, Any]:
    """Aggregated feedback statistics."""
    entries = list_feedback()
    total = len(entries)
    thumbs_up = sum(1 for e in entries if e.get("type") == "thumbs_up")
    thumbs_down = sum(1 for e in entries if e.get("type") == "thumbs_down")
    bugs = sum(1 for e in entries if e.get("type") == "bug")
    suggestions = sum(1 for e in entries if e.get("type") == "suggestion")
    features = sum(1 for e in entries if e.get("type") == "feature")
    positive_rate = round(thumbs_up / max(thumbs_up + thumbs_down, 1) * 100, 1)
    return {
        "total": total,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "bugs": bugs,
        "suggestions": suggestions,
        "features": features,
        "positive_rate": positive_rate,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chat Session History endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/sessions")
def get_sessions() -> list[dict[str, Any]]:
    """List recent Q&A sessions (newest first)."""
    return list(reversed(load_sessions()))


@app.delete("/sessions")
def clear_sessions() -> dict[str, Any]:
    """Clear all session history."""
    save_sessions([])
    return {"cleared": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Unified Search endpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/search")
def unified_search(q: str = "") -> dict[str, Any]:
    """Search across documents, sessions, and knowledge chunks."""
    query = q.strip().lower()
    if not query:
        return {"documents": [], "sessions": [], "chunks": []}

    # Search documents by filename
    doc_results: list[dict[str, Any]] = []
    try:
        for f in LIBRARY.rglob("*"):
            if f.is_file() and query in f.name.lower():
                stat = f.stat()
                doc_results.append({
                    "path": f.relative_to(LIBRARY).as_posix(),
                    "filename": f.name,
                    "category": category(f),
                    "size_human": f"{round(stat.st_size / 1024, 1)} KB" if stat.st_size < 1048576 else f"{round(stat.st_size / 1048576, 1)} MB",
                })
                if len(doc_results) >= 5:
                    break
    except (FileNotFoundError, OSError):
        pass

    # Search sessions by question text
    session_results: list[dict[str, Any]] = []
    for s in reversed(load_sessions()):
        if query in s.get("question", "").lower() or query in s.get("answer", "").lower():
            session_results.append(s)
            if len(session_results) >= 5:
                break

    # Search indexed chunks by keyword match
    chunk_results: list[dict[str, Any]] = []
    with index_lock:
        active_metadata = metadata
    for item in active_metadata:
        text = item.get("text", "")
        if query in text.lower():
            chunk_results.append({
                "file": item.get("file", ""),
                "page": item.get("page"),
                "excerpt": text[:200],
                "citation_id": item.get("citation_id", ""),
            })
            if len(chunk_results) >= 5:
                break

    return {"documents": doc_results, "sessions": session_results, "chunks": chunk_results}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OCR Settings endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/ocr/settings")
def ocr_settings() -> dict[str, Any]:
    """Return current OCR configuration."""
    return {
        "mode": OCR_MODE,
        "unlimited_ocr_configured": bool(UNLIMITED_OCR_URL),
        "unlimited_ocr_url": UNLIMITED_OCR_URL or None,
        "model": UNLIMITED_OCR_MODEL,
        "dpi": UNLIMITED_OCR_DPI,
        "max_pages": UNLIMITED_OCR_MAX_PAGES,
        "max_tokens": UNLIMITED_OCR_MAX_TOKENS,
        "timeout_seconds": UNLIMITED_OCR_TIMEOUT_SECONDS,
    }
