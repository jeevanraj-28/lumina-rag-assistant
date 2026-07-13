#!/usr/bin/env bash
# =============================================================================
#  Lumina RAG — Startup Script
#  Usage: bash start.sh [--rebuild] [--no-browser]
#
#  Options:
#    --rebuild      Force a full FAISS index rebuild on startup via POST /ingest
#    --no-browser   Skip auto-opening the browser after the server is ready
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[Lumina]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[  OK  ]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[ WARN ]${RESET} $*"; }
error()   { echo -e "${RED}${BOLD}[ERROR ]${RESET} $*" >&2; }

# ── Banner ─────────────────────────────────────────────────────────────────────
echo -e "${BOLD}${CYAN}"
echo "  ██╗     ██╗   ██╗███╗   ███╗██╗███╗   ██╗ █████╗     ██████╗  █████╗  ██████╗ "
echo "  ██║     ██║   ██║████╗ ████║██║████╗  ██║██╔══██╗    ██╔══██╗██╔══██╗██╔════╝ "
echo "  ██║     ██║   ██║██╔████╔██║██║██╔██╗ ██║███████║    ██████╔╝███████║██║  ███╗"
echo "  ██║     ██║   ██║██║╚██╔╝██║██║██║╚██╗██║██╔══██║    ██╔══██╗██╔══██║██║   ██║"
echo "  ███████╗╚██████╔╝██║ ╚═╝ ██║██║██║ ╚████║██║  ██║    ██║  ██║██║  ██║╚██████╔╝"
echo "  ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ "
echo -e "${RESET}"
echo -e "  ${BOLD}Local Intelligence — 100% Private, 100% Offline${RESET}"
echo ""

# ── Argument parsing ──────────────────────────────────────────────────────────
REBUILD=false
OPEN_BROWSER=true

for arg in "$@"; do
  case $arg in
    --rebuild)     REBUILD=true ;;
    --no-browser)  OPEN_BROWSER=false ;;
    *)             warn "Unknown argument: $arg" ;;
  esac
done

# ── Config (override with .env) ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Load .env if it exists, otherwise copy from .env.example
if [[ -f "$ENV_FILE" ]]; then
  info "Loading environment from .env"
  set -a; source "$ENV_FILE"; set +a
elif [[ -f "$SCRIPT_DIR/.env.example" ]]; then
  warn ".env not found — copying from .env.example (review and edit as needed)"
  cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
  set -a; source "$ENV_FILE"; set +a
fi

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
OLLAMA_EMBEDDING_MODEL="${OLLAMA_EMBEDDING_MODEL:-all-minilm}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
RAG_DATA_DIR="${RAG_DATA_DIR:-data}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
OLLAMA_PID=""
UVICORN_PID=""

cleanup() {
  echo ""
  info "Shutting down Lumina RAG..."
  [[ -n "$UVICORN_PID" ]] && kill "$UVICORN_PID" 2>/dev/null && info "FastAPI server stopped."
  [[ -n "$OLLAMA_PID" ]]  && kill "$OLLAMA_PID"  2>/dev/null && info "Ollama stopped."
  success "Goodbye! 👋"
}
trap cleanup EXIT INT TERM

# ── WSL / Windows PATH fix ────────────────────────────────────────────────────
# WSL (and Git Bash) don't inherit the Windows PATH where ollama.exe lives.
# Scan common Windows drive mount points to find and inject it.
if ! command -v ollama &>/dev/null; then
  for p in \
    /mnt/c/Users/*/AppData/Local/Programs/Ollama \
    /mnt/c/Program\ Files/Ollama \
    /c/Users/*/AppData/Local/Programs/Ollama; do
    if [[ -f "$p/ollama.exe" ]]; then
      export PATH="$p:$PATH"
      info "Added Ollama to PATH: $p"
      break
    fi
  done
fi


# ── Step 1: Check system dependencies ─────────────────────────────────────────
info "Checking system dependencies..."

if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  error "Python 3.11+ is required but not found. Install from https://python.org"
  exit 1
fi

PYTHON=$(command -v python3 2>/dev/null || command -v python)
PYTHON_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python $PYTHON_VERSION found at $PYTHON"

# Detect ollama or ollama.exe (WSL)
OLLAMA_CMD=""
if command -v ollama &>/dev/null; then
  OLLAMA_CMD="ollama"
elif command -v ollama.exe &>/dev/null; then
  OLLAMA_CMD="ollama.exe"
fi

if [[ -z "$OLLAMA_CMD" ]]; then
  error "Ollama is not installed or not in PATH."
  error "Install from: https://ollama.com/"
  exit 1
fi
success "Ollama found: $OLLAMA_CMD"

# ── Step 2: Create data directories ───────────────────────────────────────────
info "Ensuring data directories exist..."
mkdir -p \
  "$SCRIPT_DIR/$RAG_DATA_DIR/inbox" \
  "$SCRIPT_DIR/$RAG_DATA_DIR/library" \
  "$SCRIPT_DIR/$RAG_DATA_DIR/index"
success "Data directories ready."

# ── Step 3: Python virtual environment ────────────────────────────────────────
# Detect whether venv uses Windows-style Scripts/ (WSL / Git Bash running on
# a Windows-created .venv) or Unix-style bin/ (native Linux .venv).
VENV_PYTHON=""
VENV_PIP=""
if [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
  # Windows .venv accessed from WSL — use .exe directly (WSL2 interop)
  VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
  VENV_PIP="$VENV_DIR/Scripts/pip.exe"
  export PATH="$VENV_DIR/Scripts:$PATH"
elif [[ -f "$VENV_DIR/bin/python" ]]; then
  # Native Linux .venv
  VENV_PYTHON="$VENV_DIR/bin/python"
  VENV_PIP="$VENV_DIR/bin/pip"
  export PATH="$VENV_DIR/bin:$PATH"
fi

if [[ -n "$VENV_PYTHON" ]]; then
  PYTHON="$VENV_PYTHON"
  PYTHON_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  info "Using venv Python $PYTHON_VERSION at $PYTHON"
else
  info "No existing venv found — creating one at .venv ..."
  "$PYTHON" -m venv "$VENV_DIR"
  if [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
    VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
    VENV_PIP="$VENV_DIR/Scripts/pip.exe"
    export PATH="$VENV_DIR/Scripts:$PATH"
  else
    VENV_PYTHON="$VENV_DIR/bin/python"
    VENV_PIP="$VENV_DIR/bin/pip"
    export PATH="$VENV_DIR/bin:$PATH"
  fi
  PYTHON="$VENV_PYTHON"
  success "Virtual environment created."
fi
success "Virtual environment ready ($("$PYTHON" --version 2>&1))."

# ── Step 4: Install / sync Python dependencies ────────────────────────────────
info "Checking Python dependencies..."
if ! "$VENV_PIP" show fastapi &>/dev/null 2>&1; then
  info "Installing dependencies from requirements.txt..."
  "$VENV_PIP" install --quiet -r "$SCRIPT_DIR/requirements.txt"
  success "Dependencies installed."
else
  info "Dependencies already installed."
fi

# ── Step 5: Start / connect to Ollama server ──────────────────────────────────
# In WSL2, Windows processes bind to the Windows loopback. `localhost` is
# forwarded via WSL2 mirrored networking; the raw 127.0.0.1 IP is not.
OLLAMA_HEALTH_URL="${OLLAMA_BASE_URL/127.0.0.1/localhost}"

info "Checking Ollama at $OLLAMA_HEALTH_URL ..."

if curl -sf --max-time 3 "$OLLAMA_HEALTH_URL/api/tags" &>/dev/null; then
  success "Ollama is already running — skipping start."
else
  info "Starting Ollama server..."
  "$OLLAMA_CMD" serve &>/dev/null &
  OLLAMA_PID=$!
  info "Ollama server started (PID: $OLLAMA_PID) — waiting for it to be ready..."

  WAIT=0
  until curl -sf --max-time 2 "$OLLAMA_HEALTH_URL/api/tags" &>/dev/null; do
    sleep 1
    WAIT=$((WAIT + 1))
    if [[ $WAIT -eq 10 ]]; then
      info "Still waiting for Ollama... (if running on Windows, ensure it is started)"
    fi
    if [[ $WAIT -gt 60 ]]; then
      error "Ollama did not become ready after 60 seconds."
      error "If running from WSL, start Ollama on Windows first, then re-run this script."
      exit 1
    fi
  done
  success "Ollama is ready."
fi


# ── Step 6: Pull required models ──────────────────────────────────────────────
pull_model_if_missing() {
  local model="$1"
  if "$OLLAMA_CMD" list 2>/dev/null | grep -q "^${model}"; then
    success "Model '${model}' already available."
  else
    info "Pulling model '${model}' (this may take a while on first run)..."
    "$OLLAMA_CMD" pull "$model"
    success "Model '${model}' ready."
  fi
}

pull_model_if_missing "$OLLAMA_MODEL"
pull_model_if_missing "$OLLAMA_EMBEDDING_MODEL"

# ── Step 7: Start FastAPI / Uvicorn ───────────────────────────────────────────
info "Starting Lumina RAG server on http://${APP_HOST}:${APP_PORT} ..."
cd "$SCRIPT_DIR"

"$PYTHON" -m uvicorn app.main:app \
  --host "$APP_HOST" \
  --port "$APP_PORT" \
  --reload \
  --log-level info &
UVICORN_PID=$!

# Wait for the server health check
info "Waiting for the API server to be ready..."
WAIT=0
until curl -sf "http://127.0.0.1:${APP_PORT}/health" &>/dev/null; do
  sleep 1
  WAIT=$((WAIT + 1))
  if [[ $WAIT -gt 30 ]]; then
    warn "Server health check timed out — it may still be starting up."
    break
  fi
done

echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "  ${BOLD}✅  Lumina RAG is running!${RESET}"
echo -e "  🌐  Web UI  : ${CYAN}http://localhost:${APP_PORT}${RESET}"
echo -e "  📖  API Docs: ${CYAN}http://localhost:${APP_PORT}/docs${RESET}"
echo -e "  🤖  Model   : ${BOLD}${OLLAMA_MODEL}${RESET}"
echo -e "  📁  Data Dir: ${BOLD}${RAG_DATA_DIR}/${RESET}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# ── Step 8 (optional): Rebuild FAISS index ────────────────────────────────────
if [[ "$REBUILD" == "true" ]]; then
  info "Triggering FAISS index rebuild (--rebuild flag set)..."
  sleep 2
  REBUILD_RESP=$(curl -sf -X POST "http://127.0.0.1:${APP_PORT}/ingest" \
    -H "Content-Type: application/json" \
    -d '{"rebuild": true}' || echo "FAILED")
  if [[ "$REBUILD_RESP" == "FAILED" ]]; then
    warn "Index rebuild request failed — trigger it manually via the UI."
  else
    success "Index rebuild triggered."
  fi
fi

# ── Step 9 (optional): Open browser ───────────────────────────────────────────
if [[ "$OPEN_BROWSER" == "true" ]]; then
  sleep 1
  if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:${APP_PORT}" &>/dev/null &
  elif command -v open &>/dev/null; then
    open "http://localhost:${APP_PORT}" &>/dev/null &
  elif command -v start &>/dev/null; then
    start "http://localhost:${APP_PORT}" &>/dev/null &
  fi
fi

info "Press ${BOLD}Ctrl+C${RESET} to stop all services."
echo ""

# ── Keep running until Ctrl+C ─────────────────────────────────────────────────
wait "$UVICORN_PID"
