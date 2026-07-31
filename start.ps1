# Lumina RAG - Windows PowerShell Startup Script
Param(
    [switch]$Rebuild,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Lumina RAG - Local Document AI Assistant (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Load .env if present
$EnvFile = Join-Path $ScriptDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+?)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

$OllamaModel = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "qwen2.5:3b" }
$EmbeddingModel = if ($env:OLLAMA_EMBEDDING_MODEL) { $env:OLLAMA_EMBEDDING_MODEL } else { "all-minilm" }
$AppPort = if ($env:APP_PORT) { $env:APP_PORT } else { "8000" }

# 1. Check Ollama Desktop / Service
Write-Host "`n[Lumina] Checking Ollama service at http://localhost:11434 ..." -ForegroundColor Cyan
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 -ErrorAction Stop
    Write-Host "[  OK  ] Ollama Desktop is running and reachable!" -ForegroundColor Green
} catch {
    Write-Host "[ WARN ] Could not reach Ollama at http://localhost:11434." -ForegroundColor Yellow
    Write-Host "[ WARN ] Please make sure Ollama Desktop is launched from your Start Menu." -ForegroundColor Yellow
}

# 2. Check Virtual Environment
$VenvPython = ".\.venv\Scripts\python.exe"
$VenvPip = ".\.venv\Scripts\pip.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "`n[Lumina] Virtual environment not found. Creating .venv ..." -ForegroundColor Cyan
    python -m venv .venv
}

# 3. Install dependencies
Write-Host "[Lumina] Checking Python dependencies..." -ForegroundColor Cyan
& $VenvPython -m pip install -q -r requirements.txt
Write-Host "[  OK  ] Dependencies ready." -ForegroundColor Green

# 4. Pull models if missing
function Ensure-OllamaModel($Model) {
    try {
        $list = ollama list 2>$null
        if ($list -match [regex]::Escape($Model)) {
            Write-Host "[  OK  ] Model '$Model' already available." -ForegroundColor Green
        } else {
            Write-Host "[Lumina] Pulling model '$Model' (first run may take a while)..." -ForegroundColor Cyan
            ollama pull $Model
            Write-Host "[  OK  ] Model '$Model' ready." -ForegroundColor Green
        }
    } catch {
        Write-Host "[ WARN ] Could not check/pull model '$Model' via CLI." -ForegroundColor Yellow
    }
}
Ensure-OllamaModel $OllamaModel
Ensure-OllamaModel $EmbeddingModel

$DataDir = ".\data"
New-Item -ItemType Directory -Force -Path "$DataDir\inbox", "$DataDir\library", "$DataDir\index", "$DataDir\trash", "$DataDir\memory", "$DataDir\feedback", "$DataDir\sessions" | Out-Null
Write-Host "[  OK  ] Data directories verified." -ForegroundColor Green

# 6. Launch Application
Write-Host "`n[Lumina] Starting Lumina RAG on http://localhost:$AppPort ..." -ForegroundColor Cyan

$UvicornJob = Start-Job -ScriptBlock {
    param($Python, $Port)
    Set-Location $using:ScriptDir
    & $Python -m uvicorn app.main:app --host 0.0.0.0 --port $Port --reload 2>&1
} -ArgumentList (Resolve-Path $VenvPython), $AppPort

# Wait for health check
$Ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$AppPort/health" -TimeoutSec 2 -ErrorAction Stop
        $Ready = $true
        break
    } catch { }
}
if ($Ready) {
    Write-Host "[  OK  ] Server is ready at http://localhost:$AppPort" -ForegroundColor Green
} else {
    Write-Host "[ WARN ] Server health check timed out - it may still be starting." -ForegroundColor Yellow
}

if ($Rebuild) {
    Write-Host "[Lumina] Triggering FAISS index rebuild..." -ForegroundColor Yellow
    try {
        $body = '{"rebuild": true}'
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$AppPort/ingest" -Method POST -ContentType "application/json" -Body $body -TimeoutSec 10
        Write-Host "[  OK  ] Index rebuild triggered." -ForegroundColor Green
    } catch {
        Write-Host "[ WARN ] Index rebuild request failed - trigger it manually via the UI." -ForegroundColor Yellow
    }
}

if (-not $NoBrowser) {
    Start-Process "http://localhost:$AppPort"
}

Write-Host "`nPress Ctrl+C to stop Lumina RAG.`n" -ForegroundColor Cyan
try {
    Receive-Job -Job $UvicornJob -Wait
} finally {
    Stop-Job -Job $UvicornJob -ErrorAction SilentlyContinue
    Remove-Job -Job $UvicornJob -ErrorAction SilentlyContinue
}
