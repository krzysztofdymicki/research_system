# Research System

AI-assisted workflow for discovering academic papers from arXiv and CORE, saving metadata to SQLite, downloading PDFs reliably (with landing-page salvage), extracting Markdown from PDFs, and scoring relevance with a local LLM (LM Studio friendly). Comes with a simple Tkinter GUI.

## Features

- Search providers: arXiv, CORE (v3 API)
- Reliable PDF downloads with salvage from landing pages
- PDF → Markdown via `pymupdf4llm`
- SQLite persistence: publications + search_results
- LLM relevance scoring (title+abstract vs original query)
- GUI: search, results browsing, run AI, show-only-kept, download/extract kept, reset controls
  

## Install

```powershell
pip install -e .
```

Python 3.8+ required.

## Configure

Environment variables (optional but recommended):

- `CORE_API_KEY`: token for CORE v3 API
- `RS_LLM_ENDPOINT`: OpenAI-compatible chat completions endpoint (default `http://localhost:1234/v1/chat/completions`)
- `RS_LLM_MODEL`: default `google/gemma-3-12b`
- `RS_LLM_TEMPERATURE`: default `0.2`
- `RS_LLM_MAX_TOKENS`: default `256`
- `RS_RESET_ON_START`: `1|true|yes` to reset DB and `papers/` at next GUI start
- `RS_SALVAGE_HEAD_TIMEOUT`: default `20`; `RS_SALVAGE_GET_TIMEOUT`: `60`; `RS_SALVAGE_MAX_CANDIDATES`: `5`

Quick set in PowerShell:

```powershell
$env:CORE_API_KEY = 'your_core_api_key'
$env:RS_LLM_ENDPOINT = 'http://localhost:1234/v1/chat/completions'
$env:RS_LLM_MODEL = 'google/gemma-3-12b'
```

## Run

GUI:

```powershell
python .\src\app.py
```

Note: Source modules no longer include ad-hoc self-tests.

## How It Works

1. Search providers return `Publication` objects (`src/models.py`).
2. Results are saved into `research.db` (tables `publications`, `search_results`).
3. LLM analysis compares each row’s title+abstract against its original query; updates `relevance_score` and `relevance_label`.
4. You can download PDFs and extract Markdown for kept items from the GUI.

## Notes & Tips

- CORE requires a valid `CORE_API_KEY`.
- LLM integration is HTTP-only; LM Studio must be running a compatible server at `RS_LLM_ENDPOINT`.
- The LLM response is parsed robustly from JSON or fenced code blocks; analysis JSON is stored in `search_results.analysis_json`.
- Use the GUI “Reset DB + papers now” or set `RS_RESET_ON_START` before launching to fully reset state.

## Modules

- `src/sources/arxiv_source.py`, `src/sources/core_source.py`, `src/sources/source.py`
- `src/orchestrator.py`, `src/llm.py`, `src/db.py`, `src/models.py`
- `src/app.py`

## License

Proprietary/internal by default. Do not redistribute without permission.
