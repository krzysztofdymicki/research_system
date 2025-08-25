# Research System

Local-first workflow for academic papers: search arXiv/CORE, curate relevant results, download PDFs, extract content, run strict JSON extraction with Pydantic validation, and prepare BERTopic corpora with interactive topic visualizations.

## Highlights
- Search arXiv and CORE; save to SQLite.
- Tkinter GUI tabs: Search, Search Results, Publications, JSON viewer, config editor.
- Relevance Score column in Publications (join from raw search results).
- Configurable extraction (prompt/classes/examples), strict validation, save only valid JSON.
- Normalization + CSV exports for BERTopic (per-extraction and per-publication).
- `bertopic_train.py` trains and outputs doc→topic mapping, topics list, and HTML visualizations.
- Test DB utility for fast iterations.

## Setup (Windows PowerShell)
1) Create and activate a Python 3.11 venv:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2) Install the project and core deps:
```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```
3) Optional — Topic modeling and visuals:
```powershell
pip install bertopic umap-learn hdbscan scikit-learn
```
4) Run the GUI:
```powershell
python .\src\app.py
```

## Core Files
- `src/app.py` — Tkinter GUI: search, analyze, promote, download, extract, view JSON, edit config.
- `src/sources/*.py` — arXiv and CORE search implementations.
- `src/db.py` — SQLite schema + helpers; `publications.extractions_json` stores validated extractions.
- `src/extraction_config.py` — Config for prompt, allowed classes, examples.
- `src/extractors/langextract_adapter.py` — Cloud adapter, unfenced→fenced JSON parsing, Pydantic validation, quiet logs.
- `src/normalize_extractions.py` — Minimal normalization, per-extraction/per-publication CSV exports, optional dedupe.
- `src/make_test_db.py` — Create small subset DB for testing.
- `src/bertopic_train.py` — Train BERTopic; exports mapping, topics CSV, and Plotly HTML.

## Exports
- Per-extraction CSV:
```powershell
python -m src.normalize_extractions --db .\research.db --class both --dedupe-within-publication --export-csv .\debug\bertopic_docs.csv
```
- Per-publication CSV:
```powershell
python -m src.normalize_extractions --db .\research.db --class both --dedupe-within-publication --export-csv-publication .\debug\bertopic_pubs.csv
```
Normalization includes: Unicode NFKC, lowercasing, whitespace collapse, de-spaced token merging. Dedupe key: `(class, text_norm)`, preferring rows with non-empty sector.

## BERTopic & Visualizations
Train and output mapping + topics + HTML:
```powershell
python .\src\bertopic_train.py `
	--csv .\debug\bertopic_docs.csv `
	--out-mapping .\debug\bertopic_doc_topics.csv `
	--out-topics .\debug\bertopic_topics.csv `
	--viz-dir .\debug\bertopic_viz `
	--topn 8 `
	--min-df 2
```
Outputs:
- `bertopic_doc_topics.csv` — `doc_id, publication_id, topic, probability, top_words`
- `bertopic_topics.csv` — BERTopic topic info table
- `bertopic_viz/` — `topics.html`, `barchart.html`, `hierarchy.html`, `heatmap.html`

## Troubleshooting
- Prefer Python 3.11 venv for BERTopic.
- Ensure `pip` and `python` are from the same venv.
- If exports are empty, confirm `publications.extractions_json` has validated items.

## License
MIT
