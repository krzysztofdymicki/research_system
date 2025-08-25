@echo off
echo Enhanced BERTopic Training with Recommended Settings
echo =====================================================

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Export normalized documents with deduplication
echo.
echo Step 1: Exporting normalized documents...
python -m src.normalize_extractions ^
  --db research.db ^
  --class both ^
  --dedupe-within-publication ^
  --export-csv debug\bertopic_docs_enhanced.csv

REM Train BERTopic with enhanced settings
echo.
echo Step 2: Training BERTopic with enhancements...
python src\bertopic_train.py ^
  --csv debug\bertopic_docs_enhanced.csv ^
  --out-mapping debug\bertopic_enhanced_mapping.csv ^
  --out-topics debug\bertopic_enhanced_topics.csv ^
  --viz-dir debug\bertopic_enhanced_viz ^
  --save-model debug\bertopic_model ^
  --embedding-model bge ^
  --representation keybert ^
  --min-df 2 ^
  --max-df 0.9 ^
  --ngram-range "1,3" ^
  --umap-neighbors 15 ^
  --umap-components 5 ^
  --min-cluster-size 10 ^
  --reduce-outliers ^
  --hierarchical ^
  --topics-per-class ^
  --topn 10

echo.
echo Training complete! Check debug\bertopic_enhanced_viz for visualizations.
pause