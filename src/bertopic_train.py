import csv
import os
import argparse
from typing import List, Dict, Any


def read_docs(csv_path: str) -> Dict[str, List[Any]]:
    docs: List[str] = []
    doc_ids: List[str] = []
    pub_ids: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"doc_id", "text", "publication_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in CSV: {', '.join(sorted(missing))}")
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            docs.append(text)
            doc_ids.append(row.get("doc_id") or "")
            pub_ids.append(row.get("publication_id") or "")
    return {"docs": docs, "doc_ids": doc_ids, "pub_ids": pub_ids}


def main():
    ap = argparse.ArgumentParser(description="Train a minimal BERTopic model from CSV and output doc→topic mapping")
    ap.add_argument("--csv", required=True, help="Input CSV from normalize_extractions (per-extraction export)")
    ap.add_argument("--out-mapping", required=True, help="Output CSV with columns: doc_id, publication_id, topic, probability, top_words")
    ap.add_argument("--topn", type=int, default=5, help="Number of top words to include per topic in mapping (default 5; 0 disables)")
    ap.add_argument("--min-df", type=int, default=None, help="Optional CountVectorizer min_df to reduce vocab size")
    args = ap.parse_args()

    data = read_docs(args.csv)
    docs = data["docs"]
    doc_ids = data["doc_ids"]
    pub_ids = data["pub_ids"]
    if not docs:
        raise SystemExit("No documents to train on.")

    try:
        from bertopic import BERTopic
    except ModuleNotFoundError as e:
        raise SystemExit(
            "BERTopic is not installed. Install with:\n"
            "pip install bertopic umap-learn hdbscan scikit-learn\n"
            "If using VS Code, ensure the same interpreter runs both pip and python."
        ) from e
    vectorizer_model = None
    if args.min_df is not None:
        try:
            from sklearn.feature_extraction.text import CountVectorizer
        except ModuleNotFoundError as e:
            raise SystemExit(
                "scikit-learn is not installed. Install with:\n"
                "pip install scikit-learn"
            ) from e
        vectorizer_model = CountVectorizer(min_df=args.min_df)

    if vectorizer_model is not None:
        topic_model = BERTopic(vectorizer_model=vectorizer_model)
    else:
        topic_model = BERTopic()

    topics, probs = topic_model.fit_transform(docs)

    # Pre-compute top words per topic if requested
    top_words_cache: Dict[int, str] = {}
    if args.topn and args.topn > 0:
        for t in set(topics):
            if t == -1:
                top_words_cache[t] = ""
                continue
            items = topic_model.get_topic(t) or []
            words = [w for (w, _score) in items[: args.topn]]
            top_words_cache[t] = " ".join(words)

    os.makedirs(os.path.dirname(args.out_mapping) or ".", exist_ok=True)
    with open(args.out_mapping, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "publication_id", "topic", "probability", "top_words"])
        for i, doc_id in enumerate(doc_ids):
            t = topics[i]
            p_str = ""
            if probs is not None:
                try:
                    p_str = f"{float(probs[i]):.6f}"
                except Exception:
                    p_str = ""
            writer.writerow([
                doc_id,
                pub_ids[i] if i < len(pub_ids) else "",
                t,
                p_str,
                top_words_cache.get(t, ""),
            ])


if __name__ == "__main__":
    main()
