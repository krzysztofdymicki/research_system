import csv
import os
import argparse
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def read_docs(csv_path: str) -> Dict[str, List[str]]:
    """Read documents from CSV file."""
    docs: List[str] = []
    doc_ids: List[str] = []
    pub_ids: List[str] = []
    classes: List[str] = []  # Add class tracking
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
            # Track class if present (for class-based analysis)
            classes.append(row.get("class") or "")
    return {"docs": docs, "doc_ids": doc_ids, "pub_ids": pub_ids, "classes": classes}


def get_bert_paths(base_name: str = "bertopic_run") -> Dict[str, str]:
    """Get standardized paths for BERTopic outputs in debug/BERT structure."""
    import os
    import datetime
    
    # Create timestamp for unique run identification
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{base_name}_{timestamp}"
    
    # Base BERT directory
    bert_dir = os.path.join("debug", "BERT")
    
    paths = {
        'base_dir': bert_dir,
        'run_id': run_id,
        'mappings_dir': os.path.join(bert_dir, "mappings"),
        'topics_dir': os.path.join(bert_dir, "topics"), 
        'models_dir': os.path.join(bert_dir, "models"),
        'viz_dir': os.path.join(bert_dir, "visualizations", run_id),
        'mapping_file': os.path.join(bert_dir, "mappings", f"{run_id}_mapping.csv"),
        'topics_file': os.path.join(bert_dir, "topics", f"{run_id}_topics.csv"),
        'model_file': os.path.join(bert_dir, "models", f"{run_id}.safetensors")
    }
    
    # Ensure directories exist
    for dir_path in [paths['mappings_dir'], paths['topics_dir'], paths['models_dir'], paths['viz_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    return paths


def _top_words_from_topic(topic_items: Any, n: int) -> str:
    """Extract top n words from topic representation."""
    if not isinstance(n, int) or n <= 0:
        return ""
    words: List[str] = []
    if isinstance(topic_items, list):
        for item in topic_items:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                words.append(str(item[0]))
                if len(words) >= n:
                    break
    return " ".join(words)


def train_single_class(docs: List[str], doc_ids: List[str], pub_ids: List[str], 
                      class_name: str, args: Any, topic_model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Train BERTopic on documents from a single class."""
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"Training BERTopic for class: {class_name}")
    print(f"Documents: {len(docs)}")
    print(f"{'='*60}")
    
    if not docs:
        print(f"No documents for class {class_name}, skipping...")
        return {}
    
    from bertopic import BERTopic
    
    # Create new model instance for this class
    topic_model = BERTopic(**topic_model_config)
    
    # Train model
    print(f"Training on {len(docs)} documents for class '{class_name}'...")
    topics_result, probs = topic_model.fit_transform(docs)
    
    # Ensure topics is always a numpy array for type consistency
    topics: np.ndarray[Any, Any] = np.array(topics_result) if topics_result is not None else np.array([])
    if len(topics) == 0:
        print(f"Topic modeling failed for class {class_name} - no topics generated")
        return {}
    
    if probs is not None and isinstance(probs, list):
        probs = np.array(probs)
    
    # Print initial statistics
    topic_info = topic_model.get_topic_info()
    print(f"Initial topics for {class_name}: {len(topic_info) - 1} (excluding outliers)")
    print(f"Outliers (topic -1): {sum(1 for t in topics if t == -1)} documents")
    
    # Post-processing: Reduce outliers
    if args.reduce_outliers:
        print(f"\nReducing outliers for {class_name} using strategy: {args.outlier_strategy}")
        old_outliers = sum(1 for t in topics if t == -1)
        # Convert to list for reduce_outliers function
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
        reduced_topics = topic_model.reduce_outliers(docs, topics_list, strategy=args.outlier_strategy)
        # Convert back to numpy array
        topics = np.array(reduced_topics) if reduced_topics is not None else topics
        new_outliers = sum(1 for t in topics if t == -1)
        print(f"Outliers for {class_name} reduced from {old_outliers} to {new_outliers}")
        
        # Update the model with new topics
        topic_model.update_topics(docs, topics=topics.tolist() if hasattr(topics, 'tolist') else list(topics))
    
    # Post-processing: Reduce number of topics
    if args.nr_topics:
        print(f"\nReducing {class_name} to {args.nr_topics} topics...")
        reduced_topics_result = topic_model.reduce_topics(docs, nr_topics=args.nr_topics)
        if reduced_topics_result is not None:
            topics = np.array(reduced_topics_result)
        topic_info = topic_model.get_topic_info()
        print(f"Final topics for {class_name}: {len(topic_info) - 1} (excluding outliers)")
    
    # Generate topic labels using the representation model
    print(f"\nGenerating topic labels for {class_name}...")
    topic_model.generate_topic_labels(nr_words=3, separator=" | ")
    
    return {
        'model': topic_model,
        'topics': topics,
        'probs': probs,
        'docs': docs,
        'doc_ids': doc_ids,
        'pub_ids': pub_ids,
        'class_name': class_name
    }


def write_class_results(result: Dict[str, Any], args: Any, class_suffix: str, bert_paths: Dict[str, str]) -> None:
    """Write results for a single class using BERT directory structure."""
    if not result:
        return
        
    topic_model = result['model']
    topics = result['topics']
    probs = result['probs']
    docs = result['docs']
    doc_ids = result['doc_ids']
    pub_ids = result['pub_ids']
    class_name = result['class_name']
    
    # Pre-compute top words per topic
    top_words_cache: Dict[int, str] = {}
    if isinstance(args.topn, int) and args.topn > 0 and topics is not None:
        # Ensure topics is a list or has tolist method
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
        unique_topics = set(topics_list)
        for t in unique_topics:
            if t == -1:
                top_words_cache[t] = "outlier"
                continue
            topic_words = topic_model.get_topic(t)
            if topic_words:
                top_words_cache[t] = _top_words_from_topic(topic_words, args.topn)
            else:
                top_words_cache[t] = ""

    # Create class-specific output filenames using BERT structure
    run_id = bert_paths['run_id']
    class_mapping_path = os.path.join(bert_paths['mappings_dir'], f"{run_id}_{class_suffix}_mapping.csv")
    
    # Write mapping CSV for this class
    out_dir = os.path.dirname(class_mapping_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(class_mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "publication_id", "class", "topic", "topic_label", "probability", "top_words"])
        
        # Get topic labels
        topic_labels = topic_model.topic_labels_
        
        if topics is not None:
            for i, doc_id in enumerate(doc_ids):
                if i < len(topics):
                    t = int(topics[i])
                    p_str = ""
                    if probs is not None and i < len(probs):
                        try:
                            p_str = f"{float(probs[i]):.6f}"
                        except (ValueError, TypeError):
                            p_str = ""
                    
                    # Get topic label - handle both dict and list types
                    if hasattr(topic_labels, 'get') and topic_labels:
                        topic_label = topic_labels.get(str(t), f"Topic {t}") if t != -1 else "Outlier"
                    else:
                        topic_label = f"Topic {t}" if t != -1 else "Outlier"
                    
                    writer.writerow([
                        doc_id,
                        pub_ids[i] if i < len(pub_ids) else "",
                        class_name,
                        t,
                        topic_label,
                        p_str,
                        top_words_cache.get(t, ""),
                    ])
    print(f"Wrote {class_name} mapping CSV: {class_mapping_path}")

    # Write topics CSV if requested
    if args.out_topics:
        class_topics_path = os.path.join(bert_paths['topics_dir'], f"{run_id}_{class_suffix}_topics.csv")
        
        try:
            topic_info = topic_model.get_topic_info()
            out_dir = os.path.dirname(class_topics_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(class_topics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["class", "topic_id", "count", "name", "top_words"])
                for _, row in topic_info.iterrows():
                    topic_id = int(row["Topic"])
                    count = int(row["Count"])
                    name = str(row["Name"]) if "Name" in row else f"Topic {topic_id}"
                    top_words = top_words_cache.get(topic_id, "")
                    writer.writerow([class_name, topic_id, count, name, top_words])
            print(f"Wrote {class_name} topics CSV: {class_topics_path}")
        except Exception as e:
            print(f"Could not write {class_name} topics CSV: {e}")

    # Save model if requested
    if args.save_model:
        class_model_path = os.path.join(bert_paths['models_dir'], f"{run_id}_{class_suffix}.safetensors")
        
        try:
            save_dir = os.path.dirname(class_model_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            topic_model.save(class_model_path, serialization="safetensors")
            print(f"Saved {class_name} model to: {class_model_path}")
        except Exception as e:
            print(f"Could not save {class_name} model: {e}")

    # Visualizations (Plotly HTML) if requested
    if args.viz_dir:
        class_viz_dir = os.path.join(bert_paths['viz_dir'], f"class_{class_suffix}")
        os.makedirs(class_viz_dir, exist_ok=True)
        
        # Convert topics to list format for visualizations
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
        
        visualizations: List[Tuple[str, Any]] = [
            ("topics", lambda: topic_model.visualize_topics()),
            ("barchart", lambda: topic_model.visualize_barchart(top_n_topics=20)),
            ("hierarchy", lambda: topic_model.visualize_hierarchy()),
            ("heatmap", lambda: topic_model.visualize_heatmap()),
            ("documents", lambda: topic_model.visualize_documents(docs, topics=topics_list)),
        ]
        
        for name, viz_func in visualizations:
            try:
                fig = viz_func()
                path = os.path.join(class_viz_dir, f"{name}.html")
                fig.write_html(path)
                print(f"Wrote {class_name}: {path}")
            except Exception as e:
                print(f"Could not create {class_name} {name} visualization: {e}")

    # Print class statistics
    print(f"\n{class_name} Training complete!")
    if topics is not None:
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
        unique_count = len(set(topics_list)) - (1 if -1 in set(topics_list) else 0)
        outlier_count = sum(1 for t in topics_list if t == -1)
        print(f"{class_name} - Total topics: {unique_count}")
        print(f"{class_name} - Documents with outliers: {outlier_count}")
    else:
        print(f"Warning: No topics generated for {class_name}")


def main() -> None:
    """Main training function."""
    ap = argparse.ArgumentParser(description="Enhanced BERTopic trainer with multiple improvements")
    ap.add_argument("--csv", required=True, help="Input CSV from normalize_extractions")
    ap.add_argument("--out-mapping", default=None, help="Output CSV: doc_id, publication_id, topic, probability, top_words (auto-generated if not specified)")
    ap.add_argument("--out-topics", default=None, help="Optional topics CSV (topic id, count, name/label, top words) (auto-generated if not specified)")
    ap.add_argument("--viz-dir", default=None, help="Optional directory to write interactive HTML visualizations (auto-generated if not specified)")
    ap.add_argument("--save-model", default=None, help="Optional path to save the trained BERTopic model (auto-generated if not specified)")
    ap.add_argument("--run-name", default="bertopic_run", help="Base name for this training run (default: bertopic_run)")
    ap.add_argument("--topn", type=int, default=5, help="Number of top words to include per topic in mapping")
    ap.add_argument("--min-df", type=int, default=2, help="CountVectorizer min_df to reduce vocab size (default 2)")
    ap.add_argument("--max-df", type=float, default=0.95, help="CountVectorizer max_df to remove too common terms (default 0.95)")
    ap.add_argument("--ngram-range", type=str, default="1,3", help="N-gram range as 'min,max' (default '1,3')")
    
    # Embedding options
    ap.add_argument("--embedding-model", default="default", 
                   choices=["default", "bge", "multilingual", "instructor", "mpnet"],
                   help="Choose embedding model (default: all-MiniLM-L6-v2)")
    
    # Representation options
    ap.add_argument("--representation", default="keybert",
                   choices=["default", "keybert", "mmr", "pos", "multi"],
                   help="Topic representation method (default: keybert)")
    ap.add_argument("--mmr-diversity", type=float, default=0.3, help="Diversity for MMR representation (0-1)")
    
    # UMAP options
    ap.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors (default 15, try 10-50)")
    ap.add_argument("--umap-components", type=int, default=5, help="UMAP n_components (default 5, try 3-10)")
    ap.add_argument("--umap-metric", default="cosine", choices=["cosine", "euclidean"], help="UMAP metric")
    
    # HDBSCAN options
    ap.add_argument("--min-cluster-size", type=int, default=10, help="HDBSCAN min_cluster_size (lower = more topics)")
    ap.add_argument("--min-samples", type=int, default=5, help="HDBSCAN min_samples (lower = less noise)")
    
    # Post-processing options
    ap.add_argument("--reduce-outliers", action="store_true", help="Reduce outliers using embeddings")
    ap.add_argument("--outlier-strategy", default="embeddings", 
                   choices=["probabilities", "embeddings", "c-tf-idf"],
                   help="Strategy for outlier reduction")
    ap.add_argument("--nr-topics", type=int, default=None, help="Reduce to N topics after training")
    ap.add_argument("--hierarchical", action="store_true", help="Generate hierarchical topic structure")
    ap.add_argument("--topics-per-class", action="store_true", help="Analyze topics per class if class column exists")
    
    args = ap.parse_args()

    # Setup automatic BERT paths if not specified
    bert_paths = get_bert_paths(args.run_name)
    
    # Override arguments with auto-generated paths if not provided
    if args.out_mapping is None:
        args.out_mapping = bert_paths['mapping_file']
    if args.out_topics is None:
        args.out_topics = bert_paths['topics_file'] 
    if args.viz_dir is None:
        args.viz_dir = bert_paths['viz_dir']
    if args.save_model is None:
        args.save_model = bert_paths['model_file']
    
    print(f"BERTopic Run ID: {bert_paths['run_id']}")
    print(f"Outputs will be saved to: {bert_paths['base_dir']}")
    print(f"Mapping: {args.out_mapping}")
    print(f"Topics: {args.out_topics}")
    print(f"Visualizations: {args.viz_dir}")
    print(f"Model: {args.save_model}")
    print("-" * 80)

    data = read_docs(args.csv)
    docs = data["docs"]
    doc_ids = data["doc_ids"]
    pub_ids = data["pub_ids"]
    classes = data["classes"]
    if not docs:
        raise SystemExit("No documents to train on.")

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        import hdbscan
        import numpy as np
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Required packages not installed. Install with:\n"
            "pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn numpy pandas\n"
        ) from e
    
    # Configure embedding model
    print(f"Configuring embedding model: {args.embedding_model}")
    embedding_model: Any
    if args.embedding_model == "bge":
        embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    elif args.embedding_model == "multilingual":
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    elif args.embedding_model == "instructor":
        embedding_model = SentenceTransformer("hkunlp/instructor-base")
    elif args.embedding_model == "mpnet":
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    else:  # default
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Configure representation model
    print(f"Configuring representation: {args.representation}")
    representation_model: Any = None
    if args.representation == "keybert":
        from bertopic.representation import KeyBERTInspired
        representation_model = KeyBERTInspired()
    elif args.representation == "mmr":
        from bertopic.representation import MaximalMarginalRelevance
        representation_model = MaximalMarginalRelevance(diversity=args.mmr_diversity)
    elif args.representation == "pos":
        try:
            from bertopic.representation import PartOfSpeech
            import spacy
            # Try to load spacy model, download if needed
            try:
                nlp = spacy.load("en_core_web_sm")
                representation_model = PartOfSpeech("en_core_web_sm")
            except OSError:
                print("Spacy model not found. Installing en_core_web_sm...")
                os.system("python -m spacy download en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
                representation_model = PartOfSpeech("en_core_web_sm")
        except Exception as e:
            print(f"Could not load POS representation: {e}")
            from bertopic.representation import KeyBERTInspired
            representation_model = KeyBERTInspired()
    elif args.representation == "multi":
        # Multi-aspect representation
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
        representation_model = [
            KeyBERTInspired(),
            MaximalMarginalRelevance(diversity=0.5)
        ]
    # else: use default c-TF-IDF (representation_model = None)
    
    # Configure vectorizer
    ngram_parts = args.ngram_range.split(',')
    if len(ngram_parts) != 2:
        raise ValueError(f"Invalid ngram-range format: {args.ngram_range}. Expected 'min,max'")
    ngram_min, ngram_max = int(ngram_parts[0]), int(ngram_parts[1])
    
    vectorizer_model = CountVectorizer(
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(ngram_min, ngram_max),
        stop_words="english"  # Remove English stop words
    )
    
    # Configure UMAP
    umap_model = UMAP(
        n_neighbors=args.umap_neighbors,
        n_components=args.umap_components,
        metric=args.umap_metric,
        low_memory=False,
        random_state=42
    )
    
    # Configure HDBSCAN
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric='euclidean',
        prediction_data=True  # Enables soft clustering for outlier reduction
    )
    
    # Prepare configuration for BERTopic models
    topic_model_config = {
        'embedding_model': embedding_model,
        'umap_model': umap_model,
        'hdbscan_model': hdbscan_model,
        'vectorizer_model': vectorizer_model,
        'representation_model': representation_model,
        'calculate_probabilities': True,
        'verbose': True
    }

    # Check if we should train per class
    if args.topics_per_class and classes:
        print("Training separate models for each class...")
        
        # Group documents by class
        class_groups: Dict[str, Dict[str, List[Any]]] = {}
        for i, cls in enumerate(classes):
            if cls not in class_groups:
                class_groups[cls] = {'docs': [], 'doc_ids': [], 'pub_ids': []}
            class_groups[cls]['docs'].append(docs[i])
            class_groups[cls]['doc_ids'].append(doc_ids[i])
            class_groups[cls]['pub_ids'].append(pub_ids[i])
        
        print(f"Found {len(class_groups)} classes: {list(class_groups.keys())}")
        
        # Train separate models for each class
        all_results = []
        for class_name, class_data in class_groups.items():
            result = train_single_class(
                class_data['docs'], 
                class_data['doc_ids'], 
                class_data['pub_ids'],
                class_name, 
                args, 
                topic_model_config
            )
            if result:
                all_results.append(result)
        
        # Write results for each class
        for i, result in enumerate(all_results):
            class_suffix = result['class_name'].lower().replace(' ', '_').replace('/', '_').replace('\\', '_')
            write_class_results(result, args, class_suffix, bert_paths)
        
        print(f"\n{'='*80}")
        print("SUMMARY - Training completed for all classes:")
        for result in all_results:
            if result:
                topics = result['topics']
                class_name = result['class_name']
                if topics is not None:
                    topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
                    unique_count = len(set(topics_list)) - (1 if -1 in set(topics_list) else 0)
                    outlier_count = sum(1 for t in topics_list if t == -1)
                    print(f"  {class_name}: {unique_count} topics, {outlier_count} outliers, {len(topics_list)} documents")
        print(f"{'='*80}")
        return  # Exit after per-class training
        
    # Original single-model training (fallback)
    print("Training single model on all documents...")
    
    # Build BERTopic model with all configurations
    print("Building BERTopic model...")
    topic_model = BERTopic(**topic_model_config)

    # Train model
    print(f"Training on {len(docs)} documents...")
    topics_result, probs = topic_model.fit_transform(docs)
    
    # Ensure topics is always a numpy array for type consistency
    topics: np.ndarray[Any, Any] = np.array(topics_result) if topics_result is not None else np.array([])
    if len(topics) == 0:
        raise ValueError("Topic modeling failed - no topics generated")
    
    if probs is not None and isinstance(probs, list):
        probs = np.array(probs)
    
    # Print initial statistics
    topic_info = topic_model.get_topic_info()
    print(f"\nInitial topics: {len(topic_info) - 1} (excluding outliers)")
    print(f"Outliers (topic -1): {sum(1 for t in topics if t == -1)} documents")
    
    # Post-processing: Reduce outliers
    if args.reduce_outliers:
        print(f"\nReducing outliers using strategy: {args.outlier_strategy}")
        old_outliers = sum(1 for t in topics if t == -1)
        # Convert to list for reduce_outliers function
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
        reduced_topics = topic_model.reduce_outliers(docs, topics_list, strategy=args.outlier_strategy)
        # Convert back to numpy array
        topics = np.array(reduced_topics) if reduced_topics is not None else topics
        new_outliers = sum(1 for t in topics if t == -1)
        print(f"Outliers reduced from {old_outliers} to {new_outliers}")
        
        # Update the model with new topics
        topic_model.update_topics(docs, topics=topics.tolist() if hasattr(topics, 'tolist') else list(topics))
    
    # Post-processing: Reduce number of topics
    if args.nr_topics:
        print(f"\nReducing to {args.nr_topics} topics...")
        reduced_topics_result = topic_model.reduce_topics(docs, nr_topics=args.nr_topics)
        if reduced_topics_result is not None:
            topics = np.array(reduced_topics_result)
        topic_info = topic_model.get_topic_info()
        print(f"Final topics: {len(topic_info) - 1} (excluding outliers)")
    
    # Generate topic labels using the representation model
    print("\nGenerating topic labels...")
    topic_model.generate_topic_labels(nr_words=3, separator=" | ")

    # Optionally save model
    if args.save_model:
        try:
            save_dir = os.path.dirname(args.save_model)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            topic_model.save(args.save_model, serialization="safetensors")
            print(f"Saved model to: {args.save_model}")
        except Exception as e:
            print(f"Could not save model: {e}")

    # Pre-compute top words per topic
    top_words_cache: Dict[int, str] = {}
    if isinstance(args.topn, int) and args.topn > 0 and topics is not None:
        # Ensure topics is a list or has tolist method
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
        unique_topics = set(topics_list)
        for t in unique_topics:
            if t == -1:
                top_words_cache[t] = "outlier"
                continue
            topic_words = topic_model.get_topic(t)
            if topic_words:
                top_words_cache[t] = _top_words_from_topic(topic_words, args.topn)
            else:
                top_words_cache[t] = ""

    # Write mapping CSV
    out_dir = os.path.dirname(args.out_mapping)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(args.out_mapping, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "publication_id", "topic", "topic_label", "probability", "top_words"])
        
        # Get topic labels
        topic_labels = topic_model.topic_labels_
        
        if topics is not None:
            for i, doc_id in enumerate(doc_ids):
                if i < len(topics):
                    t = int(topics[i])
                    p_str = ""
                    if probs is not None and i < len(probs):
                        try:
                            p_str = f"{float(probs[i]):.6f}"
                        except (ValueError, TypeError):
                            p_str = ""
                    
                    # Get topic label - handle both dict and list types
                    if hasattr(topic_labels, 'get') and topic_labels:
                        topic_label = topic_labels.get(str(t), f"Topic {t}") if t != -1 else "Outlier"
                    else:
                        topic_label = f"Topic {t}" if t != -1 else "Outlier"
                    
                    writer.writerow([
                        doc_id,
                        pub_ids[i] if i < len(pub_ids) else "",
                        t,
                        topic_label,
                        p_str,
                        top_words_cache.get(t, ""),
                    ])
    print(f"Wrote mapping CSV: {args.out_mapping}")

    # Write topics CSV if requested
    if args.out_topics:
        try:
            import pandas as pd
            info = topic_model.get_topic_info()
            out_dir = os.path.dirname(args.out_topics)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            info.to_csv(args.out_topics, index=False)
            print(f"Wrote topics CSV: {args.out_topics}")
            
            # Print top 10 topics for quick inspection
            print("\nTop 10 topics by document count:")
            top_topics = info[info['Topic'] != -1].head(10)
            for _, row in top_topics.iterrows():
                print(f"  Topic {row['Topic']}: {row['Name']} ({row['Count']} docs)")
        except Exception as e:
            print(f"Could not write topics CSV: {e}")

    # Hierarchical topics if requested
    if args.hierarchical:
        try:
            print("\nGenerating hierarchical topic structure...")
            hierarchical_topics = topic_model.hierarchical_topics(docs)
            
            # Save tree visualization
            tree = topic_model.get_topic_tree(hierarchical_topics)
            if args.viz_dir:
                tree_path = os.path.join(args.viz_dir, "topic_tree.txt")
                os.makedirs(args.viz_dir, exist_ok=True)
                with open(tree_path, "w", encoding="utf-8") as f:
                    f.write(tree)
                print(f"Wrote topic tree: {tree_path}")
            else:
                print("\nTopic Tree (first 50 lines):")
                lines = tree.split('\n')
                for line in lines[:50]:
                    print(line)
        except Exception as e:
            print(f"Could not generate hierarchical topics: {e}")
    
    # Topics per class if requested and classes exist
    if args.topics_per_class and any(classes):
        try:
            print("\nAnalyzing topics per class...")
            # Filter out empty classes
            valid_docs = []
            valid_classes = []
            for doc, cls in zip(docs, classes):
                if cls:
                    valid_docs.append(doc)
                    valid_classes.append(cls)
            
            if valid_docs:
                topics_per_class = topic_model.topics_per_class(valid_docs, classes=valid_classes)
                
                # Save visualization if viz_dir specified
                if args.viz_dir:
                    os.makedirs(args.viz_dir, exist_ok=True)
                    fig = topic_model.visualize_topics_per_class(topics_per_class)
                    fig.write_html(os.path.join(args.viz_dir, "topics_per_class.html"))
                    print(f"Wrote: {os.path.join(args.viz_dir, 'topics_per_class.html')}")
                
                # Print summary
                unique_classes = set(valid_classes)
                for cls in unique_classes:
                    if cls in topics_per_class:
                        class_topics = topics_per_class[cls]
                        print(f"\n  Class '{cls}': {len(class_topics)} topics")
        except Exception as e:
            print(f"Could not analyze topics per class: {e}")

    # Visualizations (Plotly HTML) if requested
    if args.viz_dir and topics is not None:
        os.makedirs(args.viz_dir, exist_ok=True)
        
        # Convert topics to list format for visualizations
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
        
        visualizations: List[Tuple[str, Any]] = [
            ("topics", lambda: topic_model.visualize_topics()),
            ("barchart", lambda: topic_model.visualize_barchart(top_n_topics=20)),
            ("hierarchy", lambda: topic_model.visualize_hierarchy()),
            ("heatmap", lambda: topic_model.visualize_heatmap()),
            ("documents", lambda: topic_model.visualize_documents(docs, topics=topics_list)),
        ]
        
        for name, viz_func in visualizations:
            try:
                fig = viz_func()
                path = os.path.join(args.viz_dir, f"{name}.html")
                fig.write_html(path)
                print(f"Wrote: {path}")
            except Exception as e:
                print(f"Could not create {name} visualization: {e}")
    
    print("\nTraining complete!")
    if topics is not None:
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else list(topics)
        unique_count = len(set(topics_list)) - (1 if -1 in set(topics_list) else 0)
        outlier_count = sum(1 for t in topics_list if t == -1)
        print(f"Total topics: {unique_count}")
        print(f"Documents with outliers: {outlier_count}")
    else:
        print("Warning: No topics generated")


if __name__ == "__main__":
    main()
