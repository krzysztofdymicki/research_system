import streamlit as st
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Optional
import threading
import queue
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services import (
    SearchOptions,
    run_search_and_save,
    analyze_with_progress,
    promote_kept,
    download_pdfs_batch,
    extract_markdown_batch,
)
from src.db.db import (
    init_db,
    list_raw_results,
    list_publications,
    reset_db,
    update_publication_extractions,
)
from src.extractors.langextract_adapter import extract_from_publication
from src.extractors.utils import (
    load_extraction_config,
    save_extraction_config,
    get_default_config,
)
from src.sources.arxiv_source import ArxivSource

# Page config
st.set_page_config(
    page_title="Research System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'search_results_df' not in st.session_state:
    st.session_state.search_results_df = None
if 'publications_df' not in st.session_state:
    st.session_state.publications_df = None
if 'last_search_time' not in st.session_state:
    st.session_state.last_search_time = None
if 'analysis_progress' not in st.session_state:
    st.session_state.analysis_progress = {"current": 0, "total": 0, "kept": 0}

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
    .row-table {
        font-size: 14px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 10px 0;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load search results and publications from database."""
    conn = init_db()
    
    # Load search results
    results = list_raw_results(conn, limit=500)
    if results:
        df = pd.DataFrame(results)
        # Format columns
        df['analyzed'] = df['relevance_score'].apply(lambda x: '✓' if x is not None else '✗')
        df['score'] = df['relevance_score'].apply(lambda x: int(x) if x is not None else '-')
        st.session_state.search_results_df = df[['title', 'source', 'query', 'score', 'analyzed', 'url', 'abstract']]
    else:
        st.session_state.search_results_df = pd.DataFrame()
    
    # Load publications
    pubs = list_publications(conn, limit=200)
    if pubs:
        df_pub = pd.DataFrame(pubs)
        # Format columns
        df_pub['has_pdf'] = df_pub['pdf_path'].apply(lambda x: '✓' if x else '✗')
        df_pub['has_markdown'] = df_pub['markdown'].apply(lambda x: '✓' if x else '✗')
        df_pub['has_extractions'] = df_pub['extractions_json'].apply(lambda x: '✓' if x else '✗')
        df_pub['score'] = df_pub['relevance_score'].apply(lambda x: int(x) if x is not None else '-')
        st.session_state.publications_df = df_pub[['title', 'source', 'score', 'has_pdf', 'has_markdown', 'has_extractions', 'id']]
    else:
        st.session_state.publications_df = pd.DataFrame()


def main():
    st.title("📚 Research System")
    st.markdown("*Automated academic paper discovery and analysis*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database management
        st.subheader("Database")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data"):
                load_data()
                st.success("Data refreshed!")
        with col2:
            if st.button("🗑️ Reset All", type="secondary"):
                if st.checkbox("Confirm reset"):
                    reset_db("research.db")
                    load_data()
                    st.success("Database reset!")
        
        # Stats
        st.subheader("📊 Statistics")
        conn = init_db()
        results_count = len(list_raw_results(conn, limit=10000))
        pubs_count = len(list_publications(conn, limit=10000))
        
        col1, col2 = st.columns(2)
        col1.metric("Results", results_count)
        col2.metric("Publications", pubs_count)
        
        if st.session_state.last_search_time:
            st.caption(f"Last search: {st.session_state.last_search_time}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📋 Results", "📚 Publications", "⚙️ Extraction Config"])
    
    with tab1:
        render_search_tab()
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_publications_tab()
    
    with tab4:
        render_config_tab()


def render_search_tab():
    st.header("🔍 Paper Search")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder="Enter your search terms...",
                help="Use quotes for exact phrases"
            )
        
        with col2:
            max_results = st.number_input(
                "Max results per source",
                min_value=1,
                max_value=50,
                value=5,
                step=1
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sources:**")
            use_arxiv = st.checkbox("arXiv", value=True)
            use_core = st.checkbox("CORE", value=True)
        
        with col2:
            st.markdown("**arXiv search in:**")
            arxiv_title = st.checkbox("Title", value=True)
            arxiv_abstract = st.checkbox("Abstract", value=False)
        
        submitted = st.form_submit_button("🚀 Search", type="primary", use_container_width=True)
    
    # Query preview
    if query:
        with st.expander("🔎 Query Preview", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**arXiv query:**")
                if use_arxiv:
                    preview = ArxivSource.build_arxiv_query(
                        query, in_title=arxiv_title, in_abstract=arxiv_abstract
                    )
                    st.code(preview, language=None)
                else:
                    st.caption("(arXiv disabled)")
            
            with col2:
                st.markdown("**CORE query:**")
                if use_core:
                    st.code(query.replace('"', '').replace("'", ''), language=None)
                else:
                    st.caption("(CORE disabled)")
    
    # Handle search
    if submitted:
        if not query:
            st.error("Please enter a search query")
        elif use_arxiv and not (arxiv_title or arxiv_abstract):
            st.error("For arXiv, select at least one field: Title or Abstract")
        else:
            with st.spinner("🔄 Searching..."):
                opts = SearchOptions(
                    query=query,
                    max_results=max_results,
                    use_arxiv=use_arxiv,
                    use_core=use_core,
                    arxiv_in_title=arxiv_title,
                    arxiv_in_abstract=arxiv_abstract,
                )
                
                total, saved = run_search_and_save(opts)
                st.session_state.last_search_time = datetime.now().strftime("%H:%M:%S")
                
                st.success(f"✅ Found {total} results, saved {saved} unique items")
                load_data()
                
                # Show quick stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Found", total)
                col2.metric("Unique Saved", saved)
                col3.metric("Duplicates", total - saved)


def render_results_tab():
    st.header("📋 Search Results")
    
    # Reload button
    if st.button("🔄 Refresh Results"):
        load_data()
    
    if st.session_state.search_results_df is None or st.session_state.search_results_df.empty:
        st.info("No search results yet. Run a search first!")
        return
    
    # Analysis section
    with st.container():
        st.subheader("🤖 AI Analysis")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            research_title = st.text_input(
                "Research context (optional)",
                placeholder="Override search query for analysis...",
                help="Leave empty to use original search query"
            )
        
        with col2:
            threshold = st.number_input(
                "Score threshold",
                min_value=0,
                max_value=100,
                value=70,
                help="Minimum score to keep"
            )
        
        with col3:
            analyze_btn = st.button("🎯 Analyze Pending", type="primary")
        
        with col4:
            promote_btn = st.button("⬆️ Promote Kept")
        
        # Progress bar for analysis
        if analyze_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(done, total, kept):
                if total > 0:
                    progress = done / total
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing: {done}/{total} ({kept} kept)")
            
            with st.spinner("Running AI analysis..."):
                analyzed, kept = analyze_with_progress(
                    None, threshold, None, progress_callback,
                    research_title if research_title else None
                )
            
            st.success(f"✅ Analyzed {analyzed} items, {kept} scored ≥ {threshold}")
            load_data()
        
        if promote_btn:
            with st.spinner("Promoting kept items..."):
                promoted = promote_kept(threshold)
                st.success(f"✅ Promoted {promoted} items to Publications")
                load_data()
    
    # Results table
    st.subheader("📊 Results Table")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_source = st.selectbox(
            "Filter by source",
            ["All"] + list(st.session_state.search_results_df['source'].unique())
        )
    with col2:
        filter_analyzed = st.selectbox(
            "Filter by analysis",
            ["All", "Analyzed", "Pending"]
        )
    with col3:
        filter_score = st.slider(
            "Minimum score",
            0, 100, 0,
            disabled=(filter_analyzed != "Analyzed")
        )
    
    # Apply filters
    df_filtered = st.session_state.search_results_df.copy()
    
    if filter_source != "All":
        df_filtered = df_filtered[df_filtered['source'] == filter_source]
    
    if filter_analyzed == "Analyzed":
        df_filtered = df_filtered[df_filtered['analyzed'] == '✓']
        df_filtered = df_filtered[df_filtered['score'] != '-']
        df_filtered = df_filtered[df_filtered['score'] >= filter_score]
    elif filter_analyzed == "Pending":
        df_filtered = df_filtered[df_filtered['analyzed'] == '✗']
    
    # Display table
    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=400,
        column_config={
            "title": st.column_config.TextColumn("Title", width="large"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "query": st.column_config.TextColumn("Query", width="medium"),
            "score": st.column_config.NumberColumn("Score", width="small"),
            "analyzed": st.column_config.TextColumn("Analyzed", width="small"),
            "url": st.column_config.LinkColumn("URL", width="small"),
        },
        hide_index=True,
    )
    
    # Selected item details
    if st.checkbox("Show abstract for selected"):
        selected_idx = st.number_input(
            "Row index",
            min_value=0,
            max_value=len(df_filtered)-1 if len(df_filtered) > 0 else 0,
            value=0
        )
        if len(df_filtered) > selected_idx:
            row = df_filtered.iloc[selected_idx]
            st.text_area("Abstract", row.get('abstract', 'No abstract'), height=150)


def render_publications_tab():
    st.header("📚 Publications")
    
    # Action buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("📥 Download PDFs"):
            with st.spinner("Downloading PDFs..."):
                attempted, downloaded = download_pdfs_batch()
                st.success(f"Downloaded {downloaded}/{attempted} PDFs")
                load_data()
    
    with col2:
        if st.button("📝 Extract Markdown"):
            with st.spinner("Extracting text..."):
                attempted, extracted = extract_markdown_batch()
                st.success(f"Extracted {extracted}/{attempted} documents")
                load_data()
    
    with col3:
        if st.button("🔬 Run NLP Extraction"):
            st.info("Select a publication from the table below first")
    
    with col4:
        if st.button("🚀 Extract All"):
            with st.spinner("Running NLP on all publications..."):
                conn = init_db()
                pubs = list_publications(conn, limit=10000)
                to_process = [p for p in pubs if p.get("markdown") and not p.get("extractions_json")]
                
                if not to_process:
                    st.info("Nothing to process")
                else:
                    progress_bar = st.progress(0)
                    status = st.empty()
                    processed = 0
                    
                    for idx, pub in enumerate(to_process):
                        status.text(f"Processing {idx+1}/{len(to_process)}...")
                        progress_bar.progress((idx + 1) / len(to_process))
                        
                        try:
                            result = extract_from_publication(str(pub["id"]))
                            if result.get("ok"):
                                json_str = json.dumps(result, ensure_ascii=False)
                                update_publication_extractions(
                                    conn, publication_id=pub["id"],
                                    extractions_json=json_str
                                )
                                processed += 1
                        except:
                            continue
                    
                    st.success(f"Processed {processed}/{len(to_process)} publications")
                    load_data()
    
    with col5:
        if st.button("🔄 Refresh"):
            load_data()
    
    # Publications table
    if st.session_state.publications_df is None or st.session_state.publications_df.empty:
        st.info("No publications yet. Analyze and promote search results first!")
        return
    
    st.subheader("📊 Publications Table")
    
    # Display enhanced table with manual selection
    df_display = st.session_state.publications_df.copy()
    
    # Add checkbox column for selection
    with st.container():
        # Simple selection approach using radio button or selectbox
        selected_title = st.selectbox(
            "Select publication:",
            options=['None'] + df_display['title'].tolist(),
            format_func=lambda x: x[:100] + '...' if x != 'None' and len(x) > 100 else x
        )
        
        # Display table
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400,
            column_config={
                "title": st.column_config.TextColumn("Title", width="large"),
                "source": st.column_config.TextColumn("Source", width="small"),
                "score": st.column_config.NumberColumn("Score", width="small"),
                "has_pdf": st.column_config.TextColumn("PDF", width="small"),
                "has_markdown": st.column_config.TextColumn("Text", width="small"),
                "has_extractions": st.column_config.TextColumn("NLP", width="small"),
            },
            hide_index=True,
        )
    
    # Handle selected publication
    if selected_title and selected_title != 'None':
        selected_pub = df_display[df_display['title'] == selected_title].iloc[0]
        pub_id = selected_pub['id']
        
        st.subheader(f"📄 Selected: {selected_pub['title'][:100]}...")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔬 Run NLP Extraction (Selected)"):
                with st.spinner("Running NLP extraction..."):
                    try:
                        result = extract_from_publication(str(pub_id))
                        if result.get("ok"):
                            json_str = json.dumps(result, ensure_ascii=False)
                            conn = init_db()
                            update_publication_extractions(
                                conn, publication_id=pub_id,
                                extractions_json=json_str
                            )
                            st.success("Extraction completed!")
                            load_data()
                        else:
                            st.error(f"Error: {result.get('error')}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("👁️ View Extractions"):
                conn = init_db()
                pubs = list_publications(conn, limit=10000)
                pub_data = next((p for p in pubs if str(p["id"]) == str(pub_id)), None)
                
                if pub_data and pub_data.get("extractions_json"):
                    try:
                        extractions = json.loads(pub_data["extractions_json"])
                        st.json(extractions)
                    except:
                        st.text(pub_data.get("extractions_json"))
                else:
                    st.info("No extractions yet")
        
        with col3:
            st.metric("Score", selected_pub.get('score', '-'))


def render_config_tab():
    st.header("⚙️ Extraction Configuration")
    
    st.markdown("""
    Configure the NLP extraction settings. This affects how the system extracts 
    sentiment analysis applications and tools from papers.
    """)
    
    # Load current config
    config = load_extraction_config()
    
    # Config editor
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # JSON editor
        config_json = st.text_area(
            "Configuration JSON",
            value=json.dumps(config, ensure_ascii=False, indent=2),
            height=400,
            help="Edit the extraction configuration"
        )
    
    with col2:
        st.markdown("### Actions")
        
        if st.button("💾 Save Config", type="primary"):
            try:
                new_config = json.loads(config_json)
                
                # Validate
                if not isinstance(new_config, dict):
                    st.error("Config must be a JSON object")
                elif "prompt" not in new_config:
                    st.error("Missing 'prompt' field")
                elif "allowed_classes" not in new_config:
                    st.error("Missing 'allowed_classes' field")
                elif "examples" not in new_config:
                    st.error("Missing 'examples' field")
                else:
                    save_extraction_config(new_config)
                    st.success("✅ Configuration saved!")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        if st.button("🔄 Reset to Default"):
            default = get_default_config()
            save_extraction_config(default)
            st.success("Reset to default configuration")
            st.rerun()
        
        if st.button("📋 Copy Config"):
            st.code(config_json, language="json")
        
        # Help section
        with st.expander("📖 Configuration Help"):
            st.markdown("""
            **Configuration fields:**
            
            - **prompt**: Instructions for the LLM extractor
            - **allowed_classes**: List of extraction classes
            - **examples**: Training examples for the extractor
            
            Each example should have:
            - `text`: Sample input text
            - `extractions`: List of expected extractions
            
            Each extraction should have:
            - `extraction_class`: One of the allowed classes
            - `extraction_text`: The extracted text
            - `attributes` (optional): Additional metadata
            """)


if __name__ == "__main__":
    # Initialize data on first run
    if 'initialized' not in st.session_state:
        load_data()
        st.session_state.initialized = True
    
    main()