from typing import List, Tuple, Callable, Optional, NamedTuple
import uuid
import json

from .sources.arxiv_source import ArxivSource
from .sources.core_source import CoreSource
from .sources.source import extract_text_from_pdf, download_pdf_for_publication
from .db.db import (
    init_db,
    insert_search,
    insert_raw_result,
    set_raw_relevance,
    list_raw_results,
    promote_to_publications,
    list_publications,
    update_publication_assets,
)
from .db.models import Publication
from .evaluators.lmstudio import LMStudioClient


class SearchOptions(NamedTuple):
    """Search configuration - simplified from dataclass to NamedTuple for immutability."""
    query: str
    max_results: int = 5
    use_arxiv: bool = True
    use_core: bool = True
    arxiv_in_title: bool = True
    arxiv_in_abstract: bool = False


def run_search_and_save(opts: SearchOptions) -> Tuple[int, int]:
    """Run search across enabled providers and save results to database.
    
    Returns:
        Tuple of (total_found, unique_saved)
    """
    conn = init_db()

    providers = []
    sources_used: List[str] = []
    
    if opts.use_arxiv:
        providers.append(ArxivSource(in_title=opts.arxiv_in_title, in_abstract=opts.arxiv_in_abstract))
        sources_used.append("arXiv")
    if opts.use_core:
        providers.append(CoreSource())
        sources_used.append("CORE")

    # Create search record
    search_id = str(uuid.uuid4())
    insert_search(
        conn,
        search_id=search_id,
        query=opts.query,
        sources=sources_used,
        max_results_per_source=opts.max_results,
    )

    total = 0
    saved = 0
    for provider in providers:
        results = provider.search(opts.query, max_results=opts.max_results)
        total += len(results)
        
        for pub in results:
            raw_id = insert_raw_result(
                conn,
                search_id=search_id,
                source=pub.source,
                original_id=pub.original_id,
                title=pub.title,
                authors=pub.authors,
                url=pub.url,
                pdf_url=pub.pdf_url,
                abstract=pub.abstract,
            )
            if raw_id != -1:
                saved += 1
            
    return total, saved


def analyze_with_progress(
    search_id: Optional[str] = None,
    threshold: int = 70,
    cancel_flag: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
    research_title: Optional[str] = None,
) -> Tuple[int, int]:
    """Analyze pending items with AI relevance scoring.

    Args:
        search_id: Optional filter by specific search
        threshold: Minimum score to consider "kept"
        cancel_flag: Callable returning True to cancel processing
        progress_cb: Called as progress_cb(processed, total, kept_count)
        research_title: Override query for analysis context
        
    Returns:
        Tuple of (analyzed_count, kept_count_above_threshold)
    """
    conn = init_db()
    rows = list_raw_results(conn, search_id=search_id, only_pending=True, limit=100000)
    total = len(rows)
    
    if total == 0:
        if progress_cb:
            try:
                progress_cb(0, 0, 0)
            except Exception:
                pass
        return 0, 0

    client = LMStudioClient()
    analyzed = 0
    kept = 0
    
    for r in rows:
        if cancel_flag and cancel_flag():
            break
            
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        query_context = research_title or (r.get("query") or "")
        
        try:
            res = client.score_relevance(query_context, title, abstract)
            score = int(res.get("score") or 0)
            analysis_json = _safe_json_dumps(res.get("raw"))
            
            set_raw_relevance(
                conn,
                raw_id=int(r["id"]),
                relevance_score=score,
                analysis_json=analysis_json,
            )
            
            if score >= threshold:
                kept += 1
                
        except Exception:
            # Mark as analyzed but with no score
            set_raw_relevance(conn, raw_id=int(r["id"]), relevance_score=None, analysis_json=None)
            
        analyzed += 1
        
        if progress_cb:
            try:
                progress_cb(analyzed, total, kept)
            except Exception:
                pass
                
    return analyzed, kept


def promote_kept(threshold: int = 70, search_id: Optional[str] = None) -> int:
    """Promote raw results with score >= threshold to publications.
    
    Returns:
        Number of items promoted
    """
    conn = init_db()
    rows = list_raw_results(conn, search_id=search_id, only_pending=False, limit=100000)
    
    ids_to_promote: List[int] = []
    for r in rows:
        score_val = r.get("relevance_score")
        try:
            score = float(score_val) if score_val is not None else None
        except (ValueError, TypeError):
            score = None
            
        if score is not None and score >= float(threshold):
            ids_to_promote.append(int(r["id"]))
    
    if not ids_to_promote:
        return 0
        
    return promote_to_publications(conn, raw_ids=ids_to_promote)


def download_pdfs_batch() -> Tuple[int, int]:
    """Download PDFs for all publications missing them.
    
    Returns:
        Tuple of (attempted, successfully_downloaded)
    """
    conn = init_db()
    publications = list_publications(conn, limit=100000)
    
    attempted = 0
    downloaded = 0
    
    for pub_row in publications:
        if pub_row.get("pdf_path"):
            continue  # Already has PDF
            
        attempted += 1
        pub_obj = _row_to_publication(pub_row)
        
        try:
            pdf_path = download_pdf_for_publication(pub_obj)
            if pdf_path:
                update_publication_assets(conn, publication_id=pub_row["id"], pdf_path=pdf_path)
                downloaded += 1
        except Exception:
            continue  # Skip failed downloads
            
    return attempted, downloaded


def extract_markdown_batch() -> Tuple[int, int]:
    """Extract markdown from PDFs for all publications missing it.
    
    Returns:
        Tuple of (attempted, successfully_extracted)
    """
    conn = init_db()
    publications = list_publications(conn, limit=100000)
    
    attempted = 0
    extracted = 0
    
    for pub_row in publications:
        pdf_path = pub_row.get("pdf_path")
        if not pdf_path or pub_row.get("markdown"):
            continue  # No PDF or already has markdown
            
        attempted += 1
        
        try:
            markdown_text = extract_text_from_pdf(pdf_path)
            if markdown_text and isinstance(markdown_text, str):
                update_publication_assets(conn, publication_id=pub_row["id"], markdown=markdown_text)
                extracted += 1
        except Exception:
            continue  # Skip failed extractions
            
    return attempted, extracted


def _safe_json_dumps(obj) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj)
    except Exception:
        return "{}"


def _row_to_publication(row: dict) -> Publication:
    """Convert database row dict to Publication model."""
    return Publication(
        original_id=row.get("original_id"),
        title=row.get("title") or "",
        authors=row.get("authors") or [],
        url=row.get("url") or "",
        pdf_url=row.get("pdf_url"),
        abstract=row.get("abstract"),
        source=row.get("source") or "",
    )