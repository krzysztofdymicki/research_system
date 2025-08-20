from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
import uuid

from .sources.arxiv_source import ArxivSource
from .sources.core_source import CoreSource
from .sources.source import extract_text_from_pdf
from .db import (
    init_db,
    insert_search,
    insert_raw_result,
    set_raw_relevance,
    list_raw_results,
    promote_to_publications,
    list_publications,
    update_publication_assets,
)
from .llm import LMStudioClient
from .models import Publication


@dataclass
class SearchOptions:
    query: str
    max_results: int = 5
    use_arxiv: bool = True
    use_core: bool = True
    download_pdfs: bool = True
    extract_markdown: bool = True


def run_search_and_save(opts: SearchOptions) -> Tuple[int, int]:
    conn = init_db()

    providers = []
    sources_used: List[str] = []
    if opts.use_arxiv:
        providers.append(ArxivSource())
        sources_used.append("arXiv")
    if opts.use_core:
        providers.append(CoreSource())
        sources_used.append("CORE")

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


def analyze_and_filter(query: str | None = None, threshold: int = 70) -> Tuple[int, int]:
    """Run LLM relevance analysis on pending raw results.

    If query is provided, restrict to that query; otherwise analyze all pending.
    Returns (analyzed_count, kept_count_by_threshold).
    """
    conn = init_db()
    rows = list_raw_results(conn, only_pending=True, limit=100000)
    if query:
        rows = [r for r in rows if (r.get("query") or "") == query]
    if not rows:
        return 0, 0

    client = LMStudioClient()
    analyzed = 0
    kept = 0
    for r in rows:
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        row_query = r.get("query") or ""
        try:
            res = client.score_relevance(row_query, title, abstract)
            score = int(res.get("score") or 0)
            set_raw_relevance(
                conn,
                raw_id=int(r["id"]),
                relevance_score=score,
                analysis_json=json_dumps_safe(res.get("raw")),
            )
            analyzed += 1
            if score >= threshold:
                kept += 1
        except Exception:
            set_raw_relevance(conn, raw_id=int(r["id"]), relevance_score=None, analysis_json=None)
            analyzed += 1
    return analyzed, kept


def analyze_by_search_id(search_id: str, threshold: int = 70) -> Tuple[int, int]:
    """Analyze only pending items belonging to a specific search id."""
    conn = init_db()
    rows = list_raw_results(conn, search_id=search_id, only_pending=True, limit=100000)
    if not rows:
        return 0, 0
    client = LMStudioClient()
    analyzed = 0
    kept = 0
    for r in rows:
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        row_query = r.get("query") or ""
        try:
            res = client.score_relevance(row_query, title, abstract)
            score = int(res.get("score") or 0)
            set_raw_relevance(
                conn,
                raw_id=int(r["id"]),
                relevance_score=score,
                analysis_json=json_dumps_safe(res.get("raw")),
            )
            analyzed += 1
            if score >= threshold:
                kept += 1
        except Exception:
            set_raw_relevance(conn, raw_id=int(r["id"]), relevance_score=None, analysis_json=None)
            analyzed += 1
    return analyzed, kept


def analyze_with_progress(
    search_id: Optional[str] = None,
    threshold: int = 70,
    cancel_flag: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[int, int]:
    """Analyze pending items with optional cancellation and progress callback.

    - cancel_flag: callable returning True if processing should stop
    - progress_cb: called as progress_cb(processed_count, total, kept_so_far)
    Returns (analyzed_count, kept_count_by_threshold).
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
        row_query = r.get("query") or ""
        try:
            res = client.score_relevance(row_query, title, abstract)
            score = int(res.get("score") or 0)
            set_raw_relevance(
                conn,
                raw_id=int(r["id"]),
                relevance_score=score,
                analysis_json=json_dumps_safe(res.get("raw")),
            )
            analyzed += 1
            if score >= threshold:
                kept += 1
        except Exception:
            set_raw_relevance(conn, raw_id=int(r["id"]), relevance_score=None, analysis_json=None)
            analyzed += 1
        if progress_cb:
            try:
                progress_cb(analyzed, total, kept)
            except Exception:
                pass
    return analyzed, kept


def json_dumps_safe(obj) -> str:
    try:
        import json

        return json.dumps(obj)
    except Exception:
        return "{}"


def _row_to_publication(r: dict) -> Publication:
    return Publication(
        original_id=r.get("original_id"),
        title=r.get("title") or "",
        authors=r.get("authors") or [],
        url=r.get("url") or "",
        pdf_url=r.get("pdf_url"),
        abstract=r.get("abstract"),
        source=r.get("source") or "",
    )


def promote_kept(threshold: int = 70, search_id: str | None = None) -> int:
    """Promote raw results with score >= threshold into publications. Returns count inserted."""
    conn = init_db()
    rows = list_raw_results(conn, search_id=search_id, only_pending=False, limit=100000)
    ids: List[int] = []
    for r in rows:
        val = r.get("relevance_score")
        try:
            score = float(val) if val is not None else None
        except Exception:
            score = None
        if score is not None and score >= float(threshold):
            ids.append(int(r["id"]))
    if not ids:
        return 0
    return promote_to_publications(conn, raw_ids=ids)


def download_pdfs_for_publications() -> Tuple[int, int]:
    conn = init_db()
    rows = list_publications(conn, limit=100000)
    attempted = 0
    downloaded = 0
    for r in rows:
        if r.get("pdf_path"):
            continue
        pub = _row_to_publication(r)
        attempted += 1
        try:
            src = (r.get("source") or "").lower()
            if src == "arxiv":
                pdf_path = ArxivSource().download_pdf(pub, debug=False)
            elif src == "core":
                pdf_path = CoreSource().download_pdf(pub, debug=False)
            else:
                pdf_path = None
            if pdf_path:
                update_publication_assets(conn, publication_id=r["id"], pdf_path=pdf_path)
                downloaded += 1
        except Exception:
            pass
    return attempted, downloaded


def extract_markdown_for_publications() -> Tuple[int, int]:
    conn = init_db()
    rows = list_publications(conn, limit=100000)
    attempted = 0
    extracted = 0
    for r in rows:
        pdf_path = r.get("pdf_path")
        if not pdf_path or r.get("markdown"):
            continue
        attempted += 1
        md = None
        try:
            md_text = extract_text_from_pdf(pdf_path)
            if isinstance(md_text, str):
                md = md_text
        except Exception:
            md = None
        if md:
            update_publication_assets(conn, publication_id=r["id"], markdown=md)
            extracted += 1
    return attempted, extracted
