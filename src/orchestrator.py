from typing import List, Tuple
from dataclasses import dataclass

from src.sources.arxiv_source import ArxivSource
from src.sources.core_source import CoreSource
from src.sources.source import download_publication, extract_markdown_from_pdf
from src.db import init_db, upsert_publication, insert_search_result, set_relevance, get_recent_results
from src.llm import LMStudioClient
from src.models import Publication


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
    if opts.use_arxiv:
        providers.append(ArxivSource())
    if opts.use_core:
        providers.append(CoreSource())

    total = 0
    saved = 0
    for provider in providers:
        results = provider.search(opts.query, max_results=opts.max_results)
        total += len(results)
        for pub in results:
            pub_id = upsert_publication(conn, pub)
            pdf_path = None
            markdown = None

            if opts.download_pdfs:
                pdf_path = download_publication(pub, debug=False)
                if pdf_path and opts.extract_markdown:
                    try:
                        # Use per-page chunks for better downstream use
                        md_data = extract_markdown_from_pdf(pdf_path, page_chunks=False)
                        if isinstance(md_data, str):
                            markdown = md_data
                    except Exception:
                        markdown = None

            insert_search_result(
                conn,
                query=opts.query,
                publication_id=pub_id,
                pdf_path=pdf_path,
                markdown=markdown,
            )
            saved += 1

    return total, saved


def analyze_and_filter(query: str | None = None, threshold: int = 70) -> Tuple[int, int]:
    """Run LLM relevance analysis. If query is None, analyze all results.

    Returns (analyzed_count, kept_count).
    """
    conn = init_db()
    if query:
        rows = get_recent_results(conn, query_eq=query, limit=100000)
    else:
        rows = get_recent_results(conn, limit=100000)
    if not rows:
        return 0, 0

    client = LMStudioClient()
    analyzed = 0
    kept = 0
    for r in rows:
        title = r.get("title") or ""
        abstract = r.get("abstract") or ""
        pub_id = r["publication_id"]
        row_query = r.get("query") or ""
        try:
            res = client.score_relevance(row_query, title, abstract)
            score = int(res.get("score") or 0)
            label = res.get("label") or ("keep" if score >= threshold else "discard")
            set_relevance(
                conn,
                query=row_query,
                publication_id=pub_id,
                relevance_score=score,
                relevance_label=label,
                analysis_json=json_dumps_safe(res.get("raw")),
            )
            analyzed += 1
            if label == "keep":
                kept += 1
        except Exception:
            set_relevance(conn, query=row_query, publication_id=pub_id, relevance_score=None, relevance_label="error", analysis_json=None)
            analyzed += 1
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


def download_pdfs_for_kept(query: str | None = None) -> Tuple[int, int]:
    conn = init_db()
    if query:
        rows = get_recent_results(conn, query_eq=query, only_kept=True, limit=100000)
    else:
        rows = get_recent_results(conn, only_kept=True, limit=100000)
    attempted = 0
    downloaded = 0
    for r in rows:
        pub = _row_to_publication(r)
        attempted += 1
        try:
            pdf_path = download_publication(pub, debug=False)
            if pdf_path:
                insert_search_result(
                    conn,
                    query=r["query"],
                    publication_id=r["publication_id"],
                    pdf_path=pdf_path,
                    markdown=None,
                )
                downloaded += 1
        except Exception:
            pass
    return attempted, downloaded


def extract_markdown_for_kept(query: str | None = None) -> Tuple[int, int]:
    conn = init_db()
    if query:
        rows = get_recent_results(conn, query_eq=query, only_kept=True, limit=100000)
    else:
        rows = get_recent_results(conn, only_kept=True, limit=100000)
    attempted = 0
    extracted = 0
    for r in rows:
        pdf_path = r.get("pdf_path")
        if not pdf_path:
            continue
        attempted += 1
        try:
            md = None
            try:
                md_out = extract_markdown_from_pdf(pdf_path, page_chunks=False)
                if isinstance(md_out, str):
                    md = md_out
            except Exception:
                md = None
            if md:
                insert_search_result(
                    conn,
                    query=r["query"],
                    publication_id=r["publication_id"],
                    pdf_path=None,
                    markdown=md,
                )
                extracted += 1
        except Exception:
            pass
    return attempted, extracted
