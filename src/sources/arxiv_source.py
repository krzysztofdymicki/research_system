import arxiv
from typing import List
import os
import requests

from ..models import Publication
from .source import Source


class ArxivSource(Source):
    """
    A source for fetching publications from arXiv.
    """

    def __init__(self, in_title: bool = True, in_abstract: bool = False):
        super().__init__("arXiv")
        # At least one should be True at call sites; default keeps prior behavior
        self.in_title = bool(in_title)
        self.in_abstract = bool(in_abstract)

    @staticmethod
    def build_arxiv_query(query: str, in_title: bool = True, in_abstract: bool = False) -> str:
        q = (query or "").strip()
        if any(prefix in q for prefix in ["ti:", "au:", "abs:", "all:", "cat:"]):
            return q
        # Extract quoted phrases first
        phrases: List[str] = []
        cur = []
        in_q = False
        qch = ''
        for ch in q:
            if ch in ('"', "'"):
                if not in_q:
                    in_q = True
                    qch = ch
                    cur = []
                elif ch == qch:
                    in_q = False
                    phrase = ''.join(cur).strip()
                    if phrase:
                        phrases.append(phrase)
                    cur = []
                else:
                    cur.append(ch)
            else:
                if in_q:
                    cur.append(ch)
        def _fields_clause(val: str) -> str:
            clauses = []
            if in_title:
                clauses.append(f'ti:"{val}"')
            if in_abstract:
                clauses.append(f'abs:"{val}"')
            if len(clauses) == 1:
                return clauses[0]
            return "(" + " OR ".join(clauses) + ")"

        if phrases:
            parts = [_fields_clause(p) for p in phrases]
            return " AND ".join(parts)
        # Fallback: tokenize words and AND by title
        tokens = [t for t in q.split() if t]
        if in_title or in_abstract:
            def _fields_token(tok: str) -> str:
                clauses = []
                if in_title:
                    clauses.append(f'ti:{tok}')
                if in_abstract:
                    clauses.append(f'abs:{tok}')
                if len(clauses) == 1:
                    return clauses[0]
                return "(" + " OR ".join(clauses) + ")"
            parts = [_fields_token(t) for t in tokens]
        else:
            parts = [f'ti:{t}' for t in tokens]
        return " AND ".join(parts) if parts else q

    def search(self, query: str, max_results: int = 10) -> List[Publication]:
        """
        Performs a synchronous search for papers on arXiv using the official API.

        Args:
            query (str): The search query string.
            max_results (int): The maximum number of results to return.

        Returns:
            List[Publication]: A list of Publication objects matching the query.
        """
        # Build arXiv API query according to rules
        arxiv_query = ArxivSource.build_arxiv_query(query, in_title=self.in_title, in_abstract=self.in_abstract)

        search_query = arxiv.Search(
            query=arxiv_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        client = arxiv.Client()
        results = client.results(search_query)

        publications = []
        for result in results:
            pub = Publication(
                original_id=result.entry_id.split('/')[-1],
                title=result.title,
                authors=[author.name for author in result.authors],
                url=result.entry_id,  # canonical arXiv abs page
                pdf_url=result.pdf_url,
                abstract=result.summary,
                source=self.name
            )
            publications.append(pub)
        return publications

    # Uses common Source.download_pdf via orchestrator.
"""
ArxivSource module defines the arXiv provider. No direct CLI harness.
Use the GUI or orchestrator in application flows.
"""