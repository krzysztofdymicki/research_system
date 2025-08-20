import arxiv
from typing import List
import sys
import os

from src.models import Publication
from src.sources.source import Source, download_publication, extract_text_from_pdf


class ArxivSource(Source):
    """
    A source for fetching publications from arXiv.
    """

    def __init__(self):
        super().__init__("arXiv")

    def search(self, query: str, max_results: int = 10) -> List[Publication]:
        """
        Performs a synchronous search for papers on arXiv using the official API.

        Args:
            query (str): The search query string.
            max_results (int): The maximum number of results to return.

        Returns:
            List[Publication]: A list of Publication objects matching the query.
        """
        search_query = arxiv.Search(
            query=query,
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
"""
ArxivSource module defines the arXiv provider. No direct CLI harness.
Use the GUI or orchestrator in application flows.
"""