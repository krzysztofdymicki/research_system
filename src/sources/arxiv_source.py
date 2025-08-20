import arxiv
from typing import List, Optional
import os
import requests

from ..models import Publication
from .source import Source


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

    def download_pdf(self, pub: Publication, download_dir: str = "papers") -> Optional[str]:
        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)

        # Build a direct arXiv PDF URL
        pdf_url = pub.pdf_url
        if not pdf_url and pub.url and "arxiv.org/abs/" in pub.url:
            pdf_url = pub.url.replace("/abs/", "/pdf/")
            if not pdf_url.endswith(".pdf"):
                pdf_url += ".pdf"
        if not pdf_url:
            return None

        file_id = pub.original_id or "arxiv"
        pdf_filename = f"{file_id.replace('/', '_').replace(':', '_')}.pdf"
        pdf_path = os.path.join(download_dir, pdf_filename)
        if os.path.exists(pdf_path):
            try:
                with open(pdf_path, "rb") as fh:
                    if fh.read(5).startswith(b"%PDF"):
                        return pdf_path
            except Exception:
                pass
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        }
        try:
            r = requests.get(pdf_url, stream=True, allow_redirects=True, headers=headers, timeout=60)
            r.raise_for_status()
            first = next(r.iter_content(chunk_size=8192), b"")
            ctype = (r.headers.get("Content-Type") or "").lower()
            if not (first.startswith(b"%PDF") or "application/pdf" in ctype):
                r.close()
                return None
            with open(pdf_path, "wb") as f:
                if first:
                    f.write(first)
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return pdf_path
        except requests.exceptions.RequestException:
            return None
"""
ArxivSource module defines the arXiv provider. No direct CLI harness.
Use the GUI or orchestrator in application flows.
"""