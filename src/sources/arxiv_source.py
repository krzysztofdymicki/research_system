import arxiv
from typing import List
import sys
import os
import argparse

try:
    from src.models import Publication
    from src.sources.source import Source, download_publication, extract_text_from_pdf
except ModuleNotFoundError:
    # Allow running this file directly: add project root to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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


def _run_test():
    """
    Internal test function to run a sample search and print the results.
    """
    print("--- Testing ArxivSource module ---")
    parser = argparse.ArgumentParser(description="Test ArxivSource")
    parser.add_argument("--query", type=str, default="quantum machine learning")
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--download", action="store_true", help="Download PDFs for results")
    args = parser.parse_args()

    arxiv_source = ArxivSource()
    query = args.query
    print(f"Searching for: '{query}'...")
    results = arxiv_source.search(query, max_results=args.max_results)

    if not results:
        print(f"No results found for '{query}'")
        return

    print(f"\nFound {len(results)} results:\n")
    for pub in results:
        print(f"  ID (internal): {pub.id}")
        print(f"  ID (original): {pub.original_id}")
        print(f"  Title: {pub.title}")
        print(f"  Authors: {', '.join(pub.authors)}")
        print(f"  URL: {pub.url}")
        print(f"  PDF URL: {pub.pdf_url}")
        print(f"  Abstract: {pub.abstract}")
        print("-" * 20)

    if args.download and results:
        print("\n--- Batch PDF download ---")
        ok = 0
        for i, pub in enumerate(results, start=1):
            print(f"[{i}/{len(results)}] Downloading: {pub.title}")
            pdf_path = download_publication(pub, debug=True)
            if pdf_path:
                ok += 1
        print(f"Downloaded {ok}/{len(results)} PDFs")


if __name__ == "__main__":
    _run_test()