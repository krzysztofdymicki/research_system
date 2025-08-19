import os
import requests
from typing import List
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from src.models import Publication
    from src.sources.source import Source, download_publication, extract_text_from_pdf
except ModuleNotFoundError:
    # Allow running this file directly: add project root to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.models import Publication
    from src.sources.source import Source, download_publication, extract_text_from_pdf

CORE_API_KEY = os.environ.get("CORE_API_KEY")

class CoreSource(Source):
    """
    A source for fetching publications from CORE.
    """
    API_URL = "https://api.core.ac.uk/v3/search/works"

    def __init__(self, debug: bool = False):
        super().__init__("CORE")
        self.debug = debug
        if not CORE_API_KEY:
            print("Warning: CORE_API_KEY not found in .env file or environment variables. CORE source will be unavailable.")

    def search(self, query: str, max_results: int = 10, offset: int = 0) -> List[Publication]:
        """
        Performs a search for papers on CORE.
        """
        if not CORE_API_KEY:
            return []

        headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
        json_payload = {
            "q": query,
            "limit": max_results
        }
        if offset and isinstance(offset, int) and offset > 0:
            json_payload["offset"] = offset

        try:
            response = requests.post(self.API_URL, json=json_payload, headers=headers)
            response.raise_for_status()
            if self.debug:
                print("[CORE SEARCH] status:", response.status_code)
                print("[CORE SEARCH] url:", response.url)
                print("[CORE SEARCH] headers content-type:", response.headers.get("Content-Type"))
            results = response.json()
            if self.debug:
                if isinstance(results, dict):
                    print("[CORE SEARCH] top-level keys:", list(results.keys()))
                    print("[CORE SEARCH] payload:", json_payload)
                    sample = (results.get("results") or [])[:1]
                    if sample:
                        print("[CORE SEARCH] first item keys:", list(sample[0].keys()))

            publications = []
            for item in results.get("results", []):
                links = item.get('links', []) or []

                def pick_link(links_list, type_name):
                    for lk in links_list:
                        if (lk or {}).get('type') == type_name and (lk or {}).get('url'):
                            return lk.get('url')
                    return None

                # Prefer DOI for human-facing URL; then display/reader; then download
                doi_val = item.get('doi')
                main_url = f"https://doi.org/{doi_val}" if doi_val else None
                display_url = pick_link(links, 'display') or pick_link(links, 'reader')
                if not main_url:
                    main_url = display_url or pick_link(links, 'download')

                # Prefer direct downloadUrl; else links:download
                pdf_url = item.get('downloadUrl') or pick_link(links, 'download')

                pub = Publication(
                    original_id=str(item.get("id")),
                    title=item.get("title"),
                    authors=[author.get("name") for author in item.get("authors", [])],
                    url=main_url or (pdf_url or ""),
                    pdf_url=pdf_url,
                    abstract=item.get("abstract"),
                    source=self.name
                )
                publications.append(pub)
            return publications

        except requests.exceptions.RequestException as e:
            print(f"Error searching CORE: {e}")
            return []

def _run_test():
    """
    Internal test function to run a sample search and print the results.
    """
    print("--- Testing CoreSource module ---")
    parser = argparse.ArgumentParser(description="Test CoreSource")
    parser.add_argument("--query", type=str, default="reinforcement learning")
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--download", action="store_true", help="Download PDFs for results")
    args = parser.parse_args()

    core_source = CoreSource(debug=True)
    query = args.query
    print(f"Searching for: '{query}'...")
    results = core_source.search(query, max_results=args.max_results, offset=args.offset)

    if not results:
        print(f"No results found for '{query}'")
        return

    print(f"\nFound {len(results)} results:\n")
    for pub in results:
        print(f"  ID (internal): {pub.id}")
        print(f"  ID (original): {pub.original_id}")
        print(f"  Title: {pub.title}")
        print(f"  Authors: {', '.join(pub.authors) if pub.authors else 'N/A'}")
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