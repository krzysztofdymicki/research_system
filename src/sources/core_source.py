import requests
from typing import List

from ..models import Publication
from .source import Source
import os
import requests
from ..config import CORE_API_KEY

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

    def download_pdf(self, pub: Publication, download_dir: str = "papers", debug: bool = False) -> str | None:
        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)

        pdf_url = pub.pdf_url or None
        if not pdf_url:
            return None
        file_id = pub.original_id or "core"
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
CoreSource module defines the CORE provider. No direct CLI harness.
Use the GUI or orchestrator in application flows.
"""