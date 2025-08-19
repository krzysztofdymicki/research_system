from abc import ABC, abstractmethod
from typing import List, Optional, Union
import os
import requests
import pymupdf4llm
import sys
import re
from urllib.parse import urlparse, urljoin

try:
    from src.models import Publication
except ImportError:
    from models import Publication


class Source(ABC):
    """
    An abstract base class for publication sources.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Publication]:
        """
        Performs a search for papers.

        Args:
            query (str): The search query string.
            max_results (int): The maximum number of results to return.

        Returns:
            List[Publication]: A list of Publication objects matching the query.
        """
        pass


def download_publication(publication: Publication, download_dir: str = "papers", debug: bool = False) -> str | None:
    """
    Downloads the PDF of a publication.

    Args:
        publication (Publication): The publication to download.
        download_dir (str): The directory to save the PDF in.

    Returns:
        str: The path to the downloaded PDF, or None if the download failed.
    """
    if not publication.pdf_url:
        # Try to use the landing page URL if available and attempt salvage
        if getattr(publication, 'url', None):
            landing_url = publication.url
            if debug:
                print(f"[FALLBACK] No pdf_url, using landing URL: {landing_url}")
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)

            file_id = publication.original_id if publication.original_id is not None else "none"
            pdf_filename = f"{file_id.replace('/', '_').replace(':', '_')}.pdf"
            pdf_path = os.path.join(download_dir, pdf_filename)
            html_path = os.path.splitext(pdf_path)[0] + ".html"

            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                if debug:
                    print("[FALLBACK GET]", landing_url)
                r = requests.get(landing_url, allow_redirects=True, headers=headers, timeout=60)
                r.raise_for_status()
                with open(html_path, 'wb') as fh:
                    fh.write(r.content)
                base_url = r.url or landing_url
                salvaged = _try_find_pdf_and_download(
                    html_path=html_path,
                    base_url=base_url,
                    target_pdf_path=pdf_path,
                    headers=headers,
                    debug=debug,
                )
                if salvaged:
                    return salvaged
            except requests.exceptions.RequestException as e:
                if debug:
                    print("[FALLBACK ERROR]", e)
            # If salvage failed
            print(f"No PDF URL and could not derive PDF from landing page for '{publication.title}'")
            return None
        else:
            print(f"No PDF URL for '{publication.title}'")
            return None

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    file_id = publication.original_id if publication.original_id is not None else "none"
    pdf_filename = f"{file_id.replace('/', '_').replace(':', '_')}.pdf"
    pdf_path = os.path.join(download_dir, pdf_filename)

    if os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as fh:
                magic = fh.read(5)
            if magic.startswith(b"%PDF"):
                print(f"PDF already exists for '{publication.title}' at {pdf_path}")
                return pdf_path
            else:
                html_path_existing = os.path.splitext(pdf_path)[0] + ".html"
                os.replace(pdf_path, html_path_existing)
                print(f"Existing file at {pdf_path} was not a PDF. Renamed to {html_path_existing} and will re-download.")
        except Exception as _:
            # If we cannot verify, fall through and attempt re-download
            pass

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        }

        # Attempt a HEAD request to inspect content type
        try:
            head_resp = requests.head(publication.pdf_url, allow_redirects=True, timeout=20, headers=headers)
            if debug:
                print("[HEAD] status:", head_resp.status_code)
                print("[HEAD] url:", head_resp.url)
                print("[HEAD] content-type:", head_resp.headers.get("Content-Type"))
                print("[HEAD] content-disposition:", head_resp.headers.get("Content-Disposition"))
        except requests.exceptions.RequestException as he:
            if debug:
                print("[HEAD] request failed:", he)

        # GET the resource and validate it's a real PDF
        response = requests.get(publication.pdf_url, stream=True, allow_redirects=True, headers=headers, timeout=60)
        if debug:
            print("[GET] status:", response.status_code)
            print("[GET] final url:", response.url)
            print("[GET] content-type:", response.headers.get("Content-Type"))
            print("[GET] content-disposition:", response.headers.get("Content-Disposition"))
        response.raise_for_status()

        # Read the first chunk to sniff file type
        first_chunk = next(response.iter_content(chunk_size=8192), b"")
        content_type = (response.headers.get("Content-Type") or "").lower()
        is_pdf = first_chunk.startswith(b"%PDF") or "application/pdf" in content_type

        if not is_pdf:
            # Probably an HTML landing page or error; save as .html for inspection
            html_path = os.path.splitext(pdf_path)[0] + ".html"
            try:
                with open(html_path, "wb") as fh:
                    if first_chunk:
                        fh.write(first_chunk)
                    for chunk in response.iter_content(chunk_size=8192):
                        fh.write(chunk)
                print(f"Expected PDF but got non-PDF content. Saved response as HTML at {html_path}")
                if debug:
                    # Print a short preview of the HTML
                    try:
                        with open(html_path, "rb") as fh2:
                            preview = fh2.read(400)
                        print((preview or b"").decode(sys.stdout.encoding or "utf-8", errors='replace'))
                    except Exception:
                        pass
                # Try to salvage a direct PDF link from the landing page or URL pattern
                final_url = None
                try:
                    final_url = head_resp.url  # type: ignore[name-defined]
                except Exception:
                    final_url = response.url
                salvaged = _try_find_pdf_and_download(
                    html_path=html_path,
                    base_url=str(final_url or publication.pdf_url or ''),
                    target_pdf_path=pdf_path,
                    headers=headers,
                    debug=debug,
                )
                if salvaged:
                    return salvaged
                return None
            finally:
                try:
                    response.close()
                except Exception:
                    pass

        # It's a PDF; stream to file
        with open(pdf_path, "wb") as f:
            if first_chunk:
                f.write(first_chunk)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded PDF for '{publication.title}' to {pdf_path}")
        return pdf_path
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF for '{publication.title}': {e}")
        return None


def _try_find_pdf_and_download(html_path: str, base_url: str, target_pdf_path: str, headers: dict, debug: bool = False) -> str | None:
    """
    Given a saved HTML landing page, attempt to find a direct PDF URL and download it.
    Handles common patterns like arXiv abs -> pdf and generic .pdf links in the HTML.
    """
    # Heuristic 1: arXiv abs -> pdf
    try:
        parsed = urlparse(base_url)
        if parsed.netloc.endswith('arxiv.org'):
            path = parsed.path or ''
            if '/abs/' in path:
                pdf_path_candidate = path.replace('/abs/', '/pdf/')
                if not pdf_path_candidate.endswith('.pdf'):
                    pdf_path_candidate += '.pdf'
                pdf_url = f"{parsed.scheme}://{parsed.netloc}{pdf_path_candidate}"
                if debug:
                    print("[SALVAGE] Trying arXiv PDF:", pdf_url)
                if _attempt_download_pdf(pdf_url, target_pdf_path, headers, debug):
                    print(f"Downloaded PDF via fallback from {pdf_url} to {target_pdf_path}")
                    return target_pdf_path
    except Exception as _:
        pass

    # Heuristic 2: scan HTML for .pdf links
    try:
        # Read a reasonable chunk to avoid huge files; 2MB cap
        with open(html_path, 'rb') as fh:
            content = fh.read(2 * 1024 * 1024)
        text = content.decode('utf-8', errors='replace')

        # Find hrefs
        hrefs = re.findall(r'href=["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
        candidates = []
        for href in hrefs:
            # Prefer .pdf links; also handle relative URLs
            absolute = urljoin(base_url, href)
            if absolute.lower().endswith('.pdf'):
                candidates.append(absolute)
            # Some sites use /pdf/... without extension; still try
            elif '/pdf/' in absolute and not absolute.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                candidates.append(absolute if absolute.lower().endswith('.pdf') else absolute + ('' if absolute.endswith('/') else ''))

        # De-duplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        if debug:
            print(f"[SALVAGE] PDF candidates found: {len(unique_candidates)}")
            for c in unique_candidates[:5]:
                print("  ", c)

        max_candidates = int(os.environ.get('RS_SALVAGE_MAX_CANDIDATES', '5'))
        for cand in unique_candidates[:max_candidates]:  # limit attempts
            if _attempt_download_pdf(cand, target_pdf_path, headers, debug):
                print(f"Downloaded PDF via fallback from {cand} to {target_pdf_path}")
                return target_pdf_path
    except Exception as _:
        pass

    if debug:
        print("[SALVAGE] No usable PDF link found in HTML")
    return None


def _attempt_download_pdf(url: str, target_pdf_path: str, headers: dict, debug: bool = False) -> bool:
    """
    Attempt to download the given URL as PDF; validate content and write to target path.
    Returns True on success.
    """
    try:
        # Quick HEAD (optional)
        try:
            head_timeout = float(os.environ.get('RS_SALVAGE_HEAD_TIMEOUT', '20'))
            h = requests.head(url, allow_redirects=True, timeout=head_timeout, headers=headers)
            if debug:
                print("[SALVAGE HEAD]", h.status_code, h.url, h.headers.get('Content-Type'))
        except requests.exceptions.RequestException:
            pass

        get_timeout = float(os.environ.get('RS_SALVAGE_GET_TIMEOUT', '60'))
        r = requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=get_timeout)
        if debug:
            print("[SALVAGE GET]", r.status_code, r.url, r.headers.get('Content-Type'))
        r.raise_for_status()
        first = next(r.iter_content(chunk_size=8192), b"")
        ctype = (r.headers.get('Content-Type') or '').lower()
        is_pdf = first.startswith(b"%PDF") or 'application/pdf' in ctype
        if not is_pdf:
            r.close()
            return False
        with open(target_pdf_path, 'wb') as f:
            if first:
                f.write(first)
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except requests.exceptions.RequestException:
        return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file and returns it in Markdown format.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text in Markdown format.
    """
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text


def extract_markdown_from_pdf(
    pdf_path: str,
    *,
    page_chunks: bool = False,
    write_images: bool = False,
    image_dir: Optional[str] = None,
    image_format: str = "png",
    dpi: int = 150,
    pages: Optional[List[int]] = None,
    extract_words: bool = False,
) -> Union[str, List[dict]]:
    """
    Advanced PDF → Markdown extraction using pymupdf4llm with tunable options.

    Args:
        pdf_path: Path to the PDF file.
        page_chunks: If True, return a list of per-page dicts (text + metadata).
        write_images: If True, export page images/vector graphics alongside text.
        image_dir: Directory for extracted images (created if missing) when write_images=True.
        image_format: Image format for extracted images (e.g., 'png', 'jpg').
        dpi: Resolution for extracted images.
        pages: Optional list of 0-based page numbers to process.
        extract_words: If True, include detailed word information per page (when page_chunks=True).

    Returns:
        Markdown string when page_chunks is False, otherwise a list of dicts (one per page).
    """
    kwargs: dict = {
        "doc": pdf_path,
    }
    if pages is not None:
        kwargs["pages"] = pages
    if page_chunks:
        kwargs["page_chunks"] = True
    if write_images:
        kwargs["write_images"] = True
        if image_dir:
            # Ensure output directory exists
            os.makedirs(image_dir, exist_ok=True)
            kwargs["image_path"] = image_dir
        kwargs["image_format"] = image_format
        kwargs["dpi"] = dpi
    if extract_words:
        kwargs["extract_words"] = True

    result = pymupdf4llm.to_markdown(**kwargs)
    return result
