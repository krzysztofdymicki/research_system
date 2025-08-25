from abc import ABC, abstractmethod
from typing import List
import os
import requests
import pymupdf4llm

from ..models import Publication


class Source(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Publication]:
        pass

    def download_pdf(self, pub: Publication, download_dir: str = "papers") -> str | None:
        if not pub:
            return None
        pdf_url = (pub.pdf_url or "").strip()
        if not pdf_url:
            return None
        
        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)
        
        file_id = (pub.original_id or "paper").replace("/", "_").replace(":", "_")
        pdf_filename = f"{file_id}.pdf"
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


def download_pdf_for_publication(pub: Publication, download_dir: str = "papers") -> str | None:
    """Helper function to download PDF without needing a Source instance."""
    if not pub or not pub.pdf_url:
        return None
    
    pdf_url = pub.pdf_url.strip()
    if not pdf_url:
        return None
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, exist_ok=True)
    
    file_id = (pub.original_id or "paper").replace("/", "_").replace(":", "_")
    pdf_filename = f"{file_id}.pdf"
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


def extract_text_from_pdf(pdf_path: str) -> str:
    return pymupdf4llm.to_markdown(pdf_path)
