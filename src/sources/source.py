from abc import ABC, abstractmethod
from typing import List
import pymupdf4llm

from ..models import Publication


class Source(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Publication]:
        pass

    @abstractmethod
    def download_pdf(self, pub: Publication, download_dir: str = "papers") -> str | None:
        pass


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file and returns it in Markdown format.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text in Markdown format.
    """
    return pymupdf4llm.to_markdown(pdf_path)
