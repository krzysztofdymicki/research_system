import arxiv
from typing import List
import os
import requests
import pymupdf4llm
import sys

# This try-except block allows the script to be run directly for testing,
# as well as be imported as a module in the main application.
try:
    from src.models import Publication
except ImportError:
    from models import Publication

def search(query: str, max_results: int = 10) -> List[Publication]:
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
        # Map the arxiv.Result object to our internal Publication model.
        pub = Publication(
            original_id=result.entry_id.split('/')[-1],
            title=result.title,
            authors=[author.name for author in result.authors],
            url=result.links[0].href,
            pdf_url=result.pdf_url,
            abstract=result.summary,
            source="arXiv"
        )
        publications.append(pub)
    return publications

def download(publication: Publication, download_dir: str = "papers") -> str | None:
    """
    Downloads the PDF of a publication.

    Args:
        publication (Publication): The publication to download.
        download_dir (str): The directory to save the PDF in.

    Returns:
        str: The path to the downloaded PDF, or None if the download failed.
    """
    if not publication.pdf_url:
        print(f"No PDF URL for '{publication.title}'")
        return None

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    file_id = publication.original_id if publication.original_id is not None else "none"
    pdf_filename = f"{file_id.replace('/', '_').replace(':', '_')}.pdf"
    pdf_path = os.path.join(download_dir, pdf_filename)

    if os.path.exists(pdf_path):
        print(f"PDF already exists for '{publication.title}' at {pdf_path}")
        return pdf_path

    try:
        response = requests.get(publication.pdf_url, stream=True)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded PDF for '{publication.title}' to {pdf_path}")
        return pdf_path
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF for '{publication.title}': {e}")
        return None

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

def _run_test():
    """
    Internal test function to run a sample search and print the results.
    """
    print("--- Testing arxiv_source module (sync version) ---")
    query = "machine learning"
    print(f"Searching for: '{query}'...")
    results = search(query, max_results=3)

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
        print(f"  ABSTRACT: {pub.abstract}")

        print("-" * 20)
    
    # Download the first result
    if results:
        print("\n--- Testing PDF download ---")
        pdf_path = download(results[0])
        if pdf_path:
            print("\n--- Testing PDF text extraction ---")
            extracted_text = extract_text_from_pdf(pdf_path)
            print(extracted_text[:500].encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding) + "...")


if __name__ == "__main__":
    # This block allows the script to be run directly for testing.
    # It requires the 'arxiv' and 'pydantic' packages to be installed.
    _run_test()
