import argparse
import os
import sys

try:
    from src.orchestrator import SearchOptions, run_search_and_save
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.orchestrator import SearchOptions, run_search_and_save


def main():
    p = argparse.ArgumentParser("research-system search")
    p.add_argument("--query", required=True)
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument("--arxiv", action="store_true", help="Use arXiv")
    p.add_argument("--core", action="store_true", help="Use CORE")
    p.add_argument("--no-download", action="store_true", help="Do not download PDFs")
    p.add_argument("--no-extract", action="store_true", help="Do not extract markdown")
    args = p.parse_args()

    use_arxiv = args.arxiv or (not args.arxiv and not args.core)
    use_core = args.core or (not args.arxiv and not args.core)

    opts = SearchOptions(
        query=args.query,
        max_results=args.max_results,
        use_arxiv=use_arxiv,
        use_core=use_core,
        download_pdfs=not args.no_download,
        extract_markdown=not args.no_extract,
    )
    total, saved = run_search_and_save(opts)
    print(f"Fetched {total} items, saved {saved} to DB.")


if __name__ == "__main__":
    main()
