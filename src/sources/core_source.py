import requests
from typing import List

from ..db.models import Publication
from .source import Source
from ..config import CORE_API_KEY, CORE_API_URL


class CoreSource(Source):
    def __init__(self):
        super().__init__("CORE")
        if not CORE_API_KEY:
            print("Warning: CORE_API_KEY not found in .env file or environment variables. CORE source will be unavailable.")

    def search(self, query: str, max_results: int = 10, offset: int = 0) -> List[Publication]:
        if not CORE_API_KEY:
            return []

        headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
        publications: List[Publication] = []
        seen_ids = set()
        cur_offset = offset if isinstance(offset, int) and offset > 0 else 0
        remaining = max(0, int(max_results) if isinstance(max_results, int) else 10)

        try:
            while remaining > 0:
                batch_limit = min(remaining, 100)
                sanitized_q = query.replace('"', '').replace("'", '') if isinstance(query, str) else query
                json_payload = {"q": sanitized_q, "limit": batch_limit}
                if cur_offset:
                    json_payload["offset"] = cur_offset

                response = requests.post(CORE_API_URL, json=json_payload, headers=headers)
                response.raise_for_status()
                results = response.json()

                items = results.get("results", []) if isinstance(results, dict) else []
                if not items:
                    break

                # Transform items into Publication, dedup by id locally
                added_this_batch = 0
                for item in items:
                    orig_id = str(item.get("id"))
                    if orig_id in seen_ids:
                        continue
                    seen_ids.add(orig_id)
                    links = item.get('links', []) or []

                    def pick_link(links_list, type_name):
                        for lk in links_list:
                            if (lk or {}).get('type') == type_name and (lk or {}).get('url'):
                                return lk.get('url')
                        return None

                    doi_val = item.get('doi')
                    main_url = f"https://doi.org/{doi_val}" if doi_val else None
                    display_url = pick_link(links, 'display') or pick_link(links, 'reader')
                    if not main_url:
                        main_url = display_url or pick_link(links, 'download')

                    def _normalize_arxiv_pdf(u: str | None) -> str | None:
                        if not u:
                            return None
                        s = u.strip()
                        if "arxiv.org/abs/" in s:
                            s = s.replace("/abs/", "/pdf/")
                            if not s.endswith(".pdf"):
                                s += ".pdf"
                            return s
                        if "arxiv.org/pdf/" in s and not s.endswith(".pdf"):
                            return s + ".pdf"
                        return s

                    candidate = item.get('downloadUrl') or pick_link(links, 'download') or display_url or pick_link(links, 'reader') or main_url
                    pdf_url = _normalize_arxiv_pdf(candidate)

                    pub = Publication(
                        original_id=orig_id,
                        title=item.get("title"),
                        authors=[author.get("name") for author in item.get("authors", [])],
                        url=main_url or (pdf_url or ""),
                        pdf_url=pdf_url,
                        abstract=item.get("abstract"),
                        source=self.name
                    )
                    publications.append(pub)
                    remaining -= 1
                    added_this_batch += 1
                    if remaining <= 0:
                        break

                # Advance offset by the number of items the API returned
                cur_offset += len(items)
                if added_this_batch == 0 or len(items) == 0:
                    break

                try:
                    total_hits = int(results.get("totalHits") or 0)
                    if cur_offset >= total_hits:
                        break
                except Exception:
                    pass

            return publications

        except requests.exceptions.RequestException as e:
            print(f"Error searching CORE: {e}")
            return []