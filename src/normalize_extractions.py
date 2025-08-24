import os
import json
import re
import csv
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from .db import init_db, list_publications
except Exception:
    from db import init_db, list_publications  # type: ignore


"""
Minimal normalization: no stop-phrase filtering and no sector synonym mapping.
We keep only light text cleanup (lowercasing, whitespace and token merge) and
extract sector as-is (normalized) if present in attrs.
"""


def _collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _nfkc(s: str) -> str:
    try:
        import unicodedata
        return unicodedata.normalize("NFKC", s)
    except Exception:
        return s


def _merge_spaced_tokens(s: str) -> str:
    # Merge sequences of single-char alphanumerics, keep hyphens as separators.
    tokens = s.split()
    out: List[str] = []
    buf: List[str] = []

    def flush_buf():
        nonlocal out, buf
        if buf:
            out.append("".join(buf))
            buf = []

    for t in tokens:
        if t == "-":
            flush_buf()
            out.append("-")
            continue
        if len(t) == 1 and re.match(r"[A-Za-z0-9]", t):
            buf.append(t)
            continue
        flush_buf()
        out.append(t)
    flush_buf()
    # Reconstruct, collapsing repeated hyphens and trimming
    joined = " ".join(out)
    joined = re.sub(r"\s*-\s*", "-", joined)
    joined = re.sub(r"-+", "-", joined)
    return joined


def normalize_text(text: str) -> str:
    s = text or ""
    s = _nfkc(s)
    s = s.lower()
    s = _collapse_whitespace(s)
    s = _merge_spaced_tokens(s)
    s = _collapse_whitespace(s)
    return s


def normalize_sector(attrs: Dict[str, Any]) -> Optional[str]:
    if not isinstance(attrs, dict):
        return None
    # Find sector-like key (case-insensitive)
    keys = [k for k in attrs.keys()]
    for key in keys:
        if key is None:
            continue
        lk = str(key).lower().strip()
        if lk in ("business sector", "sector", "industry"):
            val = attrs.get(key)
            if isinstance(val, str):
                v = normalize_text(val)
                if not v:
                    return None
                return v
    return None


def flatten_publication(pub: Dict[str, Any]) -> List[Dict[str, Any]]:
    pub_id = str(pub.get("id"))
    title = pub.get("title") or ""
    src = pub.get("source") or ""
    js = pub.get("extractions_json")
    if not js:
        return []
    try:
        data = json.loads(js)
    except Exception:
        return []
    payload = (data or {}).get("payload") or {}
    items = (payload or {}).get("extractions") or []
    out: List[Dict[str, Any]] = []
    for it in items:
        try:
            klass = it.get("class")
            text = it.get("text")
            attrs = it.get("attrs") or {}
            if not isinstance(klass, str) or not isinstance(text, str):
                continue
            text_norm = normalize_text(text)
            if not text_norm or len(text_norm) < 3:
                continue
            sector_norm = normalize_sector(attrs)
            out.append({
                "publication_id": pub_id,
                "publication_title": title,
                "source": src,
                "class": klass,
                "text_original": text,
                "text_norm": text_norm,
                "sector_norm": sector_norm,
                "attrs": attrs,
            })
        except Exception:
            continue
    return out


def _dedupe_rows_within_pub(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Key on (class, text_norm); prefer non-empty sector_norm if competing
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        key = (str(r.get("class") or ""), str(r.get("text_norm") or ""))
        if key not in seen:
            seen[key] = r
        else:
            cur = seen[key]
            cur_sector = cur.get("sector_norm") or ""
            new_sector = r.get("sector_norm") or ""
            if not cur_sector and new_sector:
                seen[key] = r
    return list(seen.values())


def load_flattened(
    limit: int = 100000,
    klass: str = "both",
    db_path: str = "research.db",
    dedupe_within_publication: bool = False,
) -> List[Dict[str, Any]]:
    conn = init_db(db_path)
    pubs = list_publications(conn, limit=limit)
    rows: List[Dict[str, Any]] = []
    for p in pubs:
        if not p.get("extractions_json"):
            continue
        flat = flatten_publication(p)
        if dedupe_within_publication:
            flat = _dedupe_rows_within_pub(flat)
        if klass != "both":
            target = (
                "sentiment analysis business application/use case"
                if klass == "apps"
                else "sentiment analysis tool/software"
            )
            flat = [r for r in flat if r.get("class") == target]
        rows.extend(flat)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Normalize extractions and optionally export CSV for BERTopic")
    ap.add_argument("--sample", type=int, default=0, help="Show first N rows after normalization (JSONL to stdout)")
    ap.add_argument("--class", dest="klass", choices=["apps", "tools", "both"], default="both", help="Filter by class")
    ap.add_argument("--export-csv", dest="export_csv", default=None, help="Path to write per-extraction CSV for BERTopic")
    ap.add_argument("--export-csv-publication", dest="export_csv_pub", default=None, help="Path to write per-publication CSV (one doc per paper; text = concatenated phrases)")
    ap.add_argument("--db", dest="db_path", default="research.db", help="Path to SQLite database (default: research.db)")
    ap.add_argument("--dedupe-within-publication", action="store_true", help="Remove duplicate (class, text_norm) within each publication, prefer non-empty sector")
    args = ap.parse_args()

    rows = load_flattened(
        klass=args.klass,
        db_path=args.db_path,
        dedupe_within_publication=args.dedupe_within_publication,
    )

    if args.sample and args.sample > 0:
        shown = 0
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
            shown += 1
            if shown >= args.sample:
                break
        if shown == 0:
            print(json.dumps({"info": "no rows to display"}))

    if args.export_csv:
        path = args.export_csv
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "doc_id",
                "text",
                "class",
                "sector_norm",
                "publication_id",
                "publication_title",
                "source",
            ])
            for idx, r in enumerate(rows):
                doc_id = f"doc-{idx+1}"
                writer.writerow([
                    doc_id,
                    r.get("text_norm") or "",
                    r.get("class") or "",
                    r.get("sector_norm") or "",
                    r.get("publication_id") or "",
                    r.get("publication_title") or "",
                    r.get("source") or "",
                ])
        print(json.dumps({"export_csv": path, "rows": len(rows)}, ensure_ascii=False))

    if args.export_csv_pub:
        path = args.export_csv_pub
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Group rows by publication
        grouped: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            pid = str(r.get("publication_id") or "")
            if not pid:
                continue
            g = grouped.get(pid)
            if not g:
                grouped[pid] = {
                    "publication_id": pid,
                    "publication_title": r.get("publication_title") or "",
                    "source": r.get("source") or "",
                    "texts": [],
                }
                g = grouped[pid]
            g["texts"].append(r.get("text_norm") or "")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "doc_id",
                "text",
                "class_filter",
                "publication_id",
                "publication_title",
                "source",
            ])
            for i, (pid, g) in enumerate(grouped.items(), start=1):
                # Join phrases into a single document; keep order
                text = " ".join([t for t in g["texts"] if t])
                doc_id = f"pub-{i}"
                writer.writerow([
                    doc_id,
                    text,
                    args.klass,
                    g.get("publication_id") or "",
                    g.get("publication_title") or "",
                    g.get("source") or "",
                ])
        print(json.dumps({"export_csv_publication": path, "rows": len(grouped)}, ensure_ascii=False))



if __name__ == "__main__":
    main()
