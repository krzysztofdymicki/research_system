import sqlite3
import json
import uuid
from typing import Optional, List, Dict, Any


def init_db(db_path: str = "research.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)  # type: ignore[call-arg]
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Searches table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS searches (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            sources_json TEXT,
            max_results_per_source INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    
    # Raw search results table (pre-AI queue)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_id TEXT NOT NULL,
            source TEXT NOT NULL,
            original_id TEXT,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            url TEXT NOT NULL,
            pdf_url TEXT,
            abstract TEXT,
            relevance_score REAL,
            analysis_json TEXT,
            analyzed_at TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(search_id, source, original_id, title),
            FOREIGN KEY (search_id) REFERENCES searches(id) ON DELETE CASCADE
        )
        """
    )
    
    # Unique index for global deduplication by title
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_raw_search_results_title
        ON raw_search_results(title COLLATE NOCASE)
        """
    )
    
    # Publications table (accepted items after AI analysis)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS publications (
            id TEXT PRIMARY KEY,
            search_result_id INTEGER NOT NULL,
            original_id TEXT,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            url TEXT NOT NULL,
            pdf_url TEXT,
            abstract TEXT,
            source TEXT NOT NULL,
            pdf_path TEXT,
            markdown TEXT,
            extractions_json TEXT,
            relevance_score REAL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(source, original_id),
            FOREIGN KEY (search_result_id) REFERENCES raw_search_results(id) ON DELETE CASCADE
        )
        """
    )
    
    return conn


def reset_db(db_path: str = "research.db") -> None:
    conn = sqlite3.connect(db_path)  # type: ignore[call-arg]
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        # Drop tables if they exist (new order: publications -> raw_search_results -> searches)
        conn.execute("DROP TABLE IF EXISTS publications")
        conn.execute("DROP TABLE IF EXISTS raw_search_results")
        conn.execute("DROP TABLE IF EXISTS searches")
        conn.commit()
    finally:
        conn.close()
    # Recreate schema and vacuum
    conn = init_db(db_path)
    conn.execute("VACUUM")
    conn.commit()
    conn.close()


def insert_search(conn: sqlite3.Connection, *, search_id: str, query: str, sources: List[str], max_results_per_source: int) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO searches(id, query, sources_json, max_results_per_source)
        VALUES(?, ?, ?, ?)
        """,
        (search_id, query, json.dumps(sources or []), max_results_per_source),
    )
    conn.commit()


def insert_raw_result(
    conn: sqlite3.Connection,
    *,
    search_id: str,
    source: str,
    original_id: Optional[str],
    title: str,
    authors: List[str],
    url: str,
    pdf_url: Optional[str],
    abstract: Optional[str],
) -> int:
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO raw_search_results(search_id, source, original_id, title, authors_json, url, pdf_url, abstract)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (search_id, source, original_id, title, json.dumps(authors or []), url, pdf_url, abstract),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return -1
    cur = conn.execute(
        "SELECT id FROM raw_search_results WHERE search_id = ? AND source = ? AND COALESCE(original_id,'') = COALESCE(?, '') AND title = ?",
        (search_id, source, original_id, title),
    )
    row = cur.fetchone()
    return int(row[0]) if row else -1


def set_raw_relevance(
    conn: sqlite3.Connection,
    *,
    raw_id: int,
    relevance_score: Optional[float],
    analysis_json: Optional[str],
):
    conn.execute(
        """
        UPDATE raw_search_results
           SET relevance_score = ?,
               analysis_json = COALESCE(?, analysis_json),
               analyzed_at = datetime('now')
         WHERE id = ?
        """,
        (relevance_score, analysis_json, raw_id),
    )
    conn.commit()


def promote_to_publications(
    conn: sqlite3.Connection,
    *,
    raw_ids: List[int],
) -> int:
    # Insert accepted items into publications; ignore duplicates
    inserted = 0
    for raw_id in raw_ids:
        cur = conn.execute("SELECT * FROM raw_search_results WHERE id = ?", (raw_id,))
        row = cur.fetchone()
        if not row:
            continue
        cols = [d[0] for d in cur.description]
        r = dict(zip(cols, row))
        pub_id = str(uuid.uuid4())
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO publications(
                    id, search_result_id, original_id, title, authors_json, url, pdf_url, abstract, source, 
                    relevance_score, pdf_path, markdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    pub_id,
                    r["id"],
                    r.get("original_id"),
                    r.get("title"),
                    r.get("authors_json"),
                    r.get("url"),
                    r.get("pdf_url"),
                    r.get("abstract"),
                    r.get("source"),
                    r.get("relevance_score"),
                ),
            )
            inserted += 1
        except Exception:
            pass
    conn.commit()
    return inserted


def list_searches(conn: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, query, sources_json, max_results_per_source, created_at FROM searches ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        try:
            r["sources"] = json.loads(r.get("sources_json") or "[]")
        except Exception:
            r["sources"] = []
    return rows


def list_raw_results(
    conn: sqlite3.Connection,
    *,
    search_id: Optional[str] = None,
    only_pending: bool = False,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    where: List[str] = []
    params: List[Any] = []
    if search_id:
        where.append("r.search_id = ?")
        params.append(search_id)
    if only_pending:
        where.append("r.relevance_score IS NULL")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        """
        SELECT r.id, r.search_id, s.query, r.source, r.original_id, r.title, r.authors_json, r.url, r.pdf_url, r.abstract, r.relevance_score, r.analysis_json, r.analyzed_at, r.created_at
          FROM raw_search_results r
          JOIN searches s ON s.id = r.search_id
        """
        + where_sql
        + """
         ORDER BY r.created_at DESC
         LIMIT ?
        """
    )
    params.append(limit)
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        try:
            r["authors"] = json.loads(r.get("authors_json") or "[]")
        except Exception:
            r["authors"] = []
    return rows


def list_publications(conn: sqlite3.Connection, limit: int = 500) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT id, search_result_id, original_id, title, authors_json, url, pdf_url, abstract, 
               source, pdf_path, markdown, extractions_json, relevance_score, created_at, updated_at
        FROM publications
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        try:
            r["authors"] = json.loads(r.get("authors_json") or "[]")
        except Exception:
            r["authors"] = []
    return rows


def get_publication_by_id(conn: sqlite3.Connection, publication_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single publication by its primary key."""
    cur = conn.execute("SELECT * FROM publications WHERE id = ?", (publication_id,))
    row = cur.fetchone()
    if not row:
        return None
    
    cols = [d[0] for d in cur.description]
    pub_dict = dict(zip(cols, row))
    
    try:
        pub_dict["authors"] = json.loads(pub_dict.get("authors_json") or "[]")
    except (json.JSONDecodeError, TypeError):
        pub_dict["authors"] = []
        
    return pub_dict

def update_publication_assets(
    conn: sqlite3.Connection,
    *,
    publication_id: str,
    pdf_path: Optional[str] = None,
    markdown: Optional[str] = None,
) -> None:
    sets = []
    params: List[Any] = []
    if pdf_path is not None:
        sets.append("pdf_path = ?")
        params.append(pdf_path)
    if markdown is not None:
        sets.append("markdown = ?")
        params.append(markdown)
    if not sets:
        return
    sets.append("updated_at = datetime('now')")
    sql = f"UPDATE publications SET {', '.join(sets)} WHERE id = ?"
    params.append(publication_id)
    conn.execute(sql, params)
    conn.commit()


def update_publication_extractions(
    conn: sqlite3.Connection,
    *,
    publication_id: str,
    extractions_json: str,
) -> None:
    conn.execute(
        """
        UPDATE publications
           SET extractions_json = ?,
               updated_at = datetime('now')
         WHERE id = ?
        """,
        (extractions_json, publication_id),
    )
    conn.commit()


def get_search(conn: sqlite3.Connection, *, search_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, query, sources_json, max_results_per_source, created_at FROM searches WHERE id = ?",
        (search_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    out = dict(zip(cols, row))
    try:
        out["sources"] = json.loads(out.get("sources_json") or "[]")
    except Exception:
        out["sources"] = []
    return out
