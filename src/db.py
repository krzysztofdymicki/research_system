import sqlite3
import json
from typing import Optional


def init_db(db_path: str = "research.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS publications (
            id TEXT PRIMARY KEY,
            original_id TEXT,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            url TEXT NOT NULL,
            pdf_url TEXT,
            abstract TEXT,
            source TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(source, original_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            publication_id TEXT NOT NULL,
            pdf_path TEXT,
            markdown TEXT,
            relevance_score REAL,
            relevance_label TEXT,
            analysis_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(query, publication_id),
            FOREIGN KEY (publication_id) REFERENCES publications(id) ON DELETE CASCADE
        )
        """
    )
    # Backward-compatible migration: add new columns if missing
    def ensure_column(table: str, column: str, coltype: str):
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            conn.commit()

    ensure_column("search_results", "relevance_score", "REAL")
    ensure_column("search_results", "relevance_label", "TEXT")
    ensure_column("search_results", "analysis_json", "TEXT")
    return conn


def reset_db(db_path: str = "research.db") -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        # Drop tables if they exist
        conn.execute("DROP TABLE IF EXISTS search_results")
        conn.execute("DROP TABLE IF EXISTS publications")
        conn.commit()
        # Recreate schema
        conn.close()
        conn = init_db(db_path)
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()


def upsert_publication(conn: sqlite3.Connection, pub) -> str:
    # pub is an instance of Publication
    authors_json = json.dumps(pub.authors or [])
    # Try existing by (source, original_id)
    cur = conn.execute(
        "SELECT id FROM publications WHERE source = ? AND original_id = ?",
        (pub.source, pub.original_id),
    )
    row = cur.fetchone()
    if row:
        pub_id = row[0]
        conn.execute(
            """
            UPDATE publications
               SET title = ?, authors_json = ?, url = ?, pdf_url = ?, abstract = ?, updated_at = datetime('now')
             WHERE id = ?
            """,
            (pub.title, authors_json, pub.url, pub.pdf_url, pub.abstract, pub_id),
        )
        conn.commit()
        return pub_id

    # Insert new
    pub_id = str(pub.id)
    conn.execute(
        """
        INSERT INTO publications(id, original_id, title, authors_json, url, pdf_url, abstract, source)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (pub_id, pub.original_id, pub.title, authors_json, pub.url, pub.pdf_url, pub.abstract, pub.source),
    )
    conn.commit()
    return pub_id


def insert_search_result(
    conn: sqlite3.Connection,
    *,
    query: str,
    publication_id: str,
    pdf_path: Optional[str],
    markdown: Optional[str],
):
    conn.execute(
        """
    INSERT OR IGNORE INTO search_results(query, publication_id, pdf_path, markdown)
    VALUES(?, ?, ?, ?)
        """,
        (query, publication_id, pdf_path, markdown),
    )
    # If exists, update pdf_path/markdown if new info is provided
    conn.execute(
        """
        UPDATE search_results
           SET pdf_path = COALESCE(?, pdf_path),
               markdown = COALESCE(?, markdown)
         WHERE query = ? AND publication_id = ?
        """,
        (pdf_path, markdown, query, publication_id),
    )
    conn.commit()


def set_relevance(
    conn: sqlite3.Connection,
    *,
    query: str,
    publication_id: str,
    relevance_score: Optional[float],
    relevance_label: Optional[str],
    analysis_json: Optional[str] = None,
):
    conn.execute(
        """
        UPDATE search_results
           SET relevance_score = ?,
               relevance_label = ?,
               analysis_json = COALESCE(?, analysis_json)
         WHERE query = ? AND publication_id = ?
        """,
        (relevance_score, relevance_label, analysis_json, query, publication_id),
    )
    conn.commit()


def get_recent_results(
    conn: sqlite3.Connection,
    *,
    query_filter: Optional[str] = None,
    query_eq: Optional[str] = None,
    only_kept: bool = False,
    limit: int = 100,
):
    sql = (
        """
        SELECT sr.id as search_result_id,
               sr.query,
               sr.pdf_path,
               CASE WHEN sr.markdown IS NOT NULL AND length(sr.markdown) > 0 THEN 1 ELSE 0 END as has_markdown,
               sr.relevance_score,
               sr.relevance_label,
               sr.created_at as result_created_at,
               p.id as publication_id,
               p.original_id,
               p.title,
               p.authors_json,
               p.url,
               p.pdf_url,
               p.abstract,
               p.source,
               p.created_at as pub_created_at
          FROM search_results sr
          JOIN publications p ON p.id = sr.publication_id
         {where}
         ORDER BY sr.created_at DESC
         LIMIT ?
        """
    )
    params = []
    where_clauses = []
    if query_filter:
        where_clauses.append("sr.query LIKE ?")
        params.append(f"%{query_filter}%")
    if query_eq:
        where_clauses.append("sr.query = ?")
        params.append(query_eq)
    if only_kept:
        where_clauses.append("LOWER(COALESCE(sr.relevance_label, '')) = 'keep'")
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = sql.format(where=where)
    params.append(limit)
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    # Decode authors_json
    for r in rows:
        try:
            r["authors"] = json.loads(r.get("authors_json") or "[]")
        except Exception:
            r["authors"] = []
    return rows
