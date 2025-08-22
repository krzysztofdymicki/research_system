import os
import json
from typing import Any, Dict, List, Optional

import requests

from ..config import LLM_ENDPOINT, LLM_MODEL, LLM_TEMPERATURE
from ..llm import LMStudioClient
from ..db import init_db, list_publications


def _prompt_template() -> str:
    return (
        "You are an information extraction assistant. Return ONLY JSON. "
        "Extract entities from the input text using exact spans (no paraphrase), in order of appearance. "
        "Schema: {\n"
        "  \"extractions\": [ {\n"
        "    \"class\": one of [\"application\", \"tool\", \"metric\", \"domain\"],\n"
        "    \"text\": exact substring from input,\n"
        "    \"attrs\": optional object with concise attributes (e.g., application_tool, dataset)\n"
        "  } ]\n"
        "}\n"
        "Rules: Do not overlap spans. Keep attributes short and meaningful."
    )


def _build_messages(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _prompt_template()},
        {"role": "user", "content": text},
    ]


def _post_chat(messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
    "temperature": float(LLM_TEMPERATURE),
    }
    resp = requests.post(LLM_ENDPOINT, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_extractions(raw_text: str) -> Dict[str, Any]:
    # Try strict JSON first, then fenced/balanced JSON via existing helper
    try:
        obj = json.loads((raw_text or "").strip())
    except Exception:
        try:
            extracted = LMStudioClient._extract_json(raw_text)  # reuse robust helper
            obj = json.loads(extracted) if extracted else {"extractions": []}
        except Exception:
            obj = {"extractions": []}
    if not isinstance(obj, dict):
        obj = {"extractions": []}
    items = obj.get("extractions") or []
    norm = []
    for it in items:
        try:
            norm.append(
                {
                    "class": str(it.get("class") or it.get("extraction_class") or "").strip(),
                    "text": str(it.get("text") or it.get("extraction_text") or "").strip(),
                    "attrs": it.get("attrs") or it.get("attributes") or {},
                }
            )
        except Exception:
            continue
    return {"ok": True, "count": len(norm), "extractions": norm, "raw_text": raw_text}


def extract_from_text(text: str) -> Dict[str, Any]:
    messages = _build_messages(text)
    raw = _post_chat(messages)
    return _parse_extractions(raw)


def extract_from_publication(publication_id: int) -> Dict[str, Any]:
    conn = init_db()
    rows = list_publications(conn, limit=100000)
    row = next((r for r in rows if int(r["id"]) == int(publication_id)), None)
    if not row:
        return {"ok": False, "error": f"Publication id not found: {publication_id}", "extractions": []}
    md = row.get("markdown")
    if not md:
        return {"ok": False, "error": f"Publication {publication_id} has no markdown. Use Extract Markdown first.", "extractions": []}
    return extract_from_text(md)


def _print_pretty(res: Dict[str, Any]) -> None:
    items = res.get("extractions") or []
    print("Class           | Text                                   | Attrs")
    print("-" * 80)
    for it in items:
        cls = (it.get("class") or "")[:14].ljust(14)
        txt = (it.get("text") or "").replace("\n", " ")[:40].ljust(40)
        attrs = json.dumps(it.get("attrs") or {})
        print(f"{cls} | {txt} | {attrs}")


def _save_html(res: Dict[str, Any], out_path: str) -> None:
    items = res.get("extractions") or []
    rows = []
    for it in items:
        rows.append(
            f"<tr><td>{it.get('class','')}</td><td>{it.get('text','')}</td><td><pre>{json.dumps(it.get('attrs') or {}, ensure_ascii=False)}</pre></td></tr>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>Extractions</title>"
        "<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 10px;font-family:Segoe UI,Arial}</style>"
        "<h2>Extractions</h2>"
        "<table><thead><tr><th>Class</th><th>Text</th><th>Attrs</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser("Local extractor (LM Studio)")
    parser.add_argument("--text", type=str, help="Direct text input")
    parser.add_argument("--publication-id", type=int, help="Run on publication markdown by ID")
    parser.add_argument("--out", type=str, help="Path to JSON output", default=None)
    parser.add_argument("--pretty", action="store_true", help="Print a compact table view")
    parser.add_argument("--html", type=str, help="Optional HTML file to save a table view")
    args = parser.parse_args(argv)

    if not args.text and args.publication_id is None:
        print(json.dumps({"ok": False, "error": "Provide --text or --publication-id"}, ensure_ascii=False))
        return 2

    try:
        res: Dict[str, Any]
        if args.text:
            res = extract_from_text(args.text)
        else:
            res = extract_from_publication(int(args.publication_id))
        js = json.dumps(res, ensure_ascii=False, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(js)
        print(js)
        if args.pretty and res.get("ok"):
            _print_pretty(res)
        if args.html and res.get("ok"):
            _save_html(res, args.html)
            print(f"Saved HTML to: {args.html}")
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
