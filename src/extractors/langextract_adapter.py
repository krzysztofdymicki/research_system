import json
from typing import Any, Dict, List, Optional

from ..config import (
    GEMINI_MODEL,
    LANGEXTRACT_API_KEY,
)
from ..db import init_db, list_publications, update_publication_extractions
from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator


ALLOWED_CLASSES = [
    "sentiment analysis business application/use case",
    "sentiment analysis tool/software",
]


class ExtractionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    class_: str = Field(alias="class")
    text: str
    attrs: Optional[Dict[str, Any]] = None

    @field_validator("class_")
    @classmethod
    def _check_class(cls, v: str) -> str:
        if v not in ALLOWED_CLASSES:
            raise ValueError(f"class must be one of {ALLOWED_CLASSES}")
        return v

    @field_validator("text")
    @classmethod
    def _check_text(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("text must be a non-empty string")
        return v.strip()


class ExtractionPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    extractions: List[ExtractionItem] = Field(default_factory=list)


def _extract_json_fenced(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    # Try fenced blocks first: ```json ... ``` or ``` ... ```
    import re
    patterns = [
        r"```json\s*(\{[\s\S]*?\})\s*```",
        r"```\s*(\{[\s\S]*?\})\s*```",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(1)
    # Fallback: best-effort brace matching from first '{' to matching '}'
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _validate_payload(raw_text: str) -> Dict[str, Any]:
    # Try strict JSON first, then extract fenced JSON if needed
    last_err: Optional[str] = None
    obj: Any = None
    for attempt in ("strict", "fallback"):
        try:
            if attempt == "strict":
                obj = json.loads((raw_text or "").strip())
            else:
                extracted = _extract_json_fenced(raw_text)
                obj = json.loads(extracted) if extracted else None
            if obj is None:
                raise ValueError("No JSON content found")
            model = ExtractionPayload.model_validate(obj)
            payload = model.model_dump(by_alias=True)
            count = len(payload.get("extractions", []))
            return {"ok": True, "count": count, "payload": payload, "raw_text": raw_text}
        except Exception as e:  # includes ValidationError
            last_err = str(e)
            continue
    return {"ok": False, "error": last_err or "Invalid JSON payload", "payload": None, "raw_text": raw_text}

def extract_from_text(text: str, provider_override: Optional[str] = None, include_debug: bool = False, enable_chunking: bool = True) -> Dict[str, Any]:
    try:
        import langextract as lx  # type: ignore

        prompt = (
            "Extract only these two classes using exact spans and order of appearance: "
            "1) sentiment analysis business application/use case, 2) sentiment analysis tool/software. "
            "Return concise attributes when obvious (e.g., vendor, purpose)."
        )
        examples = [
            lx.data.ExampleData(
                text="We deployed a sentiment analysis tool in our CRM workflow to flag negative customer feedback.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="sentiment analysis business application/use case",
                        extraction_text="flag negative customer feedback",
                        attributes={"purpose": "customer support triage"},
                    ),
                    lx.data.Extraction(
                        extraction_class="sentiment analysis tool/software",
                        extraction_text="sentiment analysis tool",
                        attributes={"vendor": "unknown"},
                    ),
                ],
            )
        ]
        # Cloud (e.g., Gemini) via langextract. Prefer unfenced first, fallback to fenced.
        def _run_cloud(fenced: bool):
            return lx.extract(
                text_or_documents=text,
                prompt_description=prompt,
                examples=examples,
                model_id=GEMINI_MODEL,
                api_key=LANGEXTRACT_API_KEY,
                fence_output=fenced,
                use_schema_constraints=True,
            )

        try:
            result = _run_cloud(False)
        except Exception as e:
            msg = str(e).lower()
            # If unfenced fails for parsing/marker reasons, try fenced
            if "marker" in msg or "fence" in msg or "parse" in msg:
                result = _run_cloud(True)
            else:
                raise

        items: List[Dict[str, Any]] = []
        documents = getattr(result, "documents", None)
        iter_docs = documents if isinstance(documents, list) else [result]
        for doc in iter_docs:
            for ex in getattr(doc, "extractions", []) or []:
                items.append({
                    "class": getattr(ex, "extraction_class", ""),
                    "text": getattr(ex, "extraction_text", ""),
                    "attrs": getattr(ex, "attributes", {}) or {},
                })
        raw_json = json.dumps({"extractions": items}, ensure_ascii=False)
        return _validate_payload(raw_json)
    except Exception as e:
        return {"ok": False, "error": str(e), "payload": None, "raw_text": ""}


def extract_from_publication(publication_id: str, provider: Optional[str] = None, include_debug: bool = False, enable_chunking: bool = True) -> Dict[str, Any]:
    conn = init_db()
    rows = list_publications(conn, limit=100000)
    row = next((r for r in rows if str(r["id"]) == str(publication_id)), None)
    if not row:
        return {"ok": False, "error": f"Publication id not found: {publication_id}", "extractions": []}
    md = row.get("markdown")
    if not md:
        return {"ok": False, "error": f"Publication {publication_id} has no markdown. Use Extract Markdown first.", "extractions": []}
    return extract_from_text(md)


    # pretty/HTML helpers removed per request


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser("Extractor (LangExtract)")
    parser.add_argument("--text", type=str, help="Direct text input")
    parser.add_argument("--publication-id", type=str, help="Run on publication markdown by ID (string UUID)")
    parser.add_argument("--out", type=str, help="Path to JSON output", default=None)
    parser.add_argument("--save-to-db", action="store_true", help="When used with --publication-id, save result JSON to DB on success")
    # pretty/html outputs removed
    args = parser.parse_args(argv)

    if not args.text and args.publication_id is None:
        print(json.dumps({"ok": False, "error": "Provide --text or --publication-id"}, ensure_ascii=False))
        return 2

    try:
        res: Dict[str, Any]
        if args.text:
            res = extract_from_text(args.text)
        else:
            res = extract_from_publication(str(args.publication_id))
        js = json.dumps(res, ensure_ascii=False, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(js)
        # Optional DB save only valid for publication runs and ok results
        if args.publication_id and args.save_to_db and res.get("ok"):
            try:
                conn = init_db()
                update_publication_extractions(conn, publication_id=str(args.publication_id), extractions_json=js)
            except Exception:
                pass
        print(js)
    # no pretty/html output
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
