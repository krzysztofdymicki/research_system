import os
import json
from typing import Any, Dict, List, Optional

import requests

from ..config import LLM_ENDPOINT, LLM_MODEL, LLM_TEMPERATURE
_DEFAULT_MAX_CHARS = int(os.getenv("RS_LLM_MAX_CHARS", "15000"))

from ..llm import LMStudioClient
from ..db import init_db, list_publications
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


def _prompt_template() -> str:
    return (
        "You are an information extraction assistant for a literature review titled 'Sentiment analysis in business: an overview of tools and applications'. Return ONLY JSON. "
        "Extract entities from the input text using exact spans (no paraphrase), in order of appearance. "
        "Schema: {\n"
    "  \"extractions\": [ {\n"
    "    \"class\": one of [\"sentiment analysis business application/use case\", \"sentiment analysis tool/software\"],\n"
        "    \"text\": exact substring from input (short, the smallest span that expresses the item),\n"
        "    \"attrs\": optional object with concise attributes (e.g., {\"vendor\": \"...\", \"purpose\": \"...\"})\n"
        "  } ]\n"
        "}\n"
    "Rules: Only extract the two classes above. Ignore domains, datasets, metrics, algorithms, generic tasks, and academic-only details. "
        "Do not overlap spans. Prefer business-relevant wording. Keep attribute values terse."
    )


def _build_messages(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _prompt_template()},
        {"role": "user", "content": text},
    ]


def _post_chat(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": float(LLM_TEMPERATURE),
    }
    resp = requests.post(LLM_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    content = (
        ((data or {}).get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return {"content": content, "payload": payload, "response": data, "status_code": resp.status_code}


def _validate_payload(raw_text: str) -> Dict[str, Any]:
    # Try strict JSON first, then extract fenced JSON if needed
    last_err: Optional[str] = None
    obj: Any = None
    for attempt in ("strict", "fallback"):
        try:
            if attempt == "strict":
                obj = json.loads((raw_text or "").strip())
            else:
                extracted = LMStudioClient._extract_json(raw_text)
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

def extract_from_text(text: str, include_debug: bool = False, enable_chunking: bool = True) -> Dict[str, Any]:
    messages = _build_messages(text)
    try:
        api = _post_chat(messages)
        raw = api.get("content") or ""
        validated = _validate_payload(raw)
        return validated
    except Exception as e:
        truncated = (text or "")[:_DEFAULT_MAX_CHARS]
        messages2 = _build_messages(truncated)
        try:
            api2 = _post_chat(messages2)
            raw2 = api2.get("content") or ""
            validated2 = _validate_payload(raw2)
            return validated2
        except Exception as e2:
            return {"ok": False, "error": f"{e}; retry failed: {e2}", "extractions": [], "raw_text": ""}


def extract_from_publication(publication_id: str, include_debug: bool = False, enable_chunking: bool = True) -> Dict[str, Any]:
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

    parser = argparse.ArgumentParser("Local extractor (LM Studio)")
    parser.add_argument("--text", type=str, help="Direct text input")
    parser.add_argument("--publication-id", type=str, help="Run on publication markdown by ID (string UUID)")
    parser.add_argument("--out", type=str, help="Path to JSON output", default=None)
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
        print(js)
    # no pretty/html output
        return 0
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
