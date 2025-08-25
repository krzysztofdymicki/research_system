import os
import json
from typing import Dict, Any, Optional

import requests
from .config import (
    LMSTUDIO_ENDPOINT,
    LMSTUDIO_MODEL,
    LMSTUDIO_TEMPERATURE,
    LMSTUDIO_MAX_TOKENS,
)


class LMStudioClient:
    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None, timeout: int = 60):
        self.endpoint = endpoint or LMSTUDIO_ENDPOINT
        self.model = model or LMSTUDIO_MODEL
        self.timeout = timeout

    @staticmethod
    def _extract_json(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        s = str(text).strip()
        # Handle Markdown code fences ```json ... ``` or ``` ... ```
        if s.startswith("```"):
            # strip leading ```
            body = s[3:]
            end = body.find("```")
            if end != -1:
                fenced = body[:end].lstrip()
                # Remove optional language tag like 'json' at the start
                if fenced.lower().startswith("json"):
                    fenced = fenced[4:].lstrip()
                return fenced.strip()
        # Fallback: extract first balanced JSON object by brace counting
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
        return None

    def score_relevance(self, query: str, title: str, abstract: Optional[str]) -> Dict[str, Any]:
        prompt = (
            "You are an assistant that evaluates how relevant a paper is to a target research title. "
            "Return strictly a JSON object with keys: score (0-100 integer), kept (true/false), "
            "label (one of: 'keep','maybe','discard'), and rationale (short string). "
            "Be conservative: only 'keep' if clearly relevant.\n\n"
            f"Research title: {query}\nPaper title: {title}\nAbstract: {abstract or ''}\n\n"
            "Respond with only JSON, no extra text."
        )

        # HTTP OpenAI-compatible endpoint only
        temperature = LMSTUDIO_TEMPERATURE
        max_tokens = LMSTUDIO_MAX_TOKENS
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Try parse JSON directly; if that fails, extract from code fences/balanced braces
        out: Dict[str, Any]
        raw_json = None
        try:
            raw_json = (content or "").strip()
            out = json.loads(raw_json)
        except Exception:
            try:
                extracted = self._extract_json(content)
                if extracted:
                    raw_json = extracted
                    out = json.loads(extracted)
                else:
                    out = {"score": 0, "kept": False, "label": "discard", "rationale": "parse_error"}
            except Exception:
                out = {"score": 0, "kept": False, "label": "discard", "rationale": "parse_error"}
        # Normalize types
        raw_score = out.get("score")
        try:
            score = int(raw_score if raw_score is not None else 0)
        except Exception:
            score = 0
        label = str(out.get("label") or "discard").lower()
        kept = bool(out.get("kept") if out.get("kept") is not None else score >= 70)
        return {"score": score, "label": label, "kept": kept, "raw": out, "raw_text": content, "raw_json": raw_json}
