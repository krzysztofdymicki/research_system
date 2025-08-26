import json
import os
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_ALLOWED_CLASSES, EXTRACTION_PROMPT, DEFAULT_EXTRACTION_EXAMPLES


def get_default_config() -> Dict[str, Any]:
    return {
        "prompt": EXTRACTION_PROMPT,
        "allowed_classes": list(DEFAULT_ALLOWED_CLASSES),
        "examples": DEFAULT_EXTRACTION_EXAMPLES,
    }


def get_config_path() -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_dir = os.path.join(base, "config")
    return os.path.join(cfg_dir, "extraction_config.json")


def load_extraction_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = path or get_config_path()
    try:
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # basic shape checks
                if not isinstance(data, dict):
                    raise ValueError("Config must be a JSON object")
                data.setdefault("prompt", EXTRACTION_PROMPT)        
                data.setdefault("allowed_classes", list(DEFAULT_ALLOWED_CLASSES))
                data.setdefault("examples", DEFAULT_EXTRACTION_EXAMPLES)
                return data
    except Exception:
        pass
    return get_default_config()


def save_extraction_config(data: Dict[str, Any], path: Optional[str] = None) -> None:
    cfg_path = path or get_config_path()
    cfg_dir = os.path.dirname(cfg_path)
    os.makedirs(cfg_dir, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
