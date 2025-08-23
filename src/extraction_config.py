import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_ALLOWED_CLASSES = [
    "sentiment analysis business application/use case",
    "sentiment analysis tool/software",
]


def _default_examples() -> List[Dict[str, Any]]:
    return [
        {
            "text": "We deployed a SensiStrength - sentiment analysis tool in our CRM workflow to flag negative customer feedback in online marketing.",
            "extractions": [
                {
                    "extraction_class": "sentiment analysis business application/use case",
                    "extraction_text": "flag customer feedback",
                },
                {
                    "extraction_class": "sentiment analysis tool/software",
                    "extraction_text": "SensiStrength",
                    "attributes": {
                        "business sector": "online marketing"
                    },
                },
            ],
        }
    ]


def get_default_config() -> Dict[str, Any]:
    return {
        "prompt": (
            "Extract only these two classes using exact spans and order of appearance: 1) sentiment analysis business application/use case, 2) sentiment analysis tool/software. Return concise attributes when obvious (e.g., vendor, purpose)."
        ),
        "allowed_classes": list(DEFAULT_ALLOWED_CLASSES),
        "examples": _default_examples(),
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
                data.setdefault("prompt", get_default_config()["prompt"])        
                data.setdefault("allowed_classes", list(DEFAULT_ALLOWED_CLASSES))
                data.setdefault("examples", _default_examples())
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
