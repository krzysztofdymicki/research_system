import os
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # fallback if dotenv not installed, though it's in deps


def _load_env() -> None:
    # Load .env from project root if available
    if load_dotenv is not None:
        try:
            # Try current working directory first, then parent folders
            load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)
            # Also attempt from this file's parent (repo root)
            here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            load_dotenv(dotenv_path=os.path.join(here, ".env"), override=False)
        except Exception:
            pass


_load_env()


def _getenv(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None and v != "" else None


def getenv_str(name: str, default: str) -> str:
    v = _getenv(name)
    return v if v is not None else default


def getenv_bool(name: str, default: bool = False) -> bool:
    v = (_getenv(name) or "").strip().lower()
    if v in {"1", "true", "yes"}:
        return True
    if v in {"0", "false", "no"}:
        return False
    return default


def getenv_float(name: str, default: float) -> float:
    v = _getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def getenv_int(name: str, default: int) -> int:
    v = _getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default


# LLM settings (OpenAI-compatible)
LLM_ENDPOINT: str = getenv_str("RS_LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
LLM_MODEL: str = getenv_str("RS_LLM_MODEL", "google/gemma-3-12b")
LLM_TEMPERATURE: float = getenv_float("RS_LLM_TEMPERATURE", 0.2)
LLM_MAX_TOKENS: int = getenv_int("RS_LLM_MAX_TOKENS", 256)


# Expose other knobs for central access (optional)
CORE_API_KEY: Optional[str] = _getenv("CORE_API_KEY")
# Reset-on-start removed; use GUI button for manual reset
