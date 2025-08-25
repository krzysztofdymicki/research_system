import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Essential API keys (must be set in .env)
CORE_API_KEY: Optional[str] = os.getenv("CORE_API_KEY")
CORE_API_URL: str = "https://api.core.ac.uk/v3/search/works"
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

# LMStudio settings (for optional local AI analysis)
LMSTUDIO_ENDPOINT: str = os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234/v1/chat/completions")
LMSTUDIO_MODEL: str = os.getenv("LMSTUDIO_MODEL", "google/gemma-3-12b")
LMSTUDIO_TEMPERATURE: float = 0.2
LMSTUDIO_MAX_TOKENS: int = 256

# Gemini model
GEMINI_MODEL: str = "gemini-2.5-flash"
