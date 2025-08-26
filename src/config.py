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

# Evaluation prompt template (used by relevance evaluators)
RELEVANCE_EVALUATION_PROMPT = (
    "You are an assistant that evaluates how relevant a paper is to a target research title. "
    "Return strictly a JSON object with keys: score (0-100 integer), kept (true/false), "
    "label (one of: 'keep','maybe','discard'), and rationale (short string). "
    "Be conservative: only 'keep' if clearly relevant.\n\n"
    "Research title: {query}\nPaper title: {title}\nAbstract: {abstract}\n\n"
    "Respond with only JSON, no extra text."
)

# Gemini model
GEMINI_MODEL: str = "gemini-2.5-flash"

# Extraction configuration
DEFAULT_ALLOWED_CLASSES = [
    "sentiment analysis business application/use case",
    "sentiment analysis tool/software",
]

EXTRACTION_PROMPT = (
    "Extract only these two classes using exact spans and order of appearance: "
    "1) sentiment analysis business application/use case, "
    "2) sentiment analysis tool/software. "
    "Return concise attributes when obvious (e.g., vendor, purpose)."
)

DEFAULT_EXTRACTION_EXAMPLES = [
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
