import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # NOTE: renamed from GROK_* -> GROQ_* to match the provider name used
    # everywhere else in the codebase (ChatGroq, GROQ_MODEL, etc).
    # Update your .env keys to GROQ_API_KEY / GROQ_FALLBACK_API_KEY.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_FALLBACK_API_KEY = os.getenv("OPENAI_FALLBACK_API_KEY")
    OPENAI_MODEL = ("gpt-5.5")

    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY") or os.getenv("PORTKEY_API")
    GPT_SLUG = "rag1"
    GPT_SLUG_2 = "rag2"



settings = Settings()


def validate_settings() -> None:
    """
    Fail fast at startup instead of deep inside a probe/API call with a
    confusing stack trace. Call this from main.py's startup event.
    """
    required = {
        "GEMINI_API_KEY": settings.GEMINI_API_KEY,
        "QDRANT_URL": settings.QDRANT_URL,
        "QDRANT_API_KEY": settings.QDRANT_API_KEY,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Check your .env file."
        )