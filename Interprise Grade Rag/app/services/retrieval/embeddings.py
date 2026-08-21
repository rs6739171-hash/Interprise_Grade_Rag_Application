import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

BATCH_SIZE = 50
GEMINI_DIM = 3072
FALLBACK_DIM = 768

active_model = None
model_type: str | None = None
# Model Initialization

def probe_gemini():
    """Try one embed call to verify Gemini is reachable. Return model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        model.embed_query("probe")
        logfire.info("Gemini Embeddings are active", dims=3072, model="google/gemini-embedding-2-preview")
        return model

    except Exception as e:
        logfire.warning(f"Gemini Probe Failed: {e}. We will use sentence transformers fallback.")
        return None

def load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")

def init():
    """Initialize embedding model once per process. Called lazily on first use."""
    global active_model, model_type
    if active_model is not None:
        return
    gemini = probe_gemini()
    if gemini:
        active_model = gemini
        model_type = "gemini"
    else:
        active_model = load_fallback()
        model_type = "fallback"

# Public Helpers
def get_embedding_dim() -> int:
    """Return the vector dim of the active model. Calls init() if needed."""
    init()
    if model_type == "gemini":
        return GEMINI_DIM
    else:
        return FALLBACK_DIM

# Batch embedding with retry
def embed_batch(batch: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts with exponential backoff retry. Safe for large batches."""
    if model_type == "gemini":
        # exponential backoff, 4 attempts
        for attempt in range(4):
            try:
                return active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resources_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Gemini rate limit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")
    else:
        return active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    init()
    if model_type == "gemini":
        return active_model.embed_query(query)
    return active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        with logfire.span("Embed batch", model=model_type, start=i, size=len(batch)):
            all_embeddings.extend(embed_batch(batch))
    return all_embeddings