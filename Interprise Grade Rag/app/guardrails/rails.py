import logfire
from langchain_openai import ChatOpenAI
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.gateway.client import get_langchain_llm
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant or Portkey gateway for intent classification.
    Safely logs a warning if keys are missing instead of crashing server startup.
    """
    global _rails

    try:
        if settings.OPENAI_API_KEY:
            guard_llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-5.5"
            )
        elif settings.PORTKEY_API_KEY:
            guard_llm = get_langchain_llm(feature="guardrails")
        else:
            logfire.warning("⚠️ Neither GROQ_API_KEY nor PORTKEY_API_KEY set — skipping guardrails initialization.")
            return

        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=YAML_CONTENT
        )

        _rails = LLMRails(config, llm=guard_llm)
        logfire.info("🛡️ NeMo Guardrails initialised.")
    except Exception as e:
        logfire.warning(f"⚠️ Guardrails initialization deferred/failed: {e}")
        _rails = None
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None