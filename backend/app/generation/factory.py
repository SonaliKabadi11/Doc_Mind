from app.config.config import Settings
from app.generation.base import LLMProvider
from app.generation.openai_compatible import OpenAICompatibleProvider

def build_llm_provider(settings:Settings) -> LLMProvider:
    """Create the LLM provider from settings.

    Raises ValueError if the API key is missing, so a misconfigured server fails at
    startup instead of on the first user request.
    """
    return OpenAICompatibleProvider(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )