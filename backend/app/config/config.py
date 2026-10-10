"""Application configuration."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
        )

    app_name: str = "DocMind"
    environment: str = "development"
    log_level: str = "INFO"

    # Embeddings / ingestion
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_upload_mb: int = 20

    # LLM (any OpenAI-compatible endpoint; Gemini by default)
    llm_provider: str = "google_genai"
    llm_model: str ="gemini-3.1-flash-lite"
    llm_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    llm_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        validation_alias="GEMINI_BASE_URL",
    )
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    # Generation guardrails
    similarity_threshold: float = 0.10  
    max_context_chars: int = 12_000


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor: Settings() is built once per process."""
    return Settings()