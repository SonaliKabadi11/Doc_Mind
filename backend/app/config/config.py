"""
Application Configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    app_name: str = "DocMind"
    environment: str = "development" 
    log_level: str = "INFO"



def get_settings() -> Settings:
    """Cached settings accessor — avoids re-reading env vars on every call."""
    return Settings()