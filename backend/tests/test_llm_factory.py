import pytest

from app.config.config import Settings
from app.generation.factory import build_llm_provider
from app.generation.openai_compatible import OpenAICompatibleProvider


def test_factory_builds_provider_from_settings():
    settings = Settings(GOOGLE_API_KEY="dummy", llm_model="some-model", _env_file=None)
    provider = build_llm_provider(settings)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model == "some-model"


def test_factory_fails_fast_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        build_llm_provider(settings)