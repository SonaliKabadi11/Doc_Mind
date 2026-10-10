import pytest
from fastapi.testclient import TestClient

import app.ingestion.base as ingestion_module
import app.main as main_module
from app.config.config import Settings
from app.generation.openai_compatible import OpenAICompatibleProvider
from app.main import app
from tests.fakes import FakeIngestion


@pytest.fixture(autouse=True)
def no_real_embedding_model(monkeypatch):
    """Lifespan must never download or load the real embedding model in tests."""
    monkeypatch.setattr(ingestion_module, "Ingestion", lambda **kwargs: FakeIngestion())


def test_lifespan_wires_service_from_settings(monkeypatch):
    settings = Settings(
        GOOGLE_API_KEY="dummy", similarity_threshold=0.5, max_context_chars=777, _env_file=None
    )
    monkeypatch.setattr(main_module, "settings", settings)

    with TestClient(app) as client:  # `with` => lifespan runs
        service = client.app.state.rag_service

    assert isinstance(service.llm, OpenAICompatibleProvider)
    assert service.similarity_threshold == 0.5
    assert service.max_context_chars == 777


def test_startup_fails_fast_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "settings", Settings(_env_file=None))

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        with TestClient(app):
            pass