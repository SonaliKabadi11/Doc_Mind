import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_rag_service
from app.main import app
from app.retrieval.in_memory_store import InMemoryStore
from app.services.rag_service import RAGService
from tests.fakes_ingestion import FakeIngestion


@pytest.fixture
def client():
    service = RAGService(FakeIngestion(), InMemoryStore())
    app.dependency_overrides[get_rag_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def _upload(client):
    return client.post(
        "/documents", files={"file": ("a.pdf", b"%PDF-1.4 fake", "application/pdf")}
    )


def test_upload_query_delete_flow(client):
    response = _upload(client)
    assert response.status_code == 201
    doc_id = response.json()["document_id"]

    response = client.post(
        f"/documents/{doc_id}/query", json={"question": "python language", "top_k": 1}
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["page"] == 2

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    response = client.post(f"/documents/{doc_id}/query", json={"question": "python"})
    assert response.status_code == 404


def test_upload_rejects_non_pdf(client):
    response = client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_query_validates_input(client):
    doc_id = _upload(client).json()["document_id"]
    assert client.post(f"/documents/{doc_id}/query", json={"question": ""}).status_code == 422
    assert (
        client.post(f"/documents/{doc_id}/query", json={"question": "x", "top_k": 100}).status_code
        == 422
    )