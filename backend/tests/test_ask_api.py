import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_rag_service
from app.exception import LLMError, LLMTimeoutError
from app.generation.prompt import REFUSAL_MESSAGE
from app.main import app
from app.retrieval.in_memory_store import InMemoryStore
from app.services.rag_service import RAGService
from tests.fakes import FakeIngestion, FakeLLM

RELEVANT = "python programming language"
UNRELATED = "quantum chromodynamics gluons"


@pytest.fixture
def make_client():
    """Factory: build a client around a given FakeLLM, upload one document, return both."""

    def _make(llm: FakeLLM) -> tuple[TestClient, str]:
        service = RAGService(FakeIngestion(), InMemoryStore(), llm, similarity_threshold=0.4)
        app.dependency_overrides[get_rag_service] = lambda: service
        client = TestClient(app)
        r = client.post(
            "/documents", files={"file": ("a.pdf", b"%PDF-1.4 fake", "application/pdf")}
        )
        return client, r.json()["document_id"]

    yield _make
    app.dependency_overrides.clear()


def ask(client: TestClient, doc_id: str, question: str = RELEVANT, **extra):
    return client.post(f"/documents/{doc_id}/ask", json={"question": question, **extra})


def test_ask_returns_answer_with_page_citation(make_client):
    client, doc_id = make_client(FakeLLM("Python is used for machine learning [1]."))

    r = ask(client, doc_id)

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Python is used for machine learning [1]."
    assert body["grounded"] is True
    assert body["question"] == RELEVANT
    assert [(c["ref"], c["page"]) for c in body["citations"]] == [(1, 2)]
    assert "Python is a popular" in body["citations"][0]["text"]


def test_refusal_is_a_200_with_grounded_false_and_no_llm_call(make_client):
    llm = FakeLLM()
    client, doc_id = make_client(llm)

    r = ask(client, doc_id, UNRELATED)

    assert r.status_code == 200  # a refusal is a valid answer, not an error
    assert r.json() == {
        "document_id": doc_id,
        "question": UNRELATED,
        "answer": REFUSAL_MESSAGE,
        "grounded": False,
        "citations": [],
    }
    assert llm.calls == []


def test_unknown_document_returns_404(make_client):
    client, _ = make_client(FakeLLM())
    assert ask(client, "nope").status_code == 404


def test_llm_error_returns_502_without_leaking_details(make_client):
    client, doc_id = make_client(FakeLLM(error=LLMError("secret-vendor-detail")))

    r = ask(client, doc_id)

    assert r.status_code == 502
    assert "secret-vendor-detail" not in r.text


def test_llm_timeout_returns_504(make_client):
    client, doc_id = make_client(FakeLLM(error=LLMTimeoutError("slow")))
    assert ask(client, doc_id).status_code == 504


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "x" * 1001}, {"question": "ok", "top_k": 0},
     {"question": "ok", "top_k": 21}, {}],
)
def test_invalid_requests_return_422(make_client, payload):
    client, doc_id = make_client(FakeLLM())
    assert client.post(f"/documents/{doc_id}/ask", json=payload).status_code == 422


def test_top_k_is_passed_through(make_client):
    client, doc_id = make_client(FakeLLM("Python [1]."))
    r = ask(client, doc_id, top_k=1)
    assert r.status_code == 200
    # one retrieved chunk -> only [1] exists in the prompt
    assert [c["ref"] for c in r.json()["citations"]] == [1]