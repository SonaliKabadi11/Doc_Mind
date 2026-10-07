import pytest

from app.exception import DocumentNotFoundError, IngestionError
from app.retrieval.in_memory_store import InMemoryStore
from app.services.rag_service import RAGService
from tests.fakes_ingestion import EmptyIngestion, FakeIngestion


@pytest.fixture
def service() -> RAGService:
    return RAGService(FakeIngestion(), InMemoryStore())


def test_ingest_returns_counts_and_stores_chunks(service):
    result = service.ingest(source="ignored.pdf")
    assert result.num_pages == 2
    assert result.num_chunks == 2
    assert result.document_id in service.store.chunk_db


def test_retrieve_returns_most_relevant_chunk_with_page(service):
    doc_id = service.ingest("ignored.pdf").document_id
    results = service.retrieve(doc_id, "python programming language", top_k=2)
    assert results[0].chunk.page == 2
    assert results[0].score > results[1].score


def test_retrieve_unknown_document_raises(service):
    with pytest.raises(DocumentNotFoundError):
        service.retrieve("nope", "anything")


def test_ingest_without_text_raises():
    service = RAGService(EmptyIngestion(), InMemoryStore())
    with pytest.raises(IngestionError):
        service.ingest("scanned.pdf")


def test_delete_removes_document(service):
    doc_id = service.ingest("ignored.pdf").document_id
    assert service.delete(doc_id) == 2
    with pytest.raises(DocumentNotFoundError):
        service.retrieve(doc_id, "cats")