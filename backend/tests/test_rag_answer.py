import pytest

from app.exception import DocumentNotFoundError, LLMError, LLMTimeoutError
from app.generation.prompt import REFUSAL_MESSAGE
from app.retrieval.in_memory_store import InMemoryStore
from app.services.rag_service import RAGService
from tests.fakes import FakeIngestion, FakeLLM

RELEVANT = "python programming language"    # matches the page-2 chunk (score ~0.70)
UNRELATED = "quantum chromodynamics gluons"  # matches nothing (score ~0.17)


def make_service(llm: FakeLLM, threshold: float = 0.4) -> tuple[RAGService, str]:
    service = RAGService(FakeIngestion(), InMemoryStore(), llm, similarity_threshold=threshold)
    return service, service.ingest("x.pdf").document_id


def test_happy_path_returns_validated_citation_with_page():
    llm = FakeLLM("Python is used for machine learning [1].")
    service, doc_id = make_service(llm)

    answer = service.answer(doc_id, RELEVANT)

    assert answer.grounded is True
    assert answer.text == "Python is used for machine learning [1]."
    assert [(c.ref, c.page) for c in answer.citations] == [(1, 2)]  # [1] = best chunk = page 2
    assert len(answer.retrieved) == 2


def test_prompt_sent_to_llm_contains_chunk_text_and_question():
    llm = FakeLLM()
    service, doc_id = make_service(llm)

    service.answer(doc_id, RELEVANT)

    assert len(llm.calls) == 1
    system, user = llm.calls[0]
    assert "ONLY the information" in system
    assert "Python is a popular programming language" in user
    assert RELEVANT in user


def test_below_threshold_refuses_without_calling_llm():
    llm = FakeLLM()
    service, doc_id = make_service(llm)

    answer = service.answer(doc_id, UNRELATED)

    assert llm.calls == []  # the whole point of the gate
    assert answer.text == REFUSAL_MESSAGE
    assert answer.grounded is False
    assert answer.citations == []
    assert len(answer.retrieved) == 2  # kept so callers can inspect the scores


def test_llm_refusal_is_normalised_to_the_standard_message():
    service, doc_id = make_service(FakeLLM(REFUSAL_MESSAGE + " Sorry about that!"))
    answer = service.answer(doc_id, RELEVANT)
    assert answer.text == REFUSAL_MESSAGE
    assert answer.grounded is False
    assert answer.citations == []


def test_answer_without_citations_is_flagged_not_grounded():
    service, doc_id = make_service(FakeLLM("Python is a language."))
    answer = service.answer(doc_id, RELEVANT)
    assert answer.text == "Python is a language."
    assert answer.grounded is False
    assert answer.citations == []


def test_invalid_citation_is_stripped_but_valid_one_kept():
    service, doc_id = make_service(FakeLLM("Python is great [9]. It is popular [1]."))
    answer = service.answer(doc_id, RELEVANT)
    assert answer.text == "Python is great. It is popular [1]."
    assert [c.ref for c in answer.citations] == [1]
    assert answer.grounded is True


def test_reply_that_is_only_an_invalid_marker_becomes_a_refusal():
    service, doc_id = make_service(FakeLLM("[9]"))
    answer = service.answer(doc_id, RELEVANT)
    assert answer.text == REFUSAL_MESSAGE
    assert answer.grounded is False


def test_llm_errors_propagate_to_the_caller():
    service, doc_id = make_service(FakeLLM(error=LLMTimeoutError("slow")))
    with pytest.raises(LLMTimeoutError):
        service.answer(doc_id, RELEVANT)

    service, doc_id = make_service(FakeLLM(error=LLMError("boom")))
    with pytest.raises(LLMError):
        service.answer(doc_id, RELEVANT)


def test_unknown_document_raises_before_any_llm_call():
    llm = FakeLLM()
    service, _ = make_service(llm)
    with pytest.raises(DocumentNotFoundError):
        service.answer("nope", RELEVANT)
    assert llm.calls == []