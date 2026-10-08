import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np

from app.exception import IngestionError
from app.models.models import Chunk, SearchResult, Answer
from app.retrieval.base import Retrieval
from app.generation.base import LLMProvider
from app.generation.citations import parse_citations
from app.generation.prompt import REFUSAL_MESSAGE, build_prompt, is_refusal


logger = logging.getLogger(__name__)

class Ingestor(Protocol):
    def extract_pages(self, source: str | Path | BinaryIO) -> list[tuple[int, str]]: ...
    def chunk_pages(self, document_id: str, pages: list[tuple[int, str]]) -> list[Chunk]: ...
    def embed_chunks(self, chunks: list[str]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...

@dataclass(frozen=True)
class IngestResult:
    document_id:str
    num_pages: int
    num_chunks: int

class RAGService:
    def __init__(self, ingestion:Ingestor, store: Retrieval, llm = LLMProvider, *, similarity_threshold : float = 0.30, max_context_chars : int = 12_000) -> None:
        self.ingestion = ingestion
        self.store = store
        self.llm = llm
        self.similarity_threshold = similarity_threshold 
        self.max_context_chars  = max_context_chars 

    def ingest(self, source: str | Path | BinaryIO, document_id: str | None = None) -> IngestResult:
        document_id = document_id or uuid.uuid4().hex
        try: 
            pages = self.ingestion.extract_pages(source)
            chunks = self.ingestion.chunk_pages(document_id, pages)
            if not chunks:
                raise IngestionError("No extractable text found (scanned PDF?)")
            embeddings = self.ingestion.embed_chunks([c.text for c in chunks])
            self.store.add_chunks(document_id, chunks, embeddings)
        except IngestionError:
            raise
        except Exception as e:
            logger.exception("Ingestion failed for coument %s", document_id)
            raise IngestionError(f"Failed to ingest document:, {e}") from e
        return IngestResult(document_id, num_pages=len(pages),num_chunks= len(chunks))

    def retrieve(self, document_id: str, query: str, top_k: int = 5) -> list[SearchResult]:
        query_embeddings = self.ingestion.embed_query(query)
        return self.store.search(document_id, query_embeddings, top_k)

    def answer(self, document_id:str, question:str, top_k:int = 5) -> Answer:
        """Answer a question from one document, with validated page citations.

        Raises DocumentNotFoundError, LLMTimeoutError, LLMError; they propagate to the
        API layer, which maps them to 404 / 504 / 502.
        """
        t0 = time.perf_counter()
        results = self.retrieve(document_id, question, top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        top_score = results[0].score if results else 0.0
        # Gate: weak retrieval means the answer is probably not in the document.
        # Refuse without calling the LLM: cheaper, faster, and cannot hallucinate.
        if not results or top_score < self.similarity_threshold:
            logger.info(
                "Refused without LLM call: top_score=%.3f < threshold=%.3f retrieval_ms=%.0f",
                top_score, self.similarity_threshold, retrieval_ms,
            )
            return self._refusal(results)

        prompt = build_prompt(question, results, self.max_context_chars)

        t1 = time.perf_counter()
        raw = self.llm.generate(prompt.system, prompt.user)
        llm_ms = (time.perf_counter() - t1) * 1000

        if is_refusal(raw):
            logger.info("LLM refused: top_score=%.3f llm_ms=%.0f", top_score, llm_ms)
            return self._refusal(results)

        parsed = parse_citations(raw, prompt.sources)

        if parsed.invalid_refs:
            logger.warning("LLM cited nonexistent sources: %s", parsed.invalid_refs)
        if not parsed.text:
            logger.warning("LLM reply was empty after citation cleanup")
            return self._refusal(results)

        grounded = bool(parsed.citations)
        if not grounded:
             logger.warning("LLM answer has no valid citations; marking grounded=False")
        logger.info(
            "Answered: top_score=%.3f sources=%d cited=%d grounded=%s "
            "retrieval_ms=%.0f llm_ms=%.0f",
            top_score, len(prompt.sources), len(parsed.citations), grounded,
            retrieval_ms, llm_ms,
        )
        return Answer(
            text=parsed.text,
            grounded = grounded,
            citations = parsed.citations,
            retrieved = results
        )




    def delete(self, document_id: str) -> int:
        return self.store.delete_document(document_id)

    @staticmethod
    def _refusal(results: list[SearchResult]) -> Answer:
        return Answer(text=REFUSAL_MESSAGE, grounded=False, citations=[], retrieved = results)