import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np

from app.exception import IngestionError
from app.models.models import Chunk, SearchResult
from app.retrieval.base import Retrieval

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
    def __init__(self, ingestion:Ingestor, store: Retrieval) -> None:
        self.ingestion = ingestion
        self.store = store

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

    def delete(self, document_id: str) -> int:
        return self.store.delete_document(document_id)