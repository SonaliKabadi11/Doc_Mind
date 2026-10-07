import logging
from collections.abc import Sequence

import numpy as np

from app.models import Chunk, SearchResult
from app.retrieval.base import DocumentNotFoundError, Retrieval

logger = logging.getLogger(__name__)


class InMemoryStore(Retrieval):
    def __init__(self):
        # Maps document_id -> dictionary of chunk/vector data
        self.chunk_db: dict[str, dict] = {}

    def add_chunks(self, document_id: str, chunks: Sequence[Chunk], embeddings: np.ndarray) -> int:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) out of sync"
            )
        self.chunk_db[document_id] = {
            "chunks": list(chunks),
            "embeddings": np.asarray(embeddings),
        }
        logger.info("Added %d chunks for document %s", len(chunks), document_id)
        return len(chunks)

    def search(
        self, document_id: str, query_embedding: np.ndarray, top_k: int
    ) -> list[SearchResult]:
        doc = self._get(document_id)
        query = np.asarray(query_embedding).reshape(-1)
        scores = doc["embeddings"] @ query  # cosine, since vectors are normalized
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
                SearchResult(
                    chunk=doc["chunks"][i],
                    score=float(scores[i]),
                    index=int(i),
                )
                for i in top_indices
            ]
        return results
      

    def delete_document(self, document_id: str) -> int:
       # no_of_chunks = len(self.chunk_db[document_id]["chunks"])
        no_of_chunks = len(self._get(document_id)["chunks"])
        del self.chunk_db[document_id]
        logger.info("Deleted %d chunks from document id %s", no_of_chunks, document_id)
        return no_of_chunks 
           
            
    def _get(self, document_id: str) -> dict:
        try:
            return self.chunk_db[document_id]
        except KeyError:
            raise DocumentNotFoundError(f"Document {document_id} not found") from None
            
    def get_chunks(self) -> list[Chunk]:
        return [chunk for document in self.chunk_db.values() for chunk in document["chunks"]]