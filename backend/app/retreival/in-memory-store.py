import logging
import sys

import numpy as np

from app.exception import CustomException
from app.retreival.base import DocumentNotFoundError, Retrieval, SearchResult

logger = logging.getLogger(__name__)


class InMemoryStore(Retrieval):
    def __init__(self):
        # Maps document_id -> dictionary of chunk/vector data
        self.chunk_db = {}

    def add_chunks(self, document_id: str, chunks: list[str], embeddings: np.ndarray) -> int:
        try:
            if len(chunks) != embeddings.shape[0]:
                raise CustomException(
                    f"Chunks length {len(chunks)} and embeddings shape "
                    f"{embeddings.shape[0]} not in sync",
                    sys,
                )
            self.chunk_db[document_id] = {
                "chunks": chunks,
                "embeddings": embeddings,
            }
            logger.info("Added %d chunks for document %s", len(chunks), document_id)
            return len(chunks)
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.exception("Failed to add chunks for document %s", document_id)
            raise CustomException(e, sys) from e

    def search(
        self, document_id: str, query_embedding: np.ndarray, top_k: int
    ) -> list[SearchResult]:
        try:
            if document_id not in self.chunk_db:
                raise DocumentNotFoundError(f"Document with id {document_id} not found!")

            doc_chunks = self.chunk_db[document_id]["chunks"]
            doc_embeddings = self.chunk_db[document_id]["embeddings"]  # (num_chunks, dim)

            # Accept either a 1D (dim,) or 2D (1, dim) query embedding.
            flat_query = np.asarray(query_embedding).reshape(-1)

            scores = np.dot(doc_embeddings, flat_query).flatten()  # (num_chunks,)

            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [
                SearchResult(
                    chunk_text=doc_chunks[i],
                    score=float(scores[i]),
                    index=int(i),
                )
                for i in top_indices
            ]
            return results
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.exception("Failed to search document %s", document_id)
            raise CustomException(e, sys) from e

    def delete_document(self, document_id: str) -> int:
        try:
            if document_id not in self.chunk_db:
                raise DocumentNotFoundError(f"Document with id {document_id} not found!")

            no_of_chunks = len(self.chunk_db[document_id]["chunks"])
            del self.chunk_db[document_id]
            logger.info("Deleted %d chunks from document id %s", no_of_chunks, document_id)
            return no_of_chunks
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.exception("Failed to delete document %s", document_id)
            raise CustomException(e, sys) from e