from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


class DocumentNotFoundError(Exception):
    """Raised when an operation references a document_id with no stored chunks."""


@dataclass
class SearchResult:
    chunk_text: str
    score: float
    index: int


class Retrieval(ABC):
    @abstractmethod
    def add_chunks(self, document_id: str, chunks: Sequence[str], embeddings: np.ndarray) -> int:
        """Adds the chunks and their embeddings to the store for a given document.

        Parameters
        ----------
        document_id: str
            id of the document these chunks belong to
        chunks: Sequence[str]
            chunk texts broken down from the document
        embeddings: np.ndarray
            normalized float array of shape (len(chunks), embedding_dimension)

        Returns
        -------
        int
            number of chunks stored

        Raises
        ------
        ValueError
            if len(chunks) does not match embeddings.shape[0]
        """
        ...

    @abstractmethod
    def search(
        self, document_id: str, query_embedding: np.ndarray, top_k: int
    ) -> List[SearchResult]:
        """Searches for the top_k chunks most similar to `query_embedding` within one document.

        Parameters
        ----------
        document_id: str
            id of the document to search within
        query_embedding: np.ndarray
            normalized embedding of the user's query, shape (embedding_dimension,)
        top_k: int
            number of most similar chunks to return

        Returns
        -------
        List[SearchResult]
            results sorted by score in descending order (most similar first)

        Raises
        ------
        DocumentNotFoundError
            if document_id has no stored chunks
        """
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Deletes all stored chunks for a document.

        Parameters
        ----------
        document_id: str
            id of the document to delete

        Returns
        -------
        int
            number of chunks deleted

        Raises
        ------
        DocumentNotFoundError
            if document_id has no stored chunks
        """
        ...