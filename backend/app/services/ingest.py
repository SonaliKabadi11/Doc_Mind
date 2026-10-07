import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

from app.ingestion.base import Ingestion  # noqa: E402
from app.retrieval.in_memory_store import InMemoryStore  # noqa: E402
from app.services.rag_service import RAGService  # noqa: E402


def ingest_document(
    document_id: str,
    ingestion: Ingestion,
    store: InMemoryStore,
    file_path: str,
) -> int:
    """
    Stores the chunks and embeddings in given document_id

    Parameters
    ----------
    document_id: str
        id of the document these chunks belong to
    ingestion: Ingestion 
        Instance of model
    store: InMemoryStore
        Instance to store the chunks and embeddings
    file_path: str
        location str of the file

    """

    result = RAGService(ingestion, store).ingest(file_path, document_id=document_id)
    return result.num_chunks
        

