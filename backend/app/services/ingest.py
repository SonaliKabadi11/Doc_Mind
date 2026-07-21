import numpy as np
import sys
import logging
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
path = Path(__file__).resolve().parents[1] / "data" / "HP1.pdf" 

logger = logging.getLogger(__name__)

from app.retreival.in_memory_store import InMemoryStore
from app.ingestion.base import Ingestion
from app.exception import CustomException

def ingest_document(document_id: str,  ingestion: Ingestion, store: InMemoryStore, file_path:str ):
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

    try:
        
        text = ingestion.extract_pdf(file_path)
        chunks = ingestion.chunk_text(text)
        embeddings = ingestion.embed_chunks(chunks)
        chunk_len = store.add_chunks(document_id, chunks, embeddings)

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion = Ingestion()
    store = InMemoryStore()
    document_id = 1
    ingest_document(document_id, ingestion, store, path)
    data = store.get_chunkDB()
    for i in data:
        print(i)
    
